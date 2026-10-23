"""Verify the origin run without changing its binding or completed artifacts."""
from __future__ import annotations

import ast
import copy
import hashlib
import os
import re
import subprocess
import uuid
from pathlib import Path

from colab.svm_resumable_store import CheckpointStore, SCHEMA, digest, file_sha, read_json, verified_copy
from .paths import CONFIG, ORIGIN_COMMIT, RUN_NAME


class MigrationBlocked(ValueError):
    pass


def blocked(message):
    raise MigrationBlocked("MIGRATION_BLOCKED: " + message)


def normalized_bytes(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def git_blob(repo: Path, relative: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(repo), "show", f"{ORIGIN_COMMIT}:{relative}"],
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        blocked(f"Origin Git blob unavailable: {relative}; fetch full history manually. {exc}")


def verify_engine_bridge(old_source: str, new_source: str) -> None:
    """Allow only extraction of the original body and three infrastructure adapters."""
    old, new = ast.parse(old_source), ast.parse(new_source)
    def find(tree, name):
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        if len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef):
            blocked(f"Exactly one synchronous definition required: {name}")
        return nodes[0]
    original, extracted = find(old, "run_resumable"), find(new, "_execute_resumable")
    expected = copy.deepcopy(original.body[1:])

    class Adapters(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in ("SessionDeadline", "DurableProgress"):
                factory = "deadline_factory" if node.func.id == "SessionDeadline" else "progress_factory"
                node.func = ast.BoolOp(op=ast.Or(), values=[ast.Name(id=factory, ctx=ast.Load()), node.func])
            elif isinstance(node.func, ast.Name) and node.func.id == "manifest_root_for":
                return ast.BoolOp(op=ast.Or(), values=[ast.Name(id="manifest_root", ctx=ast.Load()), node])
            return node

    dump = lambda nodes: ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    expected_function = copy.deepcopy(original)
    expected_function.name = "_execute_resumable"
    expected_function.body = [Adapters().visit(n) for n in expected]
    expected_function.args.args[-1].arg = "context"
    expected_function.args.kwonlyargs = [ast.arg(arg=name) for name in (
        "manifest_root", "progress_factory", "deadline_factory")]
    expected_function.args.kw_defaults = [ast.Constant(value=None) for _ in range(3)]
    if ast.dump(expected_function, include_attributes=False) != ast.dump(extracted, include_attributes=False):
        blocked("Shared execution body changed beyond the explicit adapter extraction")
    retained = lambda tree: [n for n in tree.body if not (isinstance(n, ast.FunctionDef)
        and n.name in ("run_resumable", "run_windows_resumable", "_execute_resumable"))]
    if dump(retained(old)) != dump(retained(new)):
        blocked("A pre-existing resumable helper/class/import changed")
    wrapper = ast.parse("def f():\n context = validate_authorization(authorization)\n"
                        " return _execute_resumable(repo, windows, local_runs, context)\n").body[0]
    actual_wrapper = find(new, "run_resumable")
    if (dump(actual_wrapper.body) != dump(wrapper.body)
            or ast.dump(actual_wrapper.args, include_attributes=False) != ast.dump(original.args, include_attributes=False)
            or actual_wrapper.decorator_list or ast.dump(actual_wrapper.returns or ast.Constant(None)) != ast.dump(original.returns or ast.Constant(None))):
        blocked("Legacy Colab authorization wrapper changed")
    local_entry = find(new, "run_windows_resumable")
    if (local_entry.decorator_list
            or ast.dump(local_entry.args, include_attributes=False) != ast.dump(original.args, include_attributes=False)
            or ast.dump(local_entry.returns or ast.Constant(None)) != ast.dump(original.returns or ast.Constant(None))):
        blocked("Windows entry point signature/decorators changed")


def verify_sources(repo: Path, binding: dict) -> dict:
    identity = binding["identity"]
    if identity.get("git_commit") != ORIGIN_COMMIT:
        blocked("Unsupported checkpoint origin commit")
    for name, key in {"huyen_code/ciciomt24_training.py": "training_code_sha256",
                      "huyen_code/ciciomt24_sliding_window.py": "sliding_code_sha256"}.items():
        expected = normalized_bytes(git_blob(repo, name))
        if identity.get(key) != expected or normalized_bytes((repo / name).read_bytes()) != expected:
            blocked(f"Scientific source mismatch: {name}")
    required = {"colab/svm_resumable_store.py", "colab/svm_resumable_authorization.py",
                "colab/svm_resumable_execution.py", "colab/svm_resumable_training.py"}
    if set(binding.get("resumable_source_hashes", {})) != required:
        blocked("Incomplete origin resumable source identity")
    for name, value in binding["resumable_source_hashes"].items():
        origin = git_blob(repo, name)
        if normalized_bytes(origin) != value:
            blocked(f"Origin blob disagrees with checkpoint: {name}")
        current = (repo / name).read_bytes()
        if name == "colab/svm_resumable_training.py":
            verify_engine_bridge(origin.decode("utf-8"), current.decode("utf-8"))
        elif normalized_bytes(current) != value:
            blocked(f"Unapproved infrastructure change: {name}")
    for name in ("colab/experiment_launcher.py", "colab/dataset_validator.py",
                 "colab/setup_colab.py", "colab/svm_run_authorization.py",
                 "colab/svm_official_supervisor.py", "colab/svm_official_readiness.py"):
        if normalized_bytes((repo / name).read_bytes()) != normalized_bytes(git_blob(repo, name)):
            blocked(f"Origin dependency changed: {name}")
    return {"origin_commit": ORIGIN_COMMIT, "scientific_source": "EXACT_NORMALIZED_BYTES",
            "engine": "AST_VERIFIED_ADAPTER_EXTRACTION", "origin_binding_rewritten": False}


def scan_checkpoints(root: Path, *, expected_names=None, expected_contract=None) -> dict:
    """No pickle loading/creation/k=1 fallback. Any corrupt completed stage blocks."""
    if not root.is_dir() or not (root / "run_manifest.json").is_file():
        blocked("Existing checkpoint root/run_manifest.json required")
    manifest = read_json(root / "run_manifest.json")
    binding = manifest.get("binding")
    if (manifest.get("schema") != SCHEMA or not isinstance(binding, dict)
            or manifest.get("binding_sha256") != digest(binding)):
        blocked("Invalid run manifest/binding hash")
    identity = binding.get("identity", {})
    if (identity.get("experiment"), identity.get("config"), identity.get("run_name"), identity.get("git_commit")) != (
            "mi_rfe_svm", CONFIG, RUN_NAME, ORIGIN_COMMIT):
        blocked("Wrong experiment/configuration/run/origin commit")
    features = binding.get("feature_names")
    if not isinstance(features, list) or len(features) != 380 or len(set(features)) != 380:
        blocked("Origin 380-feature schema missing")
    if expected_names is not None and features != list(expected_names):
        blocked("Canonical feature order differs")
    if expected_contract is not None and binding.get("scientific_contract") != expected_contract:
        blocked("Scientific hyperparameters differ")
    if binding.get("rows_by_split") != {"train": 673559, "val": 224373, "test": 224287}:
        blocked("Full dataset split counts differ")
    if (binding.get("seed") != 42 or binding.get("preprocessing") !=
            "canonical float32 Inf/NaN->0; Train-only MI and Top100 scalers"):
        blocked("Origin seed/preprocessing identity differs")
    for path in [root, *root.rglob("*")]:
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 1024:
            blocked(f"Symlink/junction/reparse point forbidden: {path}")
    cp = root / "checkpoints"
    if not cp.is_dir():
        blocked("MI/RFE checkpoint tree absent; no recomputation allowed")
    store = CheckpointStore(root, binding, resume=True)
    incomplete = []
    allowed = {"data_prepared", "mi", "rfe", "selection", "final", "final_test"}
    for directory in cp.iterdir():
        if ".incomplete-" in directory.name and directory.is_dir():
            incomplete.append(directory.relative_to(cp).as_posix())
            continue
        if directory.name == "k_search" and directory.is_dir():
            for kd in directory.iterdir():
                if ".incomplete-" in kd.name and kd.is_dir():
                    incomplete.append(kd.relative_to(cp).as_posix())
                elif not kd.is_dir() or not re.fullmatch(r"k_(00[1-9]|0[1-9][0-9]|100)", kd.name):
                    blocked(f"Unexpected k directory: {kd.name}")
                else:
                    store.load_required("k_search/" + kd.name)
        elif directory.name not in allowed or not directory.is_dir():
            blocked(f"Unknown checkpoint entry: {directory.name}")
        else:
            store.load_required(directory.name)
    data = store.data("data_prepared")
    if (data.get("train_rows"), data.get("val_rows"), data.get("feature_names"), data.get("test_feature_rows_read")) != (
            673559, 224373, features, 0):
        blocked("Prepared-data metadata differs")
    if data.get("label_counts") != {"train": {"benign": 1575, "attack": 671984},
                                    "val": {"benign": 505, "attack": 223868}}:
        blocked("Origin prepared-data proxy label counts differ")
    mi = store.data("mi", parents={"data": store.reference("data_prepared")})
    names, indices = mi.get("features"), mi.get("indices")
    if (not isinstance(names, list) or len(names) != 100 or len(set(names)) != 100
            or not set(names) <= set(features) or mi.get("fit_role") != "train"
            or indices != [features.index(n) for n in names]):
        blocked("Invalid MI order/preprocessing metadata")
    if set(store.load_required("mi")["files"]) != {"data.json", "mi.joblib", "top100_scaler.joblib", "mi_feature_scores.csv"}:
        blocked("MI preprocessing artifacts missing or unexpected")
    mi_ref = store.reference("mi")
    rfe = store.data("rfe", parents={"mi": mi_ref})
    ranked = rfe.get("features")
    if (not isinstance(ranked, list) or len(ranked) != 100 or len(set(ranked)) != 100
            or set(ranked) != set(names) or rfe.get("complete_ranking") is not True or rfe.get("forest_fits") != 100):
        blocked("RFE ranking incomplete/inconsistent")
    if set(store.load_required("rfe")["files"]) != {"data.json", "rfe_feature_ranking.csv"}:
        blocked("RFE artifacts missing or unexpected")
    parents = {"mi": mi_ref, "rfe": store.reference("rfe")}
    rows = store.completed_k(parents, ranked)
    if not rows:
        blocked("No completed k; migration must never start again at k=1")
    for row in rows:
        if (row.get("train_rows"), row.get("validation_rows")) != (673559, 224373):
            blocked(f"k={row['k']} used different sample counts")
    from colab.svm_resumable_training import choose_best, verified_selection
    if store.load("selection") is not None:
        best = choose_best(rows)
        frozen = store.data("selection", parents={**parents, "k_history": digest(rows)})
        if frozen.get("best_k") != best["k"] or frozen.get("features") != best["features"]:
            blocked("Frozen selection differs from all 100 validation results")
    final = store.load("final")
    if final is not None:
        frozen = verified_selection(store)
        meta = store.data("final")
        if (set(final["files"]) != {"data.json", "final_model.joblib", "final_scaler.joblib"}
                or meta.get("fit_role") != "train+val" or meta.get("fit_rows") != 897932
                or meta.get("mi_rfe_refit") is not False or meta.get("features") != frozen["features"]):
            blocked("Invalid final model/scaler checkpoint")
    final_test = store.load("final_test")
    claim = root / "results" / "test_evaluation_claim.json"
    if claim.exists() and final_test is None:
        blocked("Test claimed without a completed result: manual review, no retry")
    if final_test is not None:
        verified_selection(store)
        store.load_required("final_test", parents={"final": store.reference("final")})
        saved = read_json(claim)
        if (saved.get("binding_sha256"), saved.get("final_checkpoint_sha256"), saved.get("test_evaluation_number")) != (
                store.binding_sha256, store.reference("final"), 1):
            blocked("Final Test claim mismatch")
    hashes = {p.parent.relative_to(cp).as_posix(): file_sha(p) for p in cp.rglob("checkpoint.json")
              if not any(".incomplete-" in part for part in p.relative_to(cp).parts)}
    return {"binding": binding, "binding_sha256": store.binding_sha256,
            "run_manifest_sha256": file_sha(root / "run_manifest.json"), "checkpoint_hashes": hashes,
            "completed_k": len(rows), "next_k": len(rows) + 1 if len(rows) < 100 else None,
            "next_stage": "publication" if final_test else "final_test" if final
            else "final_fit" if len(rows) == 100 else f"k={len(rows)+1}",
            "incomplete_directories_preserved_ignored": sorted(incomplete),
            "mi": "PASS_REUSE", "rfe": "PASS_REUSE", "final_test_completed": bool(final_test)}


def tree_files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha(p)}
            for p in root.rglob("*") if p.is_file()}


def copy_checkpoint_tree(source: Path, destination: Path) -> dict:
    """Optional manual import; already-installed checkpoints need no copy."""
    if not source.is_absolute() or not destination.is_absolute():
        blocked("Absolute import paths required")
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        blocked("Import target must be absent and disjoint from source")
    scan_checkpoints(source)
    before = tree_files(source)
    temporary = destination.with_name(destination.name + ".import-incomplete-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True, exist_ok=False)
    for relative in before:
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        verified_copy(source / relative, target)
    for directory in source.rglob("*"):
        if directory.is_dir():
            (temporary / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)
    if tree_files(source) != before or tree_files(temporary) != before:
        blocked("Imported bytes differ or source changed; partial copy retained")
    scan_checkpoints(temporary)
    os.rename(temporary, destination)
    return {"status": "IMPORTED_BYTES_VERIFIED", "files": len(before), "destination": str(destination)}
