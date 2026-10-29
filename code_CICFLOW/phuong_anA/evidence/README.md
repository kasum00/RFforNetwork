# Frozen CICFLOW split evidence

`cicflow_split_evidence.json` stores gzip/base64 byte-exact snapshots of these existing repository files:

- `STAGE29_MATERIALIZATION_V2/RUN_ACCEPT_4578442/v2_materialization_validation.json`
- `STAGE29_MATERIALIZATION_V2/RUN_ACCEPT_4578442/v2_preflight_integrity.json`
- `STAGE29_MATERIALIZATION_V2/RUN_ACCEPT_4578442/revised_materialization_manifest.csv`
- `STAGE29_MATERIALIZATION_V2/RUN_ACCEPT_4578442/.stage29_v2_binding.json`
- `code_CICFLOW/stage29_v2_allocation.py`, the frozen proxy-label producer, as evidence only; it is never executed by A1.

These files were read and snapshotted, never modified. No large provenance CSV, raw flow data or Parquet is embedded. Origin relative path, original byte count and SHA256 are included. Each config's own `.stage3_preflight.json` must name/hash the four parent manifests; the frozen V2 binding in turn names the producer hash. Copying this snapshot to another dataset/split cannot satisfy those bindings.

The source assigns0 to category BENIGN and1 to other folders, explicitly PROVISIONAL_FOLDER. This explains the existing proxy mapping; it establishes no independent binary ground truth. Capture/session and near duplicates remain NOT_VERIFIED. A1 checks the archived row-membership/embargo/exact-duplicate attestations, rank-range separation and current Train/Val file hashes; it does not rerun the historical raw-flow audit.

For a different verified CICFLOW lineage, supply `EVIDENCE_ROOT` containing its original basename files plus its frozen `stage29_v2_allocation.py`. Do not edit, regenerate or relabel manifests to force a match. Colab only decompresses/reads this evidence; it never runs the archived source.
