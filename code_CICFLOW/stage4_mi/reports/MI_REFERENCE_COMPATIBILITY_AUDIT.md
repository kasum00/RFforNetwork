# Stage 4 MI — Reference Compatibility Audit

## Audit scope

Audit target:

`D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi`

Requested comparison targets:

1. The five CTU-13 reference scripts.
2. Existing MI-RFE feature-selection modules.
3. The completed/provided Stage 4 run:
   `FW256_FS8/RUN_MI_20261027_01`.

The audit was performed as a code/provenance review only. No Python,
PowerShell, unit test, dataset scan, MI fit, training, or Computer Use action
was executed. Existing Stage 0–3 artifacts and the completed run were not
modified, overwritten, or deleted.

## Executive decision

**Reference-compatible: NOT ESTABLISHED.**

The current implementation must not be labelled as reproducing the CTU-13
reference. The implementation has several choices that can change MI scores
and therefore feature ranking relative to one call over the complete 380
feature matrix. These differences are material enough that a matching score or
ranking cannot be inferred from the current code alone.

No scientific parameter was changed as part of this audit. This is
intentional: changing `StandardScaler`, seed handling, batch semantics, or
`mutual_info_classif` invocation merely to make an unverified comparison pass
would invalidate the audit.

## Implementation-side findings

The following findings are taken from the current Stage 4 implementation.

| Item | Current Stage 4 behavior | Scientific consequence |
|---|---|---|
| Feature values | Train values only; Validation/Test values are not loaded | Correct isolation for the MI stage |
| Numeric conversion | `np.asarray(X, dtype=float64)`, then `NaN/+Inf/-Inf → 0`, then `float32` | Must match the reference's coercion, imputation, precision, and invalid-value policy |
| Row handling | No row dropping | Compatible only if the reference also preserves every Train row |
| Label handling | Labels are copied unchanged; exactly two classes are required | Must match reference label encoding and class validation |
| Explicit scaler | No external `StandardScaler`; manifest records `scaler: none` | Potential mismatch if reference applies `StandardScaler` before MI |
| sklearn MI call | `mutual_info_classif(X[:, [index]], y, discrete_features=False, n_neighbors=3, random_state=42 + index)` | This is 380 one-feature calls, not one call on `X[:, :380]` |
| `discrete_features` | Explicitly `False` | Compatible only if every statistical feature is treated as continuous by reference |
| `n_neighbors` | Explicitly `3` | Any reference value other than 3 changes scores |
| Random state | Base seed `42`; seed is changed per original feature | Not equivalent to a single full-matrix call with `random_state=42` |
| Batch size | Default 20; batching is only checkpoint grouping | It does not make the MI call a 20-feature call; each feature is still fitted separately |
| Score ranking | Descending score; exact ties use original feature index ascending | May differ from reference's sort stability or tie direction |
| Top-K | First 100 after the local ranking | Top-100 identity is only comparable after score and tie semantics match |

## Most important incompatibility: one call versus 380 calls

The current code does not invoke:

```python
mutual_info_classif(
    X_train,
    y_train,
    discrete_features=False,
    n_neighbors=3,
    random_state=42,
)
```

Instead, it invokes `mutual_info_classif` once per feature, with a different
seed for each feature. The checkpoint batch of 20 does not alter that fact.

Even when all other arguments are equal, these procedures are not guaranteed
to produce the same values. For continuous variables, scikit-learn's MI
implementation uses the random state in its numerical-noise preparation. A
single full-matrix call consumes randomness across all continuous columns from
one RNG stream; the current implementation starts 380 independent calls with
380 different seeds. Consequently:

- exact MI scores can differ;
- close scores can change order;
- ties can be created or removed by floating-point differences;
- the Top-100 set can differ, especially near rank 100.

This is a deterministic implementation, but deterministic does not mean
reference-equivalent. It is deterministic with respect to the current
per-feature seed rule.

## StandardScaler assessment

The current Stage 4 contract intentionally has no explicit
`StandardScaler`. That is not sufficient to conclude compatibility:

- If the CTU-13 reference scales with `StandardScaler` before MI, the current
  pipeline differs.
- If the reference does not scale, adding a scaler would be an unjustified
  scientific change.
- `mutual_info_classif` itself performs internal numerical preparation for
  continuous features; that internal behavior must not be conflated with an
  externally fitted `StandardScaler`.
- A scaler fitted on Train only would preserve split isolation, but it could
  still alter finite-precision behavior and therefore scores near ties.

The reference's actual scaler placement, `with_mean`, `with_std`, dtype, and
fit scope must be read from the five CTU-13 scripts/modules before changing
the contract.

## Reference evidence status

| Evidence item | Status | Meaning |
|---|---|---|
| Five CTU-13 script contents | `NOT VERIFIED` | No claim is made about their exact preprocessing or MI invocation |
| Existing MI-RFE module contents | `NOT VERIFIED` | No claim is made about inherited defaults or helper behavior |
| Reference MI score artifact for this exact Train matrix | `NOT PROVIDED` | Numerical equivalence cannot be tested |
| Reference feature ranking/top-100 artifact | `NOT PROVIDED` | Ranking equivalence cannot be tested |
| Stage 4 implementation behavior | `REVIEWED` | Findings above are based on the current Stage 4 code contract |
| `FW256_FS8/RUN_MI_20261027_01` preservation | `REQUIRED` | Do not overwrite or relabel the run |

Until the first three evidence items are available, the correct status is
`PROVISIONAL_REFERENCE_UNVERIFIED`, not `PASS`.

## Impact on `FW256_FS8/RUN_MI_20261027_01`

The run must be treated as an immutable provisional artifact. This audit does
not alter it and does not convert it to an official result.

If later reference inspection establishes that the reference uses one full
380-feature call with a shared seed, the existing run is not numerically
comparable by default. Its scores/ranking should then be retained for
provenance, while a separate run ID should be created for the reference
variant. The old run must not be overwritten.

If reference inspection establishes the same per-feature seed rule,
preprocessing, and ranking semantics, a numerical regression against a saved
reference score artifact is still required before promotion.

## Required reference reconciliation before any code change

Read and record, for each of the five CTU-13 scripts and each relevant MI-RFE
module:

1. Input feature matrix construction and column order.
2. Train-only preprocessing, including invalid-value handling and dtype.
3. Whether `StandardScaler` exists, where it is fitted, and its parameters.
4. Exact `mutual_info_classif` call signature.
5. `discrete_features` value or inferred default.
6. `n_neighbors` value or inferred default.
7. Random-state value, object lifecycle, and whether one RNG is shared.
8. Whether MI is called once on all 380 columns or in feature/batch subsets.
9. Score serialization precision.
10. Ranking sort algorithm, stability, direction, and tie-breaking.
11. Top-K extraction and whether ranking is recomputed after filtering.

Only after these facts are reconciled should a code change be proposed. Any
change must be recorded as a new scientific variant with a new run ID.

## Test commands for the user

These commands are provided for user-controlled execution only. They were not
run during this audit.

### Existing Stage 4 unit tests

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE\code_CICFLOW'
python -m unittest discover -s 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\tests' -p 'test_*.py' -v
```

### Artifact-only verification of the existing run

This verifies result artifacts without fitting MI or reading dataset values:

```powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$env:PYTHONPATH = 'D:\COMPACT-MI-RFE\code_CICFLOW'
python -m stage4_mi.mi_selection --mode verify --run-dir 'D:\COMPACT-MI-RFE\code_CICFLOW\stage4_mi\runs\FW256_FS8\RUN_MI_20261027_01'
```

### Reference numerical regression — only after reference artifacts exist

There is currently no valid reference regression command because the exact
reference score/ranking artifact has not been supplied. Once it exists, the
regression must compare, without tolerance-based promotion hiding an exact
semantic mismatch:

- all 380 feature names in order;
- all 380 scores, with an explicitly documented numeric tolerance;
- ranking indices and names;
- Top-100 set and order;
- preprocessing fingerprint and MI parameters;
- source Train identity.

The regression must use a new output directory and must not rewrite
`FW256_FS8/RUN_MI_20261027_01`.

## Final audit status

`NOT REFERENCE-COMPATIBLE / NOT YET PROVEN`

The current implementation is technically deterministic and Train-isolated,
but the per-feature MI calls, per-feature seed rule, lack of explicit scaler,
and tie/ranking semantics are not proven to match the CTU-13 reference. No
scientific parameters were changed in this audit.

