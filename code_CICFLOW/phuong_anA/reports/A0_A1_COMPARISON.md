# A0 / A1 comparison — no benchmark executed

| Property | A0 | A1 |
|---|---|---|
| MI | Existing single380-column Train MI, StandardScaler, seed42, Top100 | Same function/contract |
| RF-RFE estimator |100 trees, depth15, balanced, seed42, n_jobs=-1 | Unchanged |
| RFE step / target |1 /1 |5 /1 |
| Theoretical elimination rounds,100 inputs |99 |20, final removal contains4 |
| Theoretical RF fits including final1-feature fit |100 |21 |
| sklearn rank values |100 unique |21 groups; many ties |
| Top-k order |Ascending unique rank |Rank ascending, removal importance descending, MI position ascending |
| SVM k |Every1..100 |Every1..100 |
| Best-k rule |Validation Accuracy; ties smaller k |Unchanged |
| Current task endpoint |Reference comparison only |Validation selection only; Test0 |
| Durable operation |A0 unchanged |Stage/RFE-step/k checkpoints and periodic measured heartbeat |

The fit counts are arithmetic consequences of step/target, **not measured runtime or a speedup factor**. Removing five together can change later importances, survivor, ranking, best-k and metrics. Within-group importance ordering is an explicit experimental policy and does not reconstruct A0's sequential elimination.

| Measured quantity | A0 | A1 |
|---|---|---|
| RFE runtime |NOT_PROVIDED |NOT_RUN |
| SVM search runtime |NOT_PROVIDED |NOT_RUN |
| Peak RAM |NOT_PROVIDED |NOT_RUN |
| Best k / Validation metrics |NOT_PROVIDED |NOT_RUN |
| Test metrics |Not imported |Not authorized |

Actual A1 runs create `results/A0_A1_COMPARISON.json` with recorded A1 elimination/final-forest timing, rank-group count and Validation result. A0 measurements and speedup remain null unless independently supplied and reconciled. Do not substitute A1 for an official baseline.
