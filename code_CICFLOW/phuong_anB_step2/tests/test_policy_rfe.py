import unittest
from unittest.mock import patch
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from phuong_anB_step2.contract import Blocked, core, scientific
from phuong_anB_step2.policy import (COARSE, RADIUS, POLICY_NAME, approved_policy, coarse_grid,
                                    fine_candidates, fine_grid, policy_spec, winner)
from phuong_anB_step2.evidence.original_policy_excerpt import (
    COARSE as ORIGINAL_COARSE, RADIUS as ORIGINAL_RADIUS,
    choose_best as original_choose_best, fine_grid as original_fine_grid)
from phuong_anB_step2.rfe import fit_step2, total_order, validate_result
from .fixtures import SOURCE
from .helpers import TiedForest


class PolicyRFE(unittest.TestCase):
    def test_rank_group_importance_then_original_mi_position_tie_order(self):
        names, ranks = ["a", "b", "c", "d"], [3, 3, 2, 1]
        steps = [{"step": 1, "active_indices": [0, 1, 2, 3], "removed_indices": [0, 1],
                  "squared_importances": [0.1, 0.4, 0.5, 0.9]},
                 {"step": 2, "active_indices": [2, 3], "removed_indices": [2],
                  "squared_importances": [0.5, 0.9]}]
        self.assertEqual([r["feature"] for r in total_order(names, ranks, steps)], ["d", "c", "b", "a"])
        steps[0]["squared_importances"][:2] = [0.4, 0.4]
        self.assertEqual([r["feature"] for r in total_order(names, ranks, steps)], ["d", "c", "a", "b"])

    def request(self, **changes):
        return {"rfe_step": 2, "search_policy": POLICY_NAME, "coarse": list(COARSE), "radius": RADIUS, **changes}

    def test_user_supplied_policy_is_fixed_and_needs_no_extra_approval(self):
        self.assertEqual(approved_policy(self.request()), policy_spec())
        for changed in ({"rfe_step": 1}, {"rfe_step": 5}, {"radius": 2},
                        {"coarse": list(range(100, 1, -2))+[1]},
                        {"search_policy": "best_smallest"}, {"search_policy": "all_tied"}):
            with self.assertRaises(Blocked):
                approved_policy(self.request(**changed))

    def test_coarse_exactly_matches_original_six_values_and_is_independent_of_rfe_step(self):
        self.assertEqual(COARSE, ORIGINAL_COARSE)
        self.assertEqual(RADIUS, ORIGINAL_RADIUS)
        self.assertEqual(coarse_grid(), [10, 20, 30, 50, 75, 100])
        self.assertEqual(len(coarse_grid()), 6)
        self.assertEqual(policy_spec()["max_unique_svm_fits"], 16)

    def test_fine_radius_five_matches_original_for_every_possible_anchor(self):
        for anchor in range(1, 101):
            for tried in (COARSE, (*COARSE, 27, 28), ()):
                self.assertEqual(fine_grid(anchor, tried), original_fine_grid(anchor, tried))
        self.assertEqual(fine_grid(30, COARSE), [25, 26, 27, 28, 29, 31, 32, 33, 34, 35])
        self.assertEqual(fine_grid(100, COARSE), [95, 96, 97, 98, 99])
        self.assertEqual(fine_grid(1, (1,)), [2, 3, 4, 5, 6])

    def test_coarse_tie_uses_only_smallest_anchor_as_original(self):
        rows = [{"k": k, "accuracy": 0.9 if k in (20, 75) else 0.5} for k in COARSE]
        self.assertEqual(winner(rows)["k"], 20)
        self.assertEqual(fine_candidates(rows, policy_spec()), original_fine_grid(20, COARSE))
        self.assertNotIn(74, fine_candidates(rows, policy_spec()))

    def test_winner_matches_original_highest_accuracy_smallest_k_on_tie(self):
        examples = [
            [{"k": 100, "accuracy": 0.9}, {"k": 20, "accuracy": 0.9}, {"k": 10, "accuracy": 0.8}],
            [{"k": 30, "accuracy": 0.9}, {"k": 25, "accuracy": 0.9}, {"k": 35, "accuracy": 0.7}],
            [{"k": 50, "accuracy": 0.9}, {"k": 52, "accuracy": 1.0}],
        ]
        for rows in examples:
            original = [{"k": r["k"], "metrics": {"accuracy": r["accuracy"]}} for r in rows]
            self.assertEqual(winner(rows)["k"], original_choose_best(original)["k"])
        with self.assertRaises(Blocked):
            winner([{"k": 20, "accuracy": 0.9}, {"k": 20, "accuracy": 0.9}])

    def test_fine_requires_exactly_complete_coarse_in_original_order(self):
        with self.assertRaises(Blocked):
            fine_candidates([{"k": 10, "accuracy": 0.9}], policy_spec())
        with self.assertRaises(Blocked):
            fine_candidates([{"k": k, "accuracy": 0.9} for k in reversed(COARSE)], policy_spec())

    def test_every_coarse_anchor_schedule_matches_original_driver_and_fit_limit(self):
        for anchor in COARSE:
            coarse = [{"k": k, "accuracy": 0.95 if k == anchor else 0.5} for k in COARSE]
            original_rows = [{"k": row["k"], "metrics": {"accuracy": row["accuracy"]}} for row in coarse]
            original_best = original_choose_best(original_rows)["k"]
            expected = list(ORIGINAL_COARSE) + original_fine_grid(original_best, ORIGINAL_COARSE)
            actual = coarse_grid() + fine_candidates(coarse, policy_spec())
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 11 if anchor == 100 else 16)
            self.assertEqual(len(actual), len(set(actual)))

    def test_rfe_step2_matches_sklearn_ties_and_resumed_order(self):
        X, y = np.ones((8, 100), np.float32), np.array([0, 1]*4)
        names, steps = [f"f{i}" for i in range(100)], []
        expected = RFE(TiedForest(), step=2, n_features_to_select=1).fit(X, y)
        result = fit_step2(X, y, names, TiedForest(), checkpoint=steps.append)
        np.testing.assert_array_equal(result["ranking"], expected.ranking_)
        self.assertEqual(len(steps), 50)
        self.assertEqual(result["forest_fits_in_completed_rfe"], 51)
        self.assertTrue(all(len(row["removed_indices"]) <= 2 for row in steps))
        self.assertEqual(len(steps[-1]["removed_indices"]), 1)
        resumed = fit_step2(X, y, names, TiedForest(), steps=steps[:17])
        self.assertEqual(resumed["ranking"], result["ranking"])
        self.assertEqual(resumed["records"], result["records"])
        self.assertEqual(validate_result(resumed, names), result["names"])

    def test_rfe_val_role_or_bad_checkpoint_blocked(self):
        with self.assertRaises(Blocked):
            fit_step2(np.ones((4, 3)), np.array([0, 1]*2), ["a", "b", "c"], TiedForest(), fit_role="val")
        with self.assertRaises(Blocked):
            fit_step2(np.ones((4, 3)), np.array([0, 1]*2), ["a", "b", "c"], TiedForest(), steps=[{}])

    def test_step1_ranking_cannot_be_loaded_as_step2_result(self):
        names = ["a", "b", "c"]
        result = fit_step2(np.ones((4, 3)), np.array([0, 1]*2), names, TiedForest())
        result["step"] = 1
        with self.assertRaisesRegex(Blocked, "step2"):
            validate_result(result, names)

    def test_small_real_rf_importances_match_sklearn_step2_ranking(self):
        # Small synthetic estimator only; production RF parameters stay unchanged.
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        X = rng.normal(size=(24, 7)).astype(np.float32)
        y, names = np.array([0, 1]*12), [f"f{i}" for i in range(7)]
        forest = RandomForestClassifier(n_estimators=3, max_depth=2,
            random_state=42, class_weight="balanced", n_jobs=1)
        expected = RFE(forest, step=2, n_features_to_select=1).fit(X, y)
        actual = fit_step2(X, y, names, forest)
        np.testing.assert_array_equal(actual["ranking"], expected.ranking_)
        self.assertEqual(actual["forest_fits_in_completed_rfe"], 4)

    def test_mi_is_one_train_matrix_call_after_same_standard_scaler(self):
        module = core(SOURCE)
        X = np.arange(8*380, dtype=np.float32).reshape(8, 380)
        y, names, seen = np.array([0, 1]*4), [f"f{i}" for i in range(380)], []
        def mutual_info(matrix, labels, **kwargs):
            seen.append((matrix.copy(), labels.copy(), kwargs))
            return np.linspace(0, 1, 380)
        with patch("sklearn.feature_selection.mutual_info_classif", side_effect=mutual_info):
            scores = module.reference_mi(X, y, names)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0].shape, (8, 380))
        self.assertEqual(seen[0][0].dtype, np.float32)
        np.testing.assert_allclose(seen[0][0], StandardScaler().fit_transform(X))
        np.testing.assert_array_equal(seen[0][1], y)
        self.assertEqual(seen[0][2], {"random_state": 42})
        self.assertEqual(scores.index[0], "f379")

    def test_parameters_scaler_mi_seed_match_approved_a1_helper(self):
        original = core(SOURCE).contract("svm")
        new = scientific(SOURCE, policy_spec())
        self.assertEqual(new["mi"], original["mi"])
        self.assertEqual(new["classifier"], original["classifier"])
        self.assertEqual(new["rfe"]["parameters"], original["rfe"]["parameters"])
        self.assertEqual(new["rfe"]["parameters"]["n_estimators"], 100)
        self.assertEqual(new["rfe"]["parameters"]["max_depth"], 15)
        self.assertEqual(new["rfe"]["step"], 2)
        self.assertNotIn("k_grid", new)
        self.assertEqual(new["test_evaluation_calls"], 0)
