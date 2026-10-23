"""Small synthetic unit tests; Codex does not execute them."""
import unittest

from stage28_cplus_dryrun import (
    bucket_role, flow_allocation, ratio_objective, role_targets,
    source_flow_allocation,
)


class Stage28SyntheticTests(unittest.TestCase):
    def test_60_20_20_integer_role_targets_conserve_rows(self):
        result = role_targets(4_651_628)
        self.assertEqual(sum(result.values()), 4_651_628)
        self.assertGreater(result["train"], result["val"])

    def test_bucket_with_bad_duration_is_whole_embargo(self):
        plan = {
            "mode": "THREE_WAY", "train_end_i": 0, "val_start_i": 2,
            "val_end_i": 3, "test_start_i": 5,
        }
        self.assertEqual(bucket_role(plan, 0, 1), "embargo")
        self.assertEqual(bucket_role(plan, 0, 0), "train")
        self.assertEqual(bucket_role(plan, 2, 0), "val")
        self.assertEqual(bucket_role(plan, 5, 0), "test")

    def test_ratio_objective_prefers_target(self):
        self.assertLess(ratio_objective((600, 200, 200)),
                        ratio_objective((700, 150, 150)))

    def test_capacity_allocation_detects_insufficient_role_capacity(self):
        caps = {
            "BENIGN": 30, "BRUTE_FORCE": 70, "DDOS": 0, "DOS": 0,
            "MIRAI": 0, "RECON": 0, "SPOOFING": 0, "WEB-BASED": 0,
        }
        supplies = {
            ("BENIGN", "train"): 30,
            ("BRUTE_FORCE", "train"): 40,
            ("BRUTE_FORCE", "val"): 15,
            ("BRUTE_FORCE", "test"): 15,
        }
        achieved, _ = flow_allocation(caps, supplies,
                                      {"train": 60, "val": 20, "test": 20})
        self.assertEqual(achieved, 90)

    def test_source_budget_caps_partial_file(self):
        inputs = {
            "source_budgets": {1: 5, 2: 20},
            "source_categories": {1: "BENIGN", 2: "DDOS"},
        }
        supply = {
            (1, "train"): 10, (1, "val"): 10, (1, "test"): 10,
            (2, "train"): 20, (2, "val"): 20, (2, "test"): 20,
        }
        achieved, plan = source_flow_allocation(
            inputs, supply, {"train": 10, "val": 10, "test": 10})
        self.assertEqual(achieved, 25)
        self.assertEqual(sum(plan[(1, role)] for role in ("train", "val", "test")), 5)


if __name__ == "__main__":
    unittest.main()
