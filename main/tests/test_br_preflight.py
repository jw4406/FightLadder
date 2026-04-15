import unittest

from main.br_preflight import (
    build_dedicated_job_specs,
    extract_unique_states_from_checkpoint_data,
    infer_cds_architecture,
    sanitize_for_filename,
)


class TestBRPreflightHelpers(unittest.TestCase):
    def test_sanitize_for_filename(self):
        self.assertEqual(sanitize_for_filename("Guile vs Ryu"), "Guile_vs_Ryu")
        self.assertEqual(sanitize_for_filename("!!!"), "unknown")
        self.assertEqual(sanitize_for_filename(None), "unknown")

    def test_extract_unique_states_from_checkpoint_data(self):
        data = {
            "state_list": [
                "s1",
                "s2",
                "s1",
                "s3",
                "s2",
            ]
        }
        self.assertEqual(
            extract_unique_states_from_checkpoint_data(data, task_file_path="dummy.task"),
            ["s1", "s2", "s3"],
        )

    def test_extract_unique_states_requires_state_list(self):
        with self.assertRaises(KeyError):
            extract_unique_states_from_checkpoint_data({}, task_file_path="dummy.task")

        with self.assertRaises(ValueError):
            extract_unique_states_from_checkpoint_data({"state_list": []}, task_file_path="dummy.task")

    def test_infer_cds_architecture(self):
        class FakeIPPOModule:
            __name__ = "CleanIPPOActorActorCriticPolicy"

        self.assertEqual(
            infer_cds_architecture({"model_arch_type": "ippo"}, "checkpoint.task"),
            "ippo",
        )
        self.assertEqual(
            infer_cds_architecture({"policy_class": FakeIPPOModule}, "checkpoint.task"),
            "ippo",
        )
        self.assertEqual(
            infer_cds_architecture({}, "ippo_test_checkpoint.task"),
            "ippo",
        )
        self.assertEqual(
            infer_cds_architecture({}, "spar_checkpoint.task"),
            "spar",
        )

    def test_build_dedicated_job_specs(self):
        states = ["state_a", "state_b"]
        specs = build_dedicated_job_specs(
            unique_states=states,
            replicates_per_matchup=2,
            run_eval_prot=True,
            run_eval_adv=True,
            launch_local_br_eval=False,
            state_to_matchup=lambda s: f"{s}:matchup",
        )
        # 2 states * 2 sides * 2 replicates = 8 jobs.
        self.assertEqual(len(specs), 8)
        self.assertEqual([spec["job_index"] for spec in specs], list(range(8)))

        # First two jobs should be eval_prot=True for state_a replicates.
        self.assertTrue(specs[0]["eval_prot"])
        self.assertTrue(specs[1]["eval_prot"])
        self.assertEqual(specs[0]["state_subset"], ["state_a"])
        self.assertEqual(specs[1]["state_subset"], ["state_a"])
        self.assertEqual(specs[0]["replicate_idx"], 0)
        self.assertEqual(specs[1]["replicate_idx"], 1)

        # Next two jobs should be eval_prot=False for state_a replicates.
        self.assertFalse(specs[2]["eval_prot"])
        self.assertFalse(specs[3]["eval_prot"])
        self.assertEqual(specs[2]["state_subset"], ["state_a"])
        self.assertEqual(specs[3]["state_subset"], ["state_a"])

        # Matchup label is sanitized.
        self.assertEqual(specs[0]["matchup_label"], "state_a_matchup")

    def test_build_dedicated_job_specs_rejects_bad_replicates(self):
        with self.assertRaises(ValueError):
            build_dedicated_job_specs(
                unique_states=["state_a"],
                replicates_per_matchup=0,
                run_eval_prot=True,
                run_eval_adv=False,
                launch_local_br_eval=False,
            )


if __name__ == "__main__":
    unittest.main()
