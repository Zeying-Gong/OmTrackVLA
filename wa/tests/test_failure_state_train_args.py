"""CPU-only argument gates; no files, caches, models or training are admitted."""
import argparse
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock

from wa.wm.failure_state_train_args import (
    FIELDS, PARENT_SHA256, add_arguments, source_pins_from_args,
    validate_training_recipe,
)


def enabled():
    values = {
        field: (str(i % 10) * 64 if field.endswith("_sha256") else "/not-opened/" + field)
        for i, field in enumerate(FIELDS)
    }
    return SimpleNamespace(**values, kind="jepa", evaluation_set_adaptation=True,
        dual_teacher_cache="/not-opened/old-cache",
        teacher_window_plan="/not-opened/old-plan", teacher_plan_report_sha256="a" * 64,
        dual_teacher_repeats=1, recovery_cache=None, recovery_index=None,
        completed_epochs=1, epochs=1, seed=42, batch_size=2, accumulation=2,
        world_weight=.1, history_repeat_probability=.25,
        resume="/not-opened/59866/checkpoint.pt", resume_sha256=PARENT_SHA256,
        diagnostic=False)


class FailureStateTrainArgsTests(unittest.TestCase):
    def test_add_exact_eleven_options_and_none_defaults(self):
        parser = argparse.ArgumentParser()
        self.assertIs(add_arguments(parser), parser)
        a = parser.parse_args([])
        self.assertEqual(set(vars(a)), set(FIELDS))
        self.assertTrue(all(value is None for value in vars(a).values()))
        self.assertFalse(validate_training_recipe(a))
        self.assertEqual(source_pins_from_args(a), {})

    def test_add_arguments_preserves_existing_defaults(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--history-repeat-probability", type=float, default=0.)
        before = vars(parser.parse_args([])).copy()
        add_arguments(parser)
        after = vars(parser.parse_args([]))
        self.assertEqual({key: after[key] for key in before}, before)
        self.assertFalse(validate_training_recipe(parser.parse_args([]), world=8))

    def test_existing_train_defaults_without_new_attributes_unchanged(self):
        a = SimpleNamespace(kind="jepa", batch_size=2, accumulation=2,
            epochs=1, seed=42, world_weight=.1, diagnostic=False, resume=None,
            resume_sha256=None, completed_epochs=0, history_repeat_probability=0.,
            recovery_cache=None, recovery_index=None, recovery_repeats=16,
            dual_teacher_cache=None, teacher_window_plan=None,
            teacher_plan_report_sha256=None, dual_teacher_repeats=None,
            evaluation_set_adaptation=False)
        before = deepcopy(vars(a))
        self.assertFalse(validate_training_recipe(a, world=8))
        self.assertEqual(source_pins_from_args(a), {})
        self.assertEqual(vars(a), before)

    def test_cli_destinations_and_values(self):
        parser = add_arguments(argparse.ArgumentParser())
        original = enabled()
        argv = [part for field in FIELDS for part in
                ("--" + field.replace("_", "-"), getattr(original, field))]
        actual = parser.parse_args(argv)
        self.assertEqual(vars(actual), {field: getattr(original, field) for field in FIELDS})

    def test_full_enabled_formal(self):
        self.assertIs(validate_training_recipe(enabled(), world=8), True)

    def test_world_none_defers_distributed_check(self):
        self.assertTrue(validate_training_recipe(enabled()))

    def test_every_single_field_alone_fails_closed(self):
        for field in FIELDS:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_recipe(SimpleNamespace(**{field: getattr(enabled(), field)}))

    def test_every_missing_field_fails_closed(self):
        for field in FIELDS:
            a = enabled()
            setattr(a, field, None)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_recipe(a, world=8)

    def test_empty_option_activates_and_is_rejected_not_disabled(self):
        for value in ("", " ", False, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_training_recipe(SimpleNamespace(failure_state_cache=value))

    def test_all_path_values_nonempty_strings(self):
        paths = [field for field in FIELDS if not field.endswith("_sha256")]
        paths += ["dual_teacher_cache", "teacher_window_plan", "resume"]
        for field in paths:
            for value in (None, "", "  ", False, 0, [], "/bad\x00path"):
                a = enabled()
                setattr(a, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_training_recipe(a, world=8)

    def test_all_hashes_canonical_hex(self):
        hashes = [field for field in FIELDS if field.endswith("_sha256")]
        hashes += ["teacher_plan_report_sha256", "resume_sha256"]
        for field in hashes:
            for value in (None, "", "a" * 63, "a" * 65, "g" * 64, "A" * 64,
                          "a" * 63 + "\n", "a" * 64 + "\n", False, 0, []):
                a = enabled()
                setattr(a, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_training_recipe(a, world=8)

    def test_requires_exact_parent_hash_but_does_not_guess_path_or_open_it(self):
        a = enabled()
        a.resume = "/unverified/nonexistent/parent.pt"
        self.assertTrue(validate_training_recipe(a, world=8))
        a.resume_sha256 = "b" * 64
        with self.assertRaises(ValueError):
            validate_training_recipe(a, world=8)

    def test_old_plan_hash_is_runtime_bound_not_guessed_constant(self):
        a = enabled()
        a.failure_state_old_plan_sha256 = "a" * 64
        self.assertTrue(validate_training_recipe(a, world=8))
        a.failure_state_old_plan_sha256 = "b" * 64
        self.assertTrue(validate_training_recipe(a, world=8))

    def test_evaluation_adaptation_requires_true_bool(self):
        for value in (False, None, 0, 1, "true", "false", []):
            a = enabled()
            a.evaluation_set_adaptation = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_training_recipe(a, world=8)

    def test_diagnostic_requires_bool(self):
        for value in (None, 0, 1, "true", "false", []):
            a = enabled()
            a.diagnostic = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_training_recipe(a, world=8)

    def test_formal_world_exact_eight(self):
        for world in (0, 1, 2, 4, 7, 16, 8., "8", True, False):
            with self.subTest(world=world), self.assertRaises(ValueError):
                validate_training_recipe(enabled(), world=world)

    def test_diagnostic_world_one_or_eight(self):
        a = enabled()
        a.diagnostic = True
        for world in (1, 8):
            self.assertTrue(validate_training_recipe(a, world=world))
        for world in (0, 2, 4, 16, 1., True, "1"):
            with self.subTest(world=world), self.assertRaises(ValueError):
                validate_training_recipe(a, world=world)

    def test_cluster_diagnostic_prohibition_remains_entrypoint_responsibility(self):
        a = enabled()
        a.diagnostic = True
        with mock.patch.dict("os.environ", {"MD_AK_JOB_ID": "not-a-real-job"}):
            # This pure parameter module does not replace the preexisting entry gate.
            self.assertTrue(validate_training_recipe(a, world=1))

    def test_wrong_integers_and_bool_aliases_rejected(self):
        fixed = dict(completed_epochs=1, epochs=1, seed=42, batch_size=2,
                     accumulation=2, dual_teacher_repeats=1)
        for field, target in fixed.items():
            for value in (None, True, False, float(target), str(target), target + 1, 0, -1):
                a = enabled()
                setattr(a, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_training_recipe(a, world=8)

    def test_nonfinite_or_changed_float_recipe_rejected(self):
        for field, target in (("world_weight", .1), ("history_repeat_probability", .25)):
            for value in (None, True, False, 0, 1, float("nan"), float("inf"),
                          -float("inf"), target + .001, str(target)):
                a = enabled()
                setattr(a, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_training_recipe(a, world=8)

    def test_kind_jepa_only(self):
        for value in (None, "dino", "JEPA", True, ""):
            a = enabled()
            a.kind = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_training_recipe(a, world=8)

    def test_any_legacy_recovery_option_excluded(self):
        for field in ("recovery_cache", "recovery_index"):
            for value in ("/old/cache", "", False):
                a = enabled()
                setattr(a, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_training_recipe(a, world=8)

    def test_source_redirection_rejected_when_enabled(self):
        for field in ("data_root", "source_prefix"):
            for value in ("/different/source", "", False):
                a = enabled()
                setattr(a, field, value)
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "redirection"):
                    validate_training_recipe(a, world=8)

    def test_disabled_legacy_source_redirection_unchanged(self):
        a = SimpleNamespace(data_root="/legacy/root", source_prefix="/legacy/prefix")
        self.assertFalse(validate_training_recipe(a, world=8))
        self.assertEqual(source_pins_from_args(a), {})
        self.assertEqual(a.data_root, "/legacy/root")
        self.assertEqual(a.source_prefix, "/legacy/prefix")

    def test_source_pins_exact_mapping_and_copy(self):
        a = enabled()
        result = source_pins_from_args(a)
        self.assertEqual(result, dict(old_plan=a.failure_state_old_plan_sha256,
            old_actual_exposure=a.failure_state_old_exposure_sha256,
            recovery_admission=a.failure_state_admission_sha256,
            dedup_report=a.failure_state_dedup_sha256))
        result["old_plan"] = "f" * 64
        self.assertNotEqual(result["old_plan"], a.failure_state_old_plan_sha256)

    def test_source_pins_partial_or_malformed_bundle_rejected(self):
        a = enabled()
        a.failure_state_loader_audit = None
        with self.assertRaises(ValueError):
            source_pins_from_args(a)
        a = enabled()
        a.failure_state_dedup_sha256 = "bad"
        with self.assertRaises(ValueError):
            source_pins_from_args(a)

    def test_validation_does_not_mutate_or_open_data(self):
        a = enabled()
        before = deepcopy(vars(a))
        with mock.patch("builtins.open", side_effect=AssertionError("data IO forbidden")):
            self.assertTrue(validate_training_recipe(a, world=8))
            source_pins_from_args(a)
        self.assertEqual(vars(a), before)


if __name__ == "__main__":
    unittest.main()
