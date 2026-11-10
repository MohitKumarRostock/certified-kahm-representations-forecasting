"""Checks for the simulator's sampling law and reproducible stream interface.

All sampled rows use test-only run IDs and seeds.  No primary calibration
dataset is created or written by this test module.
"""

import hashlib
import json
import random
import unittest
from collections import Counter
from unittest.mock import patch

from kahm_demo.law import BENCHMARK_ID
from kahm_demo.sampling import (
    SAMPLER_VERSION,
    derive_stream_seed,
    sample_episodes,
    stream_descriptor,
    successor_from_ticket,
)


TEST_SEED = 65537
TEST_RUN_ID = "test-sampling"


class TicketLawTests(unittest.TestCase):
    def test_exhaustive_ticket_counts_for_every_history(self):
        # This counts the 20 elementary outcomes, rather than evaluating the
        # benchmark's probability formula through the same implementation.
        for a in range(4):
            for b in range(4):
                with self.subTest(a=a, b=b):
                    counts = Counter(successor_from_ticket(a, b, t) for t in range(20))
                    expected = {c: 1 for c in range(4)}
                    expected[(a + b) % 4] = 17
                    self.assertEqual(dict(counts), expected)

    def test_uniform_input_pairs_and_stationarity_by_counting(self):
        input_pairs = Counter()
        successor_pairs = Counter()
        for pair_id in range(16):
            a, b = divmod(pair_id, 4)
            for ticket in range(20):
                input_pairs[(a, b)] += 1
                successor_pairs[(b, successor_from_ticket(a, b, ticket))] += 1
        expected = {(a, b): 20 for a in range(4) for b in range(4)}
        self.assertEqual(dict(input_pairs), expected)
        self.assertEqual(dict(successor_pairs), expected)

    def test_ticket_boundaries(self):
        self.assertEqual([successor_from_ticket(1, 2, t) for t in (0, 15)], [3, 3])
        self.assertEqual([successor_from_ticket(1, 2, t) for t in range(16, 20)],
                         [0, 1, 2, 3])

    def test_ticket_validation(self):
        for position in range(3):
            for bad in (True, False, 1.0, "1", None):
                args = [0, 0, 0]
                args[position] = bad
                with self.subTest(position=position, bad=bad):
                    with self.assertRaises(TypeError):
                        successor_from_ticket(*args)
        for args in ((-1, 0, 0), (4, 0, 0), (0, -1, 0), (0, 4, 0),
                     (0, 0, -1), (0, 0, 20)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    successor_from_ticket(*args)


class StreamTests(unittest.TestCase):
    def test_seed_derivation_known_value(self):
        # A fixed vector protects the identity of archived streams against
        # accidental changes to namespace, field encoding, or JSON ordering.
        expected_hex = "08a35f6f36219ddf0245f59286b3a04406d45fa9e5a90cdd83255d227f9ff380"
        self.assertEqual(derive_stream_seed(TEST_SEED, "training", TEST_RUN_ID),
                         int(expected_hex, 16))

    def test_metadata_reconstructs_seed_and_is_json_serializable(self):
        descriptor = stream_descriptor(TEST_SEED, "training", TEST_RUN_ID)
        self.assertEqual(json.loads(json.dumps(descriptor)), descriptor)
        self.assertEqual(descriptor["sampler_version"], SAMPLER_VERSION)
        self.assertEqual(descriptor["benchmark_id"], BENCHMARK_ID)
        material = descriptor["derivation"]["canonical_json"]
        payload = json.loads(material)
        self.assertEqual(payload["master_seed"], str(TEST_SEED))
        self.assertEqual(payload["role"], "training")
        self.assertEqual(payload["run_id"], TEST_RUN_ID)
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        self.assertEqual(descriptor["derivation"]["digest_hex"], digest)
        self.assertEqual(int(descriptor["seed_integer"]), int(digest, 16))
        self.assertEqual(int(descriptor["seed_hex"], 16), int(digest, 16))
        self.assertTrue(descriptor["rng"]["python_version"])
        self.assertTrue(descriptor["rng"]["python_full_version"])

    def test_descriptor_does_not_instantiate_rng_or_draw_rows(self):
        with patch("kahm_demo.sampling.random.Random") as rng_type:
            rng_type.VERSION = 3
            stream_descriptor(TEST_SEED, "calibration", "test-descriptor-only")
            rng_type.assert_not_called()

    def test_determinism_and_prefix_stability_for_each_role(self):
        for role in ("training", "validation", "calibration"):
            with self.subTest(role=role):
                short = sample_episodes(17, TEST_SEED, role, TEST_RUN_ID)
                long = sample_episodes(64, TEST_SEED, role, TEST_RUN_ID)
                self.assertEqual(short, sample_episodes(17, TEST_SEED, role, TEST_RUN_ID))
                self.assertEqual(short, long[:17])
                self.assertEqual([episode.episode_id for episode in long], list(range(64)))
                self.assertIsInstance(long, tuple)

    def test_roles_runs_and_master_seeds_have_distinct_stream_identities(self):
        seeds = {
            derive_stream_seed(master, role, run)
            for master in (TEST_SEED, TEST_SEED + 1)
            for role in ("training", "validation", "calibration")
            for run in ("test-stream-a", "test-stream-b")
        }
        self.assertEqual(len(seeds), 12)

    def test_large_master_seed_and_unicode_run_id_round_trip(self):
        master = 2**200 + 123
        run = 'test-ä-"quoted"-\\-stream'
        descriptor = stream_descriptor(master, "validation", run)
        payload = json.loads(descriptor["derivation"]["canonical_json"])
        self.assertEqual(int(payload["master_seed"]), master)
        self.assertEqual(payload["run_id"], run)
        self.assertEqual(int(descriptor["seed_integer"]),
                         derive_stream_seed(master, "validation", run))

    def test_calls_do_not_modify_global_random_state(self):
        before = random.getstate()
        sample_episodes(40, TEST_SEED, "training", "test-global-rng")
        stream_descriptor(TEST_SEED, "calibration", "test-global-rng")
        self.assertEqual(random.getstate(), before)

    def test_each_episode_draws_a_fresh_pair(self):
        class ScriptedRandom:
            def __init__(self):
                self.draws = iter(((16, 0), (20, 0), (16, 15), (20, 19),
                                   (16, 6), (20, 17)))
                self.call_count = 0

            def randrange(self, stop):
                expected_stop, value = next(self.draws)
                if stop != expected_stop:
                    raise AssertionError((stop, expected_stop))
                self.call_count += 1
                return value

        scripted = ScriptedRandom()
        with patch("kahm_demo.sampling.random.Random", return_value=scripted) as factory:
            rows = sample_episodes(3, TEST_SEED, "training", "test-fresh-pairs")
            factory.assert_called_once()
        self.assertEqual(scripted.call_count, 6)
        self.assertEqual([(r.previous_state, r.current_state, r.next_state) for r in rows],
                         [(0, 0, 0), (3, 3, 3), (1, 2, 1)])

    def test_repeated_episodes_are_not_deduplicated(self):
        class ConstantRandom:
            def randrange(self, stop):
                return 0

        with patch("kahm_demo.sampling.random.Random", return_value=ConstantRandom()):
            rows = sample_episodes(5, TEST_SEED, "validation", "test-duplicates")
        self.assertEqual(len(rows), 5)
        self.assertEqual([r.episode_id for r in rows], list(range(5)))
        self.assertEqual({(r.previous_state, r.current_state, r.next_state) for r in rows},
                         {(0, 0, 0)})

    def test_empty_sample_is_valid_and_still_validates_stream(self):
        self.assertEqual(sample_episodes(0, 0, "training", "test-empty"), ())
        with self.assertRaises(ValueError):
            sample_episodes(0, 0, "unknown", "test-empty")

    def test_invalid_counts_and_master_seeds(self):
        for value in (True, False, 1.0, "1", None):
            with self.subTest(count=value):
                with self.assertRaises(TypeError):
                    sample_episodes(value, TEST_SEED, "training", TEST_RUN_ID)
            with self.subTest(master_seed=value):
                with self.assertRaises(TypeError):
                    derive_stream_seed(value, "training", TEST_RUN_ID)
        with self.assertRaises(ValueError):
            sample_episodes(-1, TEST_SEED, "training", TEST_RUN_ID)
        with self.assertRaises(ValueError):
            derive_stream_seed(-1, "training", TEST_RUN_ID)

    def test_invalid_roles_and_run_ids(self):
        for role in (None, True, 1):
            with self.subTest(role=role):
                with self.assertRaises(TypeError):
                    derive_stream_seed(TEST_SEED, role, TEST_RUN_ID)
        for role in ("", "train", "Training", " training", "validation "):
            with self.subTest(role=role):
                with self.assertRaises(ValueError):
                    derive_stream_seed(TEST_SEED, role, TEST_RUN_ID)
        for run in (None, True, 1):
            with self.subTest(run_id=run):
                with self.assertRaises(TypeError):
                    derive_stream_seed(TEST_SEED, "training", run)
        for run in ("", " ", "\t\n"):
            with self.subTest(run_id=run):
                with self.assertRaises(ValueError):
                    derive_stream_seed(TEST_SEED, "training", run)


if __name__ == "__main__":
    unittest.main()
