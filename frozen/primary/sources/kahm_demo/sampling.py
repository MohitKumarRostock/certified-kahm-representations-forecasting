"""Deterministic sampling of independent one-transition benchmark episodes.

Every episode draws a fresh ordered state pair and a fresh transition ticket.
Rows are not successive observations from one chain.  Repeated histories and
triples are retained.  Separate role names identify reproducible PRNG streams;
the statistical iid premise is not proved merely by assigning different seeds.

This module performs no I/O.  In particular, calling a descriptor does not draw
calibration data.  The command-line layer controls when a planned primary
calibration stream may be materialized after the model freeze.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys

from .law import BENCHMARK_ID, Episode


SAMPLER_VERSION = "fresh-pair-ticket-v1"
_STREAM_NAMESPACE = "kahm-demonstrator-episode-stream-v1"
_ROLES = frozenset({"training", "validation", "calibration"})


def _integer_in_range(
    name: str, value: int, minimum: int, maximum: int | None = None
) -> int:
    """Require a Python int, explicitly excluding bool and coercion."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {limit}")
    return value


def _canonical_seed_material(master_seed: int, role: str, run_id: str) -> str:
    _integer_in_range("master_seed", master_seed, 0)
    if not isinstance(role, str):
        raise TypeError("role must be a string")
    if role not in _ROLES:
        raise ValueError("role must be training, validation, or calibration")
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if not run_id.strip():
        raise ValueError("run_id must be a nonempty, non-whitespace string")

    # Decimal strings preserve arbitrarily large integer seeds when a manifest
    # is read by a JSON consumer whose numeric type has limited precision.
    payload = {
        "namespace": _STREAM_NAMESPACE,
        "benchmark_id": BENCHMARK_ID,
        "master_seed": str(master_seed),
        "run_id": run_id,
        "role": role,
    }
    # Whitespace/case/Unicode in a valid run ID is preserved, not normalized.
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def derive_stream_seed(master_seed: int, role: str, run_id: str) -> int:
    """Derive one unsigned 256-bit seed from canonical JSON via SHA-256.

    The requested sample count is deliberately absent: increasing a count
    extends the same stream rather than replacing its existing prefix.
    """
    material = _canonical_seed_material(master_seed, role, run_id)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest(), "big")


def stream_descriptor(master_seed: int, role: str, run_id: str) -> dict:
    """Return JSON-serializable stream provenance without sampling any rows.

    ``master_seed_integer`` and ``seed_integer`` are base-10 strings; ``seed_hex``
    records the same derived integer in base 16.  The exact canonical JSON and
    hash interpretation make the derivation independently reconstructible.
    """
    material = _canonical_seed_material(master_seed, role, run_id)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return {
        "sampler_version": SAMPLER_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "role": role,
        "run_id": run_id,
        "master_seed_integer": str(master_seed),
        "seed_integer": str(int(digest, 16)),
        "seed_hex": "0x" + digest,
        "integer_metadata_encoding": "base-10 strings unless marked hexadecimal",
        "rng": {
            "implementation": "random.Random",
            "algorithm": "Mersenne Twister (MT19937)",
            "state_format_version": random.Random.VERSION,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_full_version": sys.version,
        },
        "derivation": {
            "algorithm": "SHA-256",
            "namespace": _STREAM_NAMESPACE,
            "canonical_json": material,
            "encoding": "UTF-8; ASCII JSON escapes; sorted keys; compact separators",
            "digest_hex": digest,
            "integer_interpretation": "unsigned big-endian 256-bit integer",
        },
        "sampling_rule": {
            "pair_draw": "pair_id = rng.randrange(16); a, b = divmod(pair_id, 4)",
            "ticket_draw": "ticket = rng.randrange(20)",
            "successor": "(a+b)%4 for tickets 0..15; ticket-16 for tickets 16..19",
            "episode_dependence": "fresh pair and ticket for every episode",
            "duplicate_policy": "retain every sampled episode",
        },
    }


def successor_from_ticket(a: int, b: int, ticket: int) -> int:
    """Map one of 20 equiprobable tickets to the declared successor state."""
    _integer_in_range("previous_state", a, 0, 3)
    _integer_in_range("current_state", b, 0, 3)
    _integer_in_range("ticket", ticket, 0, 19)
    return (a + b) % 4 if ticket < 16 else ticket - 16


def sample_episodes(
    count: int, master_seed: int, role: str, run_id: str
) -> tuple[Episode, ...]:
    """Generate ``count`` episodes using a private, freshly seeded RNG.

    Episode IDs are zero-based indices within the requested split.  Each call
    with identical stream arguments replays that stream from its beginning.
    The caller must not interpret a replay as additional independent data.
    """
    _integer_in_range("count", count, 0)
    rng = random.Random(derive_stream_seed(master_seed, role, run_id))
    episodes = []
    for episode_id in range(count):
        pair_id = rng.randrange(16)
        a, b = divmod(pair_id, 4)
        ticket = rng.randrange(20)
        episodes.append(Episode(episode_id, a, b, successor_from_ticket(a, b, ticket)))
    return tuple(episodes)
