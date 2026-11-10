"""Exact occurrence-level calibration losses for an already verified family.

The caller must establish frozen-model identity and authorize calibration access
before supplying a Dataset and the models' actual canonical prediction tables.
This module performs no sampling, model inference, or population-risk evaluation.
Every calibration occurrence is retained. Binary64 predictions are interpreted
as their exact represented rational values, with no decimal approximation.

Rows use JSON objects with candidate maps in the fixed RAW/MEM/LOCAL order.
Rational records contain reduced base-10 numerator/denominator strings. The
verifier checks these encodings and recomputes each row's losses from the supplied
episodes and predictions; a file hash alone is not a ledger verification.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import tempfile

from .contract import CANDIDATE_IDS, PRIMARY_CALIBRATION_COUNT
from .datasets import Dataset
from .law import Episode, HISTORY_KEYS, OBSERVATIONS


LEDGER_ROW_SCHEMA = "kahm.calibration_loss_row.v1"
LEDGER_SUMMARY_SCHEMA = "kahm.calibration_ledger_summary.v1"
_ROW_FIELDS = {
    "schema", "run_id", "episode_id", "previous_state", "current_state",
    "next_state", "history_id", "target", "prediction_hex", "squared_losses",
}


def _fraction_record(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _parse_fraction(record: object, description: str) -> Fraction:
    if type(record) is not dict or set(record) != {"numerator", "denominator"}:
        raise ValueError(f"{description} must be a numerator/denominator record")
    numerator_text, denominator_text = record["numerator"], record["denominator"]
    if type(numerator_text) is not str or type(denominator_text) is not str:
        raise ValueError(f"{description} requires canonical integer strings")
    try:
        numerator, denominator = int(numerator_text), int(denominator_text)
    except ValueError as error:
        raise ValueError(f"{description} contains an invalid integer string") from error
    if (str(numerator) != numerator_text or str(denominator) != denominator_text
            or denominator <= 0):
        raise ValueError(f"{description} contains noncanonical integers or a nonpositive denominator")
    value = Fraction(numerator, denominator)
    if _fraction_record(value) != record:
        raise ValueError(f"{description} must be reduced to lowest terms")
    return value


def _state(value: object, description: str) -> int:
    if type(value) is not int or not 0 <= value < 4:
        raise ValueError(f"{description} must be an integer state in 0..3")
    return value


def _validated_inputs(
    dataset: Dataset,
    predictions: dict[str, dict[tuple[int, int], float]],
) -> tuple[tuple[Episode, ...], dict[str, dict[tuple[int, int], float]]]:
    if type(dataset) is not Dataset:
        raise TypeError("calibration ledger requires a Dataset")
    if type(dataset.role) is not str or dataset.role != "calibration":
        raise ValueError("calibration ledger requires role='calibration'")
    if type(dataset.run_id) is not str or not dataset.run_id:
        raise ValueError("calibration Dataset requires a nonempty run_id")
    if type(dataset.episodes) is not tuple or len(dataset.episodes) != PRIMARY_CALIBRATION_COUNT:
        raise ValueError(f"calibration Dataset must contain exactly {PRIMARY_CALIBRATION_COUNT} episodes")
    for index, episode in enumerate(dataset.episodes):
        if type(episode) is not Episode:
            raise TypeError("calibration Dataset must contain Episode records")
        if type(episode.episode_id) is not int or episode.episode_id != index:
            raise ValueError("calibration episode IDs must be sequential from zero")
        _state(episode.previous_state, "previous_state")
        _state(episode.current_state, "current_state")
        _state(episode.next_state, "next_state")
    if type(predictions) is not dict or set(predictions) != set(CANDIDATE_IDS):
        raise ValueError("predictions must contain exactly raw, kahm_mem, and kahm_local")
    normalized = {}
    for candidate in CANDIDATE_IDS:
        table = predictions[candidate]
        if type(table) is not dict or len(table) != len(HISTORY_KEYS):
            raise ValueError(f"{candidate} must predict all 16 histories")
        for history, prediction in table.items():
            if type(history) is not tuple or len(history) != 2:
                raise ValueError("prediction histories must be ordered pairs of integer states")
            _state(history[0], "prediction previous_state")
            _state(history[1], "prediction current_state")
            if type(prediction) is not float:
                raise TypeError("ledger predictions must be actual Python float outputs")
            if not math.isfinite(prediction) or not 0.0 <= prediction <= 1.0:
                raise ValueError("ledger predictions must already be finite and in [0,1]")
        if set(table) != set(HISTORY_KEYS):
            raise ValueError(f"{candidate} must predict exactly the supported histories")
        normalized[candidate] = {history: table[history] for history in HISTORY_KEYS}
    return dataset.episodes, normalized


def _summary(run_id: str, sums: dict[str, Fraction]) -> dict:
    return {
        "schema": LEDGER_SUMMARY_SCHEMA,
        "run_id": run_id,
        "count": PRIMARY_CALIBRATION_COUNT,
        "candidate_ids": list(CANDIDATE_IDS),
        "loss_sums": {candidate: _fraction_record(sums[candidate]) for candidate in CANDIDATE_IDS},
        "empirical_risks": {
            candidate: _fraction_record(sums[candidate] / PRIMARY_CALIBRATION_COUNT)
            for candidate in CANDIDATE_IDS
        },
    }


def write_calibration_ledger(
    dataset: Dataset,
    predictions: dict[str, dict[tuple[int, int], float]],
    path: Path,
) -> dict:
    """Write one row per occurrence and atomically publish a new ledger file.

    The temporary file resides in the output parent. A hard-link commit creates
    the destination atomically only if it does not already exist, including a
    concurrent creator or dangling symlink. Failure cleans up the temporary file;
    an existing destination is never truncated or replaced.
    """
    episodes, normalized = _validated_inputs(dataset, predictions)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"calibration ledger already exists: {path}")
    exact = {
        candidate: {history: Fraction.from_float(value) for history, value in table.items()}
        for candidate, table in normalized.items()
    }
    sums = {candidate: Fraction(0) for candidate in CANDIDATE_IDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".calibration-ledger-",
            suffix=".jsonl", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for episode in episodes:
                history = (episode.previous_state, episode.current_state)
                target = OBSERVATIONS[episode.next_state]
                losses = {}
                for candidate in CANDIDATE_IDS:
                    loss = (target - exact[candidate][history]) ** 2
                    sums[candidate] += loss
                    losses[candidate] = _fraction_record(loss)
                row = {
                    "schema": LEDGER_ROW_SCHEMA,
                    "run_id": dataset.run_id,
                    "episode_id": episode.episode_id,
                    "previous_state": episode.previous_state,
                    "current_state": episode.current_state,
                    "next_state": episode.next_state,
                    "history_id": episode.history_id,
                    "target": _fraction_record(target),
                    "prediction_hex": {
                        candidate: normalized[candidate][history].hex()
                        for candidate in CANDIDATE_IDS
                    },
                    "squared_losses": losses,
                }
                stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":"),
                                        allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        result = _summary(dataset.run_id, sums)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def _read_row(line: str, index: int) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate ledger JSON key at row {index}: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite ledger JSON value at row {index}: {value}")

    try:
        row = json.loads(line, object_pairs_hook=unique, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid ledger JSON at row {index}") from error
    if type(row) is not dict or set(row) != _ROW_FIELDS:
        raise ValueError(f"incorrect ledger fields at row {index}")
    return row


def verify_calibration_ledger(
    dataset: Dataset,
    predictions: dict[str, dict[tuple[int, int], float]],
    path: Path,
) -> dict:
    """Independently recompute every saved row against the supplied observations.

    Exact rational and hexadecimal encodings, occurrence order, states, targets,
    and the entire candidate family are checked before losses enter the returned
    summary. No stored loss or summary is trusted merely because its hash agrees.
    """
    episodes, normalized = _validated_inputs(dataset, predictions)
    sums = {candidate: Fraction(0) for candidate in CANDIDATE_IDS}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for index, episode in enumerate(episodes):
            line = stream.readline()
            if not line:
                raise ValueError(f"calibration ledger ended before episode {index}")
            row = _read_row(line, index)
            if row["schema"] != LEDGER_ROW_SCHEMA or type(row["schema"]) is not str:
                raise ValueError(f"unsupported ledger row schema at row {index}")
            if type(row["run_id"]) is not str or row["run_id"] != dataset.run_id:
                raise ValueError(f"ledger run identity differs at row {index}")
            expected_integers = {
                "episode_id": episode.episode_id,
                "previous_state": episode.previous_state,
                "current_state": episode.current_state,
                "next_state": episode.next_state,
                "history_id": 4 * episode.previous_state + episode.current_state,
            }
            if any(type(row[key]) is not int or row[key] != value
                   for key, value in expected_integers.items()):
                raise ValueError(f"ledger occurrence or state identity differs at row {index}")
            target = OBSERVATIONS[episode.next_state]
            if _parse_fraction(row["target"], f"target at row {index}") != target:
                raise ValueError(f"ledger target differs at row {index}")
            for key in ("prediction_hex", "squared_losses"):
                if type(row[key]) is not dict or tuple(row[key]) != tuple(CANDIDATE_IDS):
                    raise ValueError(f"ledger {key} must retain the fixed candidate order at row {index}")
            history = (episode.previous_state, episode.current_state)
            for candidate in CANDIDATE_IDS:
                encoded = row["prediction_hex"][candidate]
                if type(encoded) is not str:
                    raise ValueError(f"ledger prediction must be a canonical hexadecimal string at row {index}")
                try:
                    prediction = float.fromhex(encoded)
                except (ValueError, OverflowError) as error:
                    raise ValueError(f"invalid ledger hexadecimal prediction at row {index}") from error
                if (not math.isfinite(prediction) or not 0.0 <= prediction <= 1.0
                        or prediction.hex() != encoded
                        or encoded != normalized[candidate][history].hex()):
                    raise ValueError(f"ledger prediction differs from the canonical model output at row {index}")
                difference = target - Fraction.from_float(prediction)
                recomputed_loss = difference * difference
                recorded_loss = _parse_fraction(row["squared_losses"][candidate],
                                                f"{candidate} squared loss at row {index}")
                if recorded_loss != recomputed_loss:
                    raise ValueError(f"ledger squared loss differs for {candidate} at row {index}")
                sums[candidate] += recomputed_loss
        if stream.readline():
            raise ValueError("calibration ledger contains additional rows")
    return _summary(dataset.run_id, sums)


__all__ = ["LEDGER_ROW_SCHEMA", "LEDGER_SUMMARY_SCHEMA",
           "write_calibration_ledger", "verify_calibration_ledger"]
