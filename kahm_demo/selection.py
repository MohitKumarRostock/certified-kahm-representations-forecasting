"""Development-only landmark choice, exact sample loss, and decoder selection.

The empirical-loss functions use recorded episode counts and the declared
dyadic observation values. They never use population probabilities or the
reference-risk oracle. Repeated histories and outcomes retain their full weight.
Selecting on validation loss is development, not a risk certificate or freeze.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Any

import numpy as np

from .geometry import Standardizer, _farthest_first, _immutable_array, _matrix
from .law import Episode, OBSERVATIONS


@dataclass(frozen=True, slots=True, eq=False)
class LandmarkPlan:
    row_indices: tuple[int, ...]
    raw_histories: np.ndarray
    occurrence_counts: tuple[int, ...]
    training_count: int
    distinct_count: int
    budget: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_indices", tuple(self.row_indices))
        object.__setattr__(self, "occurrence_counts", tuple(self.occurrence_counts))
        object.__setattr__(self, "raw_histories", _immutable_array(self.raw_histories))

    def record(self) -> dict:
        return {
            "schema": "kahm.raw_landmark_plan.v1",
            "row_indices": list(self.row_indices),
            "raw_histories_hex": [[float(value).hex() for value in row]
                                  for row in self.raw_histories],
            "occurrence_counts": list(self.occurrence_counts),
            "training_count": self.training_count,
            "distinct_count": self.distinct_count,
            "budget": self.budget,
            "effective_count": len(self.row_indices),
            "shortage_reduction": self.distinct_count < self.budget,
            "selection": "all distinct raw rows in lexicographic order when within budget; otherwise farthest-first in training-standardized raw coordinates",
            "first_choice_and_ties": "first lexicographic distinct raw row",
            "row_identity": "first zero-based occurrence in the training matrix",
            "shared_across_variants_and_caps": True,
            "encoded_landmark_duplicates_removed": False,
            "targets_used": False,
        }


def select_landmarks(raw_training: Any, standardizer: Standardizer,
                     budget: int = 16) -> LandmarkPlan:
    """Choose raw-history identities once, without inspecting encoded distances.

    All occurrences fit the supplied history standardizer. Only landmark
    identities are deduplicated here; neither fitting rows nor evaluation rows
    are removed. If encodings map distinct raw landmarks to the same vector,
    downstream Nyström eigentruncation handles that collision without changing
    the chosen identities.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("landmark budget must be a positive integer")
    if not isinstance(standardizer, Standardizer):
        raise TypeError("standardizer must be a training-fitted Standardizer")
    rows = _matrix(raw_training, "raw_training", width=standardizer.input_dimension)
    if standardizer.sample_count != len(rows):
        raise ValueError("standardizer sample count differs from the training matrix")
    distinct, first, counts = np.unique(rows, axis=0, return_index=True, return_counts=True)
    if len(distinct) <= budget:
        selected = np.arange(len(distinct))
    else:
        z = standardizer.transform(distinct)
        selected = np.asarray(_farthest_first(z, budget), dtype=np.int64)
    return LandmarkPlan(
        tuple(int(i) for i in first[selected]), distinct[selected],
        tuple(int(n) for n in counts[selected]), int(len(rows)), int(len(distinct)), budget,
    )


def _state_tuple(value: Any, width: int, description: str) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != width:
        raise TypeError(f"{description} must be a tuple of {width} integer states")
    if any(type(state) is not int for state in value):
        raise TypeError(f"{description} must contain integer states, not booleans or other types")
    if any(not 0 <= state < len(OBSERVATIONS) for state in value):
        raise ValueError(f"{description} states must lie in {{0, 1, 2, 3}}")
    return value


def empirical_counts(episodes: Iterable[Episode]) -> Counter[tuple[int, int, int]]:
    """Count every observed episode; duplicate triples are not discarded."""
    counts = Counter()
    for episode in episodes:
        if not isinstance(episode, Episode):
            raise TypeError("empirical counts require Episode records")
        triple = _state_tuple((episode.previous_state, episode.current_state, episode.next_state),
                              3, "episode triple")
        counts[triple] += 1
    if not counts:
        raise ValueError("empirical loss requires at least one recorded episode")
    return counts


def _exact_predictions(predictions: Mapping) -> dict[tuple[int, int], Fraction]:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must map ordered state pairs to actual float outputs")
    exact = {}
    for history, prediction in predictions.items():
        key = _state_tuple(history, 2, "prediction history")
        if key in exact:
            raise ValueError("duplicate prediction history")
        if type(prediction) is not float:
            raise TypeError("predictions must be actual Python float outputs")
        if not math.isfinite(prediction) or not 0 <= prediction <= 1:
            raise ValueError("predictions must already be finite and clipped to [0, 1]")
        exact[key] = Fraction.from_float(prediction)
    return exact


def empirical_mse_from_counts(counts: Mapping[tuple[int, int, int], int],
                              predictions: Mapping[tuple[int, int], float]) -> Fraction:
    """Return the exact sample MSE; no population weights enter this calculation.

    A prediction is required for each observed history. Full-support validation
    is a separate prediction-table responsibility. Every supplied prediction is
    nevertheless checked, even if its history is absent from this sample.
    """
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("counts must be a nonempty mapping of observed triples")
    exact = _exact_predictions(predictions)
    total = 0
    sse = Fraction(0)
    for key, count in counts.items():
        a, b, c = _state_tuple(key, 3, "counted episode triple")
        if type(count) is not int or count <= 0:
            raise ValueError("every observed triple must have a positive integer count")
        if (a, b) not in exact:
            raise ValueError(f"missing prediction for observed history {(a, b)}")
        sse += count * (OBSERVATIONS[c] - exact[(a, b)]) ** 2
        total += count
    return sse / total


def empirical_mse(episodes: Iterable[Episode],
                  predictions: Mapping[tuple[int, int], float]) -> Fraction:
    """Aggregate repeated recorded triples, preserving their full sample weight."""
    return empirical_mse_from_counts(empirical_counts(episodes), predictions)


def fraction_record(value: Fraction) -> dict:
    """Serialize an exact rational with a nonauthoritative float approximation."""
    if not isinstance(value, Fraction):
        raise TypeError("fraction_record requires a Fraction")
    try:
        approximation = float(value)
    except OverflowError as exc:
        raise ValueError("fraction has no finite float approximation") from exc
    if not math.isfinite(approximation):
        raise ValueError("fraction has no finite float approximation")
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "decimal_approximation": approximation}


def parse_fraction_record(record: Any) -> Fraction:
    """Require canonical reduced rational strings and a matching float display."""
    if type(record) is not dict or set(record) != {"numerator", "denominator", "decimal_approximation"}:
        raise ValueError("invalid fraction record fields")
    numerator_text, denominator_text = record["numerator"], record["denominator"]
    if type(numerator_text) is not str or type(denominator_text) is not str:
        raise TypeError("fraction numerator and denominator must be canonical integer strings")
    try:
        numerator, denominator = int(numerator_text), int(denominator_text)
    except ValueError as exc:
        raise ValueError("invalid fraction integer text") from exc
    if str(numerator) != numerator_text or str(denominator) != denominator_text or denominator <= 0:
        raise ValueError("fraction integers are noncanonical or denominator is not positive")
    value = Fraction(numerator, denominator)
    if value.numerator != numerator or value.denominator != denominator:
        raise ValueError("fraction record must be reduced to lowest terms")
    approximation = record["decimal_approximation"]
    expected = fraction_record(value)["decimal_approximation"]
    if type(approximation) is not float or not math.isfinite(approximation):
        raise ValueError("fraction display must be a finite float")
    if approximation.hex() != expected.hex():
        raise ValueError("fraction display disagrees with the exact rational")
    return value


def _trial_id(value: Any) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("trial_id must be a nonempty canonical string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("trial_id must not contain control characters")
    return value


def select_trial_records(records: Iterable[dict], bank_order: Iterable[str]) -> dict:
    """Select RAW independently and a shared bank using two within-bank minima.

    Exact validation MSE decides every comparison. A smaller grid_index breaks
    decoder ties; the supplied bank order breaks ties between bank means. This
    function permits partial grids for small fixtures but requires RAW and both
    KAHM variants for every declared bank. The workflow separately checks the
    complete prespecified 225-trial roster.
    """
    if isinstance(bank_order, (str, bytes)):
        raise TypeError("bank_order must be an ordered collection of bank IDs")
    banks = tuple(bank_order)
    if not banks or any(type(bank) is not str or re.fullmatch(r"cap_[1-9][0-9]*", bank) is None
                        for bank in banks):
        raise ValueError("bank_order must contain canonical cap_N identities")
    if len(set(banks)) != len(banks):
        raise ValueError("bank_order contains duplicate bank IDs")
    expected_fields = {"trial_id", "candidate_id", "bank_id", "grid_index", "validation_mse"}
    by_group = {(candidate, bank): [] for bank in banks for candidate in ("kahm_mem", "kahm_local")}
    by_group[("raw", None)] = []
    seen_ids = set()
    seen_grid_states = set()
    for record in records:
        if not isinstance(record, Mapping) or not expected_fields <= set(record):
            raise ValueError("trial record is missing required selection fields")
        trial = _trial_id(record["trial_id"])
        if trial in seen_ids:
            raise ValueError("duplicate trial ID")
        seen_ids.add(trial)
        candidate, bank, index = record["candidate_id"], record["bank_id"], record["grid_index"]
        if type(candidate) is not str or candidate not in ("raw", "kahm_mem", "kahm_local"):
            raise ValueError("unknown trial candidate")
        if candidate == "raw":
            if bank is not None:
                raise ValueError("RAW trials must have bank_id=None")
        elif type(bank) is not str or bank not in banks:
            raise ValueError("KAHM trial belongs to an undeclared or missing bank")
        if type(index) is not int or not 0 <= index < 45:
            raise ValueError("grid_index must be an integer in 0..44")
        state = candidate, bank, index
        if state in seen_grid_states:
            raise ValueError("duplicate candidate/bank/grid state")
        seen_grid_states.add(state)
        risk = parse_fraction_record(record["validation_mse"])
        if not 0 <= risk <= 1:
            raise ValueError("clipped squared validation loss must lie in [0, 1]")
        by_group[(candidate, bank)].append((risk, index, trial))
    if any(not rows for rows in by_group.values()):
        raise ValueError("selection requires RAW and both KAHM variants in every declared bank")
    best = {group: min(rows, key=lambda row: (row[0], row[1])) for group, rows in by_group.items()}
    per_bank = []
    bank_scores = []
    for bank in banks:
        mem, local = best[("kahm_mem", bank)], best[("kahm_local", bank)]
        score = (mem[0] + local[0]) / 2
        bank_scores.append(score)
        per_bank.append({"bank_id": bank, "mem_trial_id": mem[2], "local_trial_id": local[2],
                         "mean_validation_mse": fraction_record(score)})
    # min preserves the first index when two exact Fraction scores are equal.
    chosen = min(range(len(banks)), key=lambda index: bank_scores[index])
    return {"raw_trial_id": best[("raw", None)][2], "shared_bank_id": banks[chosen],
            "mem_trial_id": per_bank[chosen]["mem_trial_id"],
            "local_trial_id": per_bank[chosen]["local_trial_id"],
            "per_bank": per_bank}


__all__ = ["LandmarkPlan", "select_landmarks", "empirical_counts", "empirical_mse",
           "empirical_mse_from_counts", "fraction_record", "parse_fraction_record",
           "select_trial_records"]
