"""Known-law audits for the finite-support Lipschitz-class demonstrator.

These functions use the analytical population law.  They belong after the
calibration certificate has been issued; they must not select representations,
predictors, calibration streams, or confidence endpoints.  Population class
optima are reported as verified intervals unless an analytical equality or a
matching pair of bounds establishes an exact value.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from typing import Any

from .law import BENCHMARK_ID, HISTORY_KEYS, OBSERVATIONS, conditional_mean, joint_probability
from .oracle import exact_risk


AUDIT_SCHEMA = "kahm.class_population_audit.v1"
HIERARCHY_SCHEMA = "kahm.analytic_class_hierarchy.v1"
_REFERENCE_PAIRS = ((0, 5), (10, 15), (9, 13), (6, 7))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _parse_fraction(record: object) -> Fraction:
    if type(record) is not dict or set(record) != {"numerator", "denominator"}:
        raise ValueError("expected an exact fraction record")
    if any(type(record[key]) is not str for key in record):
        raise ValueError("fraction numerator and denominator must be strings")
    try:
        value = Fraction(int(record["numerator"]), int(record["denominator"]))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid fraction record") from error
    if _fraction(value) != record:
        raise ValueError("fraction record must be canonical and reduced")
    return value


def _exact_predictions(predictions: Sequence[Fraction]) -> tuple[Fraction, ...]:
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise TypeError("predictions must be a sequence of sixteen exact Fractions")
    if len(predictions) != len(HISTORY_KEYS):
        raise ValueError("predictions must cover all sixteen histories in history-ID order")
    values = tuple(predictions)
    if any(type(value) is not Fraction for value in values):
        raise TypeError("population class audits require exact Fraction predictions")
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("predictions must already lie in [0, 1]")
    return values


def population_means() -> tuple[Fraction, ...]:
    """Return the exact conditional means, indexed by history ID."""
    return tuple(conditional_mean(a, b) for a, b in HISTORY_KEYS)


def irreducible_risk() -> Fraction:
    """Sum the full-history conditional variances over all 64 outcomes."""
    means = population_means()
    return sum(
        (joint_probability(a, b, c) * (OBSERVATIONS[c] - means[4 * a + b]) ** 2
         for a, b in HISTORY_KEYS for c in range(4)),
        Fraction(0),
    )


def exact_population_risk(predictions: Sequence[Fraction]) -> Fraction:
    """Evaluate represented predictions by direct enumeration of the 64 cases."""
    values = _exact_predictions(predictions)
    return exact_risk({history: values[index] for index, history in enumerate(HISTORY_KEYS)})


def _block_size(value: int) -> int:
    if type(value) is not int or not 2 <= value <= 2048:
        raise ValueError("block size must be an integer between 2 and 2048")
    return value


def _occupancy_score(block_size: int, groups: int, variance: Fraction) -> Fraction:
    b = _block_size(block_size)
    return variance * (b - groups + groups * Fraction(groups - 1, groups) ** b) / (b - 1)


def hierarchy_reference(block_size: int) -> dict[str, Any]:
    """Exact short/full-history hierarchy with raw coordinates and kappa=4.

    Distinct feature points are at least 1/4 apart.  Their Lipschitz
    constraints therefore permit all predictions in [0,1]; observations with
    identical features still share a prediction.  Conditional on feature
    counts, the optimized squared error is the within-group sample variance.
    """
    b = _block_size(block_size)
    noise = irreducible_risk()
    constant = exact_population_risk((Fraction(1, 2),) * 16)
    return {
        "schema": HIERARCHY_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "block_size": b,
        "scope": "exact population reference; not an issued confidence endpoint",
        "metric": "Euclidean distance on original unstandardized observation coordinates",
        "kappa": _fraction(Fraction(4)),
        "short": {
            "feature_support_size": 4,
            "optimal_risk": _fraction(constant),
            "expected_block_score": _fraction(_occupancy_score(b, 4, constant)),
        },
        "full": {
            "feature_support_size": 16,
            "optimal_risk": _fraction(noise),
            "expected_block_score": _fraction(_occupancy_score(b, 16, noise)),
        },
    }


def _validate_geometry(geometry: Any) -> None:
    # The geometry constructor and numerical witness verifier perform their
    # own structural checks.  This extra requirement is specific to this law.
    if len(geometry.features) != len(HISTORY_KEYS):
        raise ValueError("the benchmark population audit needs exactly sixteen feature rows")


def _analytic_class(geometry: Any, block_sizes: tuple[int, ...]) -> dict[str, Any]:
    """Recognize uniform finite partitions with inactive cross-point constraints."""
    means = population_means()
    groups: dict[tuple[Fraction, ...], list[int]] = {}
    for index, row in enumerate(geometry.features):
        groups.setdefault(tuple(row), []).append(index)
    if any(
        geometry.features[i] != geometry.features[j]
        and geometry.kappa ** 2 * geometry.squared_distances[i][j] < 1
        for i, j in combinations(range(16), 2)
    ):
        return {"applicable": False, "reason": "some cross-point Lipschitz constraints are active"}
    representatives = [indices[0] for indices in groups.values()]
    group_predictions = [Fraction(0)] * 16
    for indices in groups.values():
        mean = sum((means[index] for index in indices), Fraction(0)) / len(indices)
        for index in indices:
            group_predictions[index] = mean
    variance = exact_population_risk(group_predictions)
    equal_mass = len({len(indices) for indices in groups.values()}) == 1
    result: dict[str, Any] = {
        "applicable": True,
        "reason": "all bounded assignments on distinct features are admissible",
        "feature_support_size": len(groups),
        "uniform_feature_masses": equal_mass,
        "optimal_risk": _fraction(variance),
        "group_representatives": representatives,
        "optimal_predictions": [_fraction(value) for value in group_predictions],
        "hierarchy": [],
    }
    if equal_mass:
        result["hierarchy"] = [
            {"block_size": b,
             "expected_block_score": _fraction(_occupancy_score(b, len(groups), variance))}
            for b in block_sizes
        ]
    return result


def _pair_lower_bound(geometry: Any) -> dict[str, Any]:
    """A separate population lower bound from disjoint conditional-mean pairs.

    This calculation does not use the optimizer or its dual witness.  A
    conservative upper distance enclosure makes every retained pair penalty
    valid.  Disjointness prevents counting one squared residual more than once.
    """
    means = population_means()
    contributions = {}
    for i, j in combinations(range(16), 2):
        gap = max(Fraction(0), abs(means[i] - means[j])
                  - geometry.kappa * geometry.distance_upper[i][j])
        contributions[(i, j)] = gap ** 2 / 32
    selected = []
    used: set[int] = set()
    for pair in sorted(contributions, key=lambda p: (-contributions[p], p)):
        if contributions[pair] > 0 and not any(index in used for index in pair):
            selected.append(pair)
            used.update(pair)
    greedy = sum((contributions[pair] for pair in selected), Fraction(0))
    reference = sum((contributions[pair] for pair in _REFERENCE_PAIRS), Fraction(0))
    if reference > greedy:
        selected = list(_REFERENCE_PAIRS)
    records = [
        {"left": i, "right": j,
         "mean_difference": _fraction(abs(means[i] - means[j])),
         "distance_upper": _fraction(geometry.distance_upper[i][j]),
         "risk_contribution": _fraction(contributions[(i, j)])}
        for i, j in selected
    ]
    lower = irreducible_risk() + sum(
        (contributions[pair] for pair in selected), Fraction(0))
    return {"lower_risk": _fraction(lower), "pairs": records,
            "method": "best of deterministic greedy and declared four-pair disjoint bounds"}


def _four_pair_reference(geometry: Any) -> dict[str, Any]:
    condition = geometry.kappa == 2 and all(
        geometry.squared_distances[i][j] <= Fraction(1, 500) ** 2
        for i, j in _REFERENCE_PAIRS)
    if not condition:
        return {"applicable": False}
    means = population_means()
    lower = irreducible_risk() + sum(
        (max(Fraction(0), abs(means[i] - means[j]) - Fraction(1, 250)) ** 2
         for i, j in _REFERENCE_PAIRS), Fraction(0)) / 32
    return {"applicable": True,
            "pairs": [list(pair) for pair in _REFERENCE_PAIRS],
            "verified_distance_upper": _fraction(Fraction(1, 500)),
            "lower_risk": _fraction(lower)}


def _build_audit(geometry: Any, witness: dict[str, Any],
                 predictions: tuple[Fraction, ...] | None,
                 block_sizes: tuple[int, ...]) -> dict[str, Any]:
    from .lipschitz_class import membership, verify_weighted_minimum

    _validate_geometry(geometry)
    noise = irreducible_risk()
    means = population_means()
    lower_dual = verify_weighted_minimum(
        geometry, (Fraction(1, 16),) * 16, means, witness, within_sse=noise)
    primal_records = witness["primal_predictions"]
    primal = {}
    for record in primal_records:
        history_id = record["history_id"]
        if type(history_id) is not int or history_id in primal or not 0 <= history_id < 16:
            raise ValueError("population primal witness must cover each history once")
        primal[history_id] = _parse_fraction(record["prediction"])
    if set(primal) != set(range(16)):
        raise ValueError("population primal witness is missing a history")
    primal_values = tuple(primal[index] for index in range(16))
    primal_membership = membership(geometry, primal_values)
    if not primal_membership["admissible"]:
        raise ValueError("population primal witness is not admissible")
    upper_primal = exact_population_risk(primal_values)
    if upper_primal != _parse_fraction(witness["upper_sse"]):
        raise ValueError("population primal objective disagrees with direct 64-case enumeration")

    analytic = _analytic_class(geometry, block_sizes)
    pairs = _pair_lower_bound(geometry)
    four_pair = _four_pair_reference(geometry)
    lower_sources = {
        "irreducible_risk": noise,
        "verified_optimization": lower_dual,
        "disjoint_pairs": _parse_fraction(pairs["lower_risk"]),
    }
    upper_sources = {
        "verified_primal": upper_primal,
        "constant_decoder": exact_population_risk((Fraction(1, 2),) * 16),
    }
    if analytic["applicable"]:
        reference = _parse_fraction(analytic["optimal_risk"])
        lower_sources["analytic_partition"] = reference
        upper_sources["analytic_partition"] = reference
    bayes_membership = membership(geometry, means)
    if bayes_membership["admissible"]:
        upper_sources["bayes_decoder"] = noise
    fitted_risk = None
    fitted_membership = None
    if predictions is not None:
        fitted_risk = exact_population_risk(predictions)
        fitted_membership = membership(geometry, predictions)
        if fitted_membership["admissible"]:
            upper_sources["fitted_decoder"] = fitted_risk
    lower = max(lower_sources.values())
    upper = min(upper_sources.values())
    if lower > upper:
        raise ValueError("population lower bound exceeds an independently feasible upper bound")
    return {
        "schema": AUDIT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "geometry_sha256": geometry.fingerprint,
        "scope": "known-law population audit; not a calibration confidence certificate",
        "history_count": 16,
        "block_sizes": list(block_sizes),
        "irreducible_risk": _fraction(noise),
        "conditional_means": [_fraction(value) for value in means],
        "optimal_risk_lower": _fraction(lower),
        "optimal_risk_upper": _fraction(upper),
        "optimal_risk_gap": _fraction(upper - lower),
        "optimal_risk_status": "exact" if lower == upper else "verified_interval",
        "lower_sources": {name: _fraction(value) for name, value in lower_sources.items()},
        "upper_sources": {name: _fraction(value) for name, value in upper_sources.items()},
        "population_optimization_witness": witness,
        "primal_admissibility": primal_membership,
        "bayes_admissibility": bayes_membership,
        "fitted_risk": None if fitted_risk is None else _fraction(fitted_risk),
        "fitted_admissibility": fitted_membership,
        "analytic_partition": analytic,
        "pair_lower_bound": pairs,
        "four_pair_reference": four_pair,
    }


def audit_class(geometry: Any, predictions: Sequence[Fraction] | None = None, *,
                block_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64),
                sweeps: int = 200) -> dict[str, Any]:
    """Audit a frozen finite-support class after its sample certificate is issued.

    The numerical optimizer proposes a dual and a feasible primal.  The audit
    independently enumerates the population risk of the primal and also checks
    analytical lower bounds.  Only equality of verified endpoints is reported
    as an exact optimum.
    """
    from .lipschitz_class import certify_weighted_minimum

    _validate_geometry(geometry)
    if not isinstance(block_sizes, Sequence) or isinstance(block_sizes, (str, bytes)):
        raise TypeError("block_sizes must be a sequence of integers")
    sizes = tuple(_block_size(value) for value in block_sizes)
    if len(set(sizes)) != len(sizes):
        raise ValueError("block sizes must be distinct")
    values = None if predictions is None else _exact_predictions(predictions)
    witness = certify_weighted_minimum(
        geometry, (Fraction(1, 16),) * 16, population_means(),
        within_sse=irreducible_risk(), sweeps=sweeps)
    return _build_audit(geometry, witness, values, sizes)


def verify_class_audit(geometry: Any, record: dict[str, Any],
                       predictions: Sequence[Fraction] | None = None) -> dict[str, Any]:
    """Recheck a retained audit without fitting or rerunning the optimizer."""
    try:
        if type(record) is not dict or record.get("schema") != AUDIT_SCHEMA:
            raise ValueError("unsupported class population audit record")
        if record.get("geometry_sha256") != geometry.fingerprint:
            raise ValueError("population audit geometry identity mismatch")
        block_sizes = record["block_sizes"]
        if type(block_sizes) is not list:
            raise ValueError("block_sizes must be a list")
        sizes = tuple(_block_size(value) for value in block_sizes)
        if len(set(sizes)) != len(sizes):
            raise ValueError("block sizes must be distinct")
        values = None if predictions is None else _exact_predictions(predictions)
        expected = _build_audit(geometry, record["population_optimization_witness"], values, sizes)
        if record != expected:
            raise ValueError("population audit fields disagree with independent recomputation")
        return {"verified": True, "optimal_risk_status": expected["optimal_risk_status"],
                "failure": None}
    except (ValueError, TypeError, KeyError, ArithmeticError, AttributeError) as error:
        return {"verified": False, "optimal_risk_status": None,
                "failure": {"code": "invalid_class_population_audit", "message": str(error)}}


__all__ = [
    "AUDIT_SCHEMA", "HIERARCHY_SCHEMA", "population_means", "irreducible_risk",
    "exact_population_risk", "hierarchy_reference", "audit_class", "verify_class_audit",
]
