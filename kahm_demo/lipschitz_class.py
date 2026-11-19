"""Verified finite-support Lipschitz class optimization.

Floating point coordinate ascent proposes dual multipliers only. Every bound
returned here is recomputed with exact rational arithmetic. Euclidean distances
are enclosed on a dyadic grid; upper enclosures are used in the subtracted dual
penalty. A feasible primal table gives a separately checked upper objective.

All sample occurrences retain their weights. Exactly coincident feature rows
(or every row when kappa is zero) are merged before optimization. The engine
certifies a finite quadratic problem, not a statistical confidence statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import combinations
import math
import re

import numpy as np

from ._util import canonical_json_bytes


SCHEMA = "kahm.lipschitz_quadratic_certificate.v1"
MAXIMUM_STATES = 256
MAXIMUM_DIMENSIONS = 4096
MAXIMUM_INTEGER_BITS = 8192
MAXIMUM_SWEEPS = 10000


def as_fraction(value) -> Fraction:
    """Interpret finite Python binary64 values exactly; reject booleans."""
    if type(value) is Fraction:
        result = value
    elif type(value) is int:
        result = Fraction(value)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite numerical input")
        result = Fraction(value)
    elif type(value) is str and re.fullmatch(r"-?0x[0-9a-f]+\.[0-9a-f]*p[+-][0-9]+", value):
        parsed = float.fromhex(value)
        if not math.isfinite(parsed) or parsed.hex() != value:
            raise ValueError("a binary64 hexadecimal input must be canonical and finite")
        result = Fraction(parsed)
    else:
        raise TypeError("numbers must be Fractions, integers, finite floats, or canonical float.hex strings")
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > MAXIMUM_INTEGER_BITS:
        raise ValueError("rational input exceeds the integer-size limit")
    return result


def fraction_record(value: Fraction) -> dict:
    value = as_fraction(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def parse_fraction_record(value) -> Fraction:
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        raise ValueError("invalid fraction record")
    a, b = value["numerator"], value["denominator"]
    if (type(a) is not str or type(b) is not str or len(a) > 2500 or len(b) > 2500
            or re.fullmatch(r"0|-?[1-9][0-9]*", a) is None
            or re.fullmatch(r"[1-9][0-9]*", b) is None):
        raise ValueError("invalid fraction integer strings")
    result = as_fraction(Fraction(int(a), int(b)))
    if fraction_record(result) != value:
        raise ValueError("fraction record must be canonical")
    return result


def _digest(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sqrt_enclosure(value: Fraction, bits: int = 80) -> tuple[Fraction, Fraction]:
    """Enclose a nonnegative square root without floating point arithmetic."""
    value = as_fraction(value)
    if value < 0 or type(bits) is not int or not 16 <= bits <= 256:
        raise ValueError("square-root input or precision is invalid")
    scale = 1 << bits
    numerator = value.numerator << (2 * bits)
    floor = math.isqrt(numerator // value.denominator)
    low = Fraction(floor, scale)
    high = low if floor * floor * value.denominator == numerator else Fraction(floor + 1, scale)
    return low, high


@dataclass(frozen=True, slots=True)
class Geometry:
    features: tuple[tuple[Fraction, ...], ...]
    kappa: Fraction
    distance_bits: int
    squared_distances: tuple[tuple[Fraction, ...], ...]
    distance_lower: tuple[tuple[Fraction, ...], ...]
    distance_upper: tuple[tuple[Fraction, ...], ...]
    fingerprint: str


def make_geometry(features, kappa, distance_bits: int = 80) -> Geometry:
    rows = tuple(tuple(as_fraction(x) for x in row) for row in features)
    if not 1 <= len(rows) <= MAXIMUM_STATES:
        raise ValueError("the feature table has an unsupported number of states")
    width = len(rows[0])
    if not 1 <= width <= MAXIMUM_DIMENSIONS or any(len(row) != width for row in rows):
        raise ValueError("feature rows must have the same positive width")
    kappa = as_fraction(kappa)
    if kappa < 0:
        raise ValueError("kappa must be nonnegative")
    # Validate precision even for singleton feature tables.
    sqrt_enclosure(Fraction(0), distance_bits)
    size = len(rows)
    squared = [[Fraction(0) for _ in rows] for _ in rows]
    lower = [[Fraction(0) for _ in rows] for _ in rows]
    upper = [[Fraction(0) for _ in rows] for _ in rows]
    for i, j in combinations(range(size), 2):
        q = sum((a - b) ** 2 for a, b in zip(rows[i], rows[j]))
        if max(q.numerator.bit_length(), q.denominator.bit_length()) > MAXIMUM_INTEGER_BITS:
            raise ValueError("squared distance exceeds the integer-size limit")
        lo, hi = sqrt_enclosure(q, distance_bits)
        squared[i][j] = squared[j][i] = q
        lower[i][j] = lower[j][i] = lo
        upper[i][j] = upper[j][i] = hi
    identity = {
        "schema": "kahm.finite_lipschitz_geometry.v1",
        "features": [[fraction_record(x) for x in row] for row in rows],
        "metric": "euclidean", "kappa": fraction_record(kappa),
        "output_interval": [0, 1], "distance_bits": distance_bits,
    }
    return Geometry(rows, kappa, distance_bits, tuple(map(tuple, squared)),
                    tuple(map(tuple, lower)), tuple(map(tuple, upper)), _digest(identity))


def _check_geometry(geometry):
    if type(geometry) is not Geometry:
        raise TypeError("a Geometry produced by make_geometry is required")


def membership(geometry: Geometry, predictions) -> dict:
    """Check all support pairs using exact squared inequalities."""
    _check_geometry(geometry)
    values = tuple(as_fraction(x) for x in predictions)
    if len(values) != len(geometry.features):
        raise ValueError("prediction and feature table lengths differ")
    violations = []
    for i, j in combinations(range(len(values)), 2):
        if (values[i] - values[j]) ** 2 > geometry.kappa ** 2 * geometry.squared_distances[i][j]:
            violations.append({"left": i, "right": j})
    output_valid = all(0 <= x <= 1 for x in values)
    return {
        "schema": "kahm.finite_class_membership.v1",
        "geometry_sha256": geometry.fingerprint,
        "predictions_sha256": _digest([fraction_record(x) for x in values]),
        "admissible": output_valid and not violations,
        "output_range_valid": output_valid,
        "checked_pairs": len(values) * (len(values) - 1) // 2,
        "violations": violations,
        "scope": "complete supplied finite support; a matching Lipschitz extension exists",
    }


def _weighted_problem(geometry, weights, means, within_sse):
    _check_geometry(geometry)
    weights = tuple(as_fraction(x) for x in weights)
    means = tuple(as_fraction(x) for x in means)
    within_sse = as_fraction(within_sse)
    size = len(geometry.features)
    if len(weights) != size or len(means) != size:
        raise ValueError("weights and means must cover the feature table")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be nonnegative with positive total")
    if any(not 0 <= x <= 1 for x in means) or within_sse < 0:
        raise ValueError("means must lie in [0,1] and within_sse must be nonnegative")
    identity = {"weights": [fraction_record(x) for x in weights],
                "means": [fraction_record(x) for x in means],
                "within_sse": fraction_record(within_sse)}
    merged = {}
    for h, weight in enumerate(weights):
        if weight:
            key = geometry.features[h] if geometry.kappa else ()
            if key not in merged:
                merged[key] = [h, Fraction(0), Fraction(0), Fraction(0)]
            row = merged[key]
            row[1] += weight
            row[2] += weight * means[h]
            row[3] += weight * means[h] ** 2
    groups = []
    for h, weight, total, squares in merged.values():
        mean = total / weight
        within_sse += squares - total ** 2 / weight
        groups.append((h, weight, mean))
    return weights, means, within_sse, groups, _digest(identity)


def _empirical_statistics(geometry, history_ids, targets):
    _check_geometry(geometry)
    histories = tuple(history_ids)
    ys = tuple(as_fraction(x) for x in targets)
    if not histories or len(histories) != len(ys) or len(histories) > 1000000:
        raise ValueError("the nonempty history and target samples must have equal lengths")
    size = len(geometry.features)
    if any(type(h) is not int or not 0 <= h < size for h in histories):
        raise ValueError("history identifiers are out of range")
    if any(not 0 <= y <= 1 for y in ys):
        raise ValueError("targets must lie in [0,1]")
    counts = [Fraction(0) for _ in range(size)]
    totals = [Fraction(0) for _ in range(size)]
    squares = [Fraction(0) for _ in range(size)]
    for h, y in zip(histories, ys):
        counts[h] += 1
        totals[h] += y
        squares[h] += y * y
    means = [total / count if count else Fraction(0) for total, count in zip(totals, counts)]
    within = sum(square - total ** 2 / count for count, total, square in zip(counts, totals, squares) if count)
    identity = _digest([[h, fraction_record(y)] for h, y in zip(histories, ys)])
    return counts, means, within, identity, len(histories)


def _propose(geometry, groups, sweeps):
    if type(sweeps) is not int or not 0 <= sweeps <= MAXIMUM_SWEEPS:
        raise ValueError("the coordinate-sweep budget is invalid")
    edges = [(i, j) for i, j in combinations(range(len(groups)), 2)
             if geometry.kappa * geometry.distance_upper[groups[i][0]][groups[j][0]] < 1]
    work = {"algorithm": "cyclic_dual_soft_threshold_v1", "maximum_sweeps": sweeps,
            "completed_sweeps": 0, "termination": "no_active_edges", "failure": None}
    if not edges:
        return [], work
    work["termination"] = "sweep_limit"
    try:
        inverse = np.array([1 / float(g[1]) for g in groups], dtype=np.float64)
        means = np.array([float(g[2]) for g in groups], dtype=np.float64)
        caps = np.array([float(geometry.kappa * geometry.distance_upper[groups[i][0]][groups[j][0]])
                         for i, j in edges], dtype=np.float64)
        multipliers = np.zeros(len(edges), dtype=np.float64)
        s = np.zeros(len(groups), dtype=np.float64)
        if not np.isfinite(inverse).all() or not np.isfinite(caps).all():
            raise FloatingPointError("non-finite proposal coefficients")
        for iteration in range(sweeps):
            change = 0.0
            for e, (i, j) in enumerate(edges):
                old = multipliers[e]
                curvature = 0.5 * (inverse[i] + inverse[j])
                gradient = means[i] - means[j] - 0.5 * (s[i] * inverse[i] - s[j] * inverse[j])
                center = old + gradient / curvature
                candidate = math.copysign(max(0.0, abs(center) - caps[e] / curvature), center)
                delta = candidate - old
                if not math.isfinite(candidate):
                    raise FloatingPointError("non-finite proposed multiplier")
                multipliers[e] = candidate
                s[i] += delta
                s[j] -= delta
                change = max(change, abs(delta))
            work["completed_sweeps"] = iteration + 1
            if change <= 1e-12 * (1.0 + float(np.max(np.abs(multipliers)))):
                work["termination"] = "numerical_stationarity"
                break
        return [{"left": groups[i][0], "right": groups[j][0], "multiplier_hex": float(multipliers[e]).hex()}
                for e, (i, j) in enumerate(edges) if multipliers[e] != 0], work
    except (ArithmeticError, ValueError, OverflowError) as error:
        work["termination"] = "proposal_failure"
        work["failure"] = str(error)
        return [], work


def _dual_value(geometry, groups, within, edges):
    by_id = {h: (weight, mean) for h, weight, mean in groups}
    s = {h: Fraction(0) for h in by_id}
    penalty = Fraction(0)
    seen = set()
    if type(edges) is not list or len(edges) > len(groups) * (len(groups) - 1) // 2:
        raise ValueError("invalid dual edge list")
    for row in edges:
        if type(row) is not dict or set(row) != {"left", "right", "multiplier_hex"}:
            raise ValueError("invalid dual edge record")
        i, j = row["left"], row["right"]
        if type(i) is not int or type(j) is not int or i not in by_id or j not in by_id or i >= j or (i, j) in seen:
            raise ValueError("dual edges must be distinct ordered pairs of occupied groups")
        multiplier = as_fraction(row["multiplier_hex"])
        if not multiplier:
            raise ValueError("zero dual multipliers must be omitted")
        seen.add((i, j))
        s[i] += multiplier
        s[j] -= multiplier
        penalty += geometry.kappa * geometry.distance_upper[i][j] * abs(multiplier)
    dual = within - penalty + sum(s[h] * mean - s[h] ** 2 / (4 * weight)
                                 for h, weight, mean in groups)
    return max(Fraction(0), dual), s


def _primal_table(geometry, weights, means, groups, s):
    total = sum(weights)
    center = sum(w * y for w, y in zip(weights, means)) / total
    proposed = {h: min(Fraction(1), max(Fraction(0), mean - s[h] / (2 * weight)))
                for h, weight, mean in groups}
    factor = Fraction(1)
    for (i, _, _), (j, _, _) in combinations(groups, 2):
        difference = abs(proposed[i] - proposed[j])
        if difference:
            factor = min(factor, geometry.kappa * geometry.distance_lower[i][j] / difference)
    feasible = {h: center + factor * (proposed[h] - center) for h, _, _ in groups}
    if factor < 1:
        # Rounded-down Euclidean distances need not satisfy the triangle
        # inequality. Their shortest-path closure is a metric dominated by
        # every original lower distance, so its Lipschitz envelopes are safe
        # for the actual Euclidean metric. This repairs small local violations
        # without contracting every prediction by the worst near-collision.
        ids = [h for h, _, _ in groups]
        cost = [[geometry.kappa * geometry.distance_lower[i][j] for j in ids] for i in ids]
        for k in range(len(ids)):
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate = cost[i][k] + cost[k][j]
                    if candidate < cost[i][j]:
                        cost[i][j] = cost[j][i] = candidate
        envelope = {
            h: (min(proposed[j] + cost[i][k] for k, j in enumerate(ids))
                + max(proposed[j] - cost[i][k] for k, j in enumerate(ids))) / 2
            for i, h in enumerate(ids)
        }
        shift = sum(weight * (mean - envelope[h]) for h, weight, mean in groups) / total
        envelope = {h: min(Fraction(1), max(Fraction(0), z + shift)) for h, z in envelope.items()}
        old_value = sum(weight * (feasible[h] - mean) ** 2 for h, weight, mean in groups)
        new_value = sum(weight * (envelope[h] - mean) ** 2 for h, weight, mean in groups)
        if new_value < old_value:
            feasible = envelope
    representative = {geometry.features[h] if geometry.kappa else (): h for h, _, _ in groups}
    return [{"history_id": h,
             "prediction": fraction_record(feasible[representative[geometry.features[h] if geometry.kappa else ()]])}
            for h, weight in enumerate(weights) if weight]


def _primal_value(geometry, weights, means, original_within, table):
    expected = {h for h, weight in enumerate(weights) if weight}
    if type(table) is not list or len(table) != len(expected):
        raise ValueError("primal table does not cover all positive-weight histories")
    values = {}
    for row in table:
        if type(row) is not dict or set(row) != {"history_id", "prediction"}:
            raise ValueError("invalid primal prediction record")
        h = row["history_id"]
        if type(h) is not int or h not in expected or h in values:
            raise ValueError("invalid or repeated primal history identifier")
        z = parse_fraction_record(row["prediction"])
        if not 0 <= z <= 1:
            raise ValueError("primal predictions must lie in [0,1]")
        values[h] = z
    for i, j in combinations(sorted(values), 2):
        if (values[i] - values[j]) ** 2 > geometry.kappa ** 2 * geometry.squared_distances[i][j]:
            raise ValueError("the proposed primal prediction table is not admissible")
    return original_within + sum(weights[h] * (values[h] - means[h]) ** 2 for h in values)


def certify_weighted_minimum(geometry: Geometry, weights, means, *,
                             within_sse=Fraction(0), sweeps: int = 200) -> dict:
    """Bound ``within_sse + sum_h weights[h]*(z[h]-means[h])**2``.

    Zero-weight histories impose no objective. Any feasible observed table has
    a clipped Lipschitz extension, so they need not be optimization variables.
    """
    original_within = as_fraction(within_sse)
    weights, means, within, groups, identity = _weighted_problem(geometry, weights, means, within_sse)
    edges, work = _propose(geometry, groups, sweeps)
    lower, s = _dual_value(geometry, groups, within, edges)
    primal = _primal_table(geometry, weights, means, groups, s)
    upper = _primal_value(geometry, weights, means, original_within, primal)
    if lower > upper:
        raise ArithmeticError("verified dual exceeds feasible primal objective")
    return {
        "schema": SCHEMA, "problem_type": "weighted",
        "geometry_sha256": geometry.fingerprint, "problem_sha256": identity,
        "weight_total": fraction_record(sum(weights)),
        "groups": [{"history_id": h, "weight": fraction_record(w), "mean": fraction_record(m)} for h, w, m in groups],
        "within_sse": fraction_record(within), "edges": edges,
        "lower_sse": fraction_record(lower), "upper_sse": fraction_record(upper),
        "primal_predictions": primal, "optimization": work,
    }


def verify_weighted_minimum(geometry: Geometry, weights, means, record, *,
                            within_sse=Fraction(0)) -> Fraction:
    """Recompute both mathematical bounds; never run the proposal optimizer."""
    original_within = as_fraction(within_sse)
    weights, means, within, groups, identity = _weighted_problem(geometry, weights, means, within_sse)
    required = {"schema", "problem_type", "geometry_sha256", "problem_sha256", "weight_total",
                "groups", "within_sse", "edges", "lower_sse", "upper_sse", "primal_predictions", "optimization"}
    if type(record) is not dict or set(record) != required:
        raise ValueError("invalid quadratic certificate schema")
    expected_groups = [{"history_id": h, "weight": fraction_record(w), "mean": fraction_record(m)} for h, w, m in groups]
    if (record["schema"] != SCHEMA or record["problem_type"] != "weighted"
            or record["geometry_sha256"] != geometry.fingerprint
            or record["problem_sha256"] != identity
            or record["weight_total"] != fraction_record(sum(weights))
            or record["groups"] != expected_groups or record["within_sse"] != fraction_record(within)):
        raise ValueError("quadratic certificate inputs or grouped statistics disagree")
    work = record["optimization"]
    if (type(work) is not dict or set(work) != {"algorithm", "maximum_sweeps", "completed_sweeps", "termination", "failure"}
            or work["algorithm"] != "cyclic_dual_soft_threshold_v1"
            or type(work["maximum_sweeps"]) is not int or not 0 <= work["maximum_sweeps"] <= MAXIMUM_SWEEPS
            or type(work["completed_sweeps"]) is not int or not 0 <= work["completed_sweeps"] <= work["maximum_sweeps"]
            or work["termination"] not in {"no_active_edges", "sweep_limit", "numerical_stationarity", "proposal_failure"}
            or (work["failure"] is not None and type(work["failure"]) is not str)):
        raise ValueError("invalid optimization work record")
    lower, _ = _dual_value(geometry, groups, within, record["edges"])
    upper = _primal_value(geometry, weights, means, original_within, record["primal_predictions"])
    if (record["lower_sse"] != fraction_record(lower) or record["upper_sse"] != fraction_record(upper)
            or lower > upper):
        raise ValueError("reported objective endpoints do not match the verified witnesses")
    return lower


def certify_empirical_minimum(geometry: Geometry, history_ids, targets, *, sweeps: int = 200) -> dict:
    """Certify the minimum total squared error, keeping every occurrence."""
    weights, means, within, identity, count = _empirical_statistics(geometry, history_ids, targets)
    result = certify_weighted_minimum(geometry, weights, means, within_sse=within, sweeps=sweeps)
    result["problem_type"] = "empirical"
    result["sample_sha256"] = identity
    result["sample_count"] = count
    return result


def verify_empirical_minimum(geometry: Geometry, history_ids, targets, record) -> Fraction:
    weights, means, within, identity, count = _empirical_statistics(geometry, history_ids, targets)
    if (type(record) is not dict or record.get("problem_type") != "empirical"
            or record.get("sample_sha256") != identity or record.get("sample_count") != count
            or type(record.get("sample_count")) is not int):
        raise ValueError("empirical certificate sample binding disagrees")
    weighted = dict(record)
    weighted["problem_type"] = "weighted"
    weighted.pop("sample_sha256")
    weighted.pop("sample_count")
    return verify_weighted_minimum(geometry, weights, means, weighted, within_sse=within)
