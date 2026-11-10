"""Load the fixed public specification and check implementation alignment."""

from __future__ import annotations

from fractions import Fraction
from importlib.resources import files
import hashlib
import json

CONTRACT_SHA256 = "83a9fa471044e967d2055e7b507132749b7e64ab1a2788eac71024d83c52b4f7"
CANDIDATE_IDS = ("raw", "kahm_mem", "kahm_local")
PRIMARY_CALIBRATION_COUNT = 4096


def load_contract() -> dict:
    contents = files("kahm_demo").joinpath("specification.json").read_bytes()
    if hashlib.sha256(contents).hexdigest() != CONTRACT_SHA256:
        raise ValueError("the bundled scientific contract has changed")
    return json.loads(contents)


def _fraction(value: dict) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def validate_contract() -> dict:
    from .law import BENCHMARK_ID, OBSERVATIONS

    contract = load_contract()
    if contract["benchmark"]["id"] != BENCHMARK_ID:
        raise ValueError("benchmark identity differs from the scientific contract")
    if tuple(map(_fraction, contract["benchmark"]["observation_values"])) != OBSERVATIONS:
        raise ValueError("observation map differs from the scientific contract")
    if tuple(c["id"] for c in contract["candidates"]) != CANDIDATE_IDS:
        raise ValueError("candidate family differs from the scientific contract")
    cert = contract["certificate"]
    if cert["independent_unit_count"] != PRIMARY_CALIBRATION_COUNT:
        raise ValueError("calibration count differs from the scientific contract")
    if 3 * _fraction(cert["per_candidate_failure_probability"]) != _fraction(cert["total_failure_probability"]):
        raise ValueError("confidence allocation is inconsistent")
    if _fraction(contract["claim_scope"]["primary_mse_tolerance"]) != Fraction(1, 10):
        raise ValueError("primary tolerance differs from the scientific contract")
    return contract

