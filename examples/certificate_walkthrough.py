"""Verify the supplied assessment and explain its reported upper endpoints.

From the extracted project root:
    python -m examples.certificate_walkthrough

With kahm_demo installed, this file may also be run directly. This walkthrough
loads the archived calibration evidence and verifies it. It does not resample,
refit, run the endpoint search, or calculate model-specific population risks.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--certificate", type=Path,
                        default=project / "certificates/primary/certificate.json")
    parser.add_argument("--bundle", type=Path, default=project / "data/primary")
    parser.add_argument("--freeze", type=Path, default=project / "frozen/primary/freeze.json")
    args = parser.parse_args()

    from kahm_demo.certificate_workflow import verify_certificate_bundle

    result = verify_certificate_bundle(args.certificate, dataset_bundle=args.bundle,
                                       freeze_path=args.freeze)
    for row in result["candidates"]:
        mean = row["empirical_mse"]
        row["empirical_mse_approximate"] = float(Fraction(int(mean["numerator"]),
                                                        int(mean["denominator"])))
    result["simultaneous_confidence"] = "19/20 under the declared iid evaluation model, conditional on development"
    result["primary_population_mse_tolerance"] = "1/10"
    result["status_meanings"] = {
        "requirement_met": "A verified upper endpoint is at most the exact tolerance.",
        "unresolved": "A verified upper endpoint exceeds the tolerance; the actual risk may still meet it.",
        "certificate_unavailable": "No endpoint is available under the required numerical checks.",
    }
    result["forecast_interval_claimed"] = False
    result["representation_superiority_claimed"] = False
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
