"""Verify the exact benchmark audit of the supplied frozen certificate family.

From the extracted project root:
    python -m examples.population_audit_walkthrough

This reads and verifies existing evidence. It recomputes exact finite-support
population risks; it does not sample data, fit models, search confidence roots,
change issued endpoints, or select a winning representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", type=Path, default=project / "audits/primary/audit.json")
    parser.add_argument("--certificate", type=Path, default=project / "certificates/primary/certificate.json")
    parser.add_argument("--bundle", type=Path, default=project / "data/primary")
    parser.add_argument("--freeze", type=Path, default=project / "frozen/primary/freeze.json")
    args = parser.parse_args()

    from kahm_demo.audit_workflow import verify_audit_bundle

    result = verify_audit_bundle(args.audit, certificate_path=args.certificate,
                                dataset_bundle=args.bundle, freeze_path=args.freeze)
    result["primary_population_mse_tolerance"] = "1/10"
    result["risk_reference"] = "exact expectation under the stipulated finite benchmark law"
    result["issued_certificate_scope"] = "original 95% simultaneous coverage under the declared iid evaluation premises, conditional on development"
    result["coverage_frequency_estimated"] = False
    result["new_confidence_guarantee_issued"] = False
    result["representation_superiority_claimed"] = False
    result["risk_decomposition"] = "full-history Bayes MSE 9/160 plus exact mean squared deviation of this frozen predictor from the conditional mean"
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
