"""Terminal walkthrough of the fixed primary demonstration; inspection is default.

Run after installing the package, or from the project as
``python -m examples.guided_tutorial --project .``. This script never starts a
fresh experiment; ``kahm_demo reproduce`` is a separate explicit operation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kahm_demo.tutorial import history_from_observations, inspect_primary, verify_primary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=("inspect", "verify"), default="inspect")
    parser.add_argument("--history", type=float, nargs=2, default=[0.25, 0.75])
    args = parser.parse_args(argv)
    try:
        history = history_from_observations(args.history)
        action = inspect_primary if args.mode == "inspect" else verify_primary
        result = action(args.project, history=history)
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"Tutorial stopped: {error}\nNo fallback from verification to inspection was performed.\n")
    mode = result["mode"]
    print(f"KAHM guided tutorial | {mode}")
    print(json.dumps({key: result[key] for key in (
        "archive_identity_checked", "model_replayed", "certificate_reverified",
        "population_risk_recomputed")}, indent=2))
    print("\nPrimary risk results | " + mode)
    print("Predictor     q: calibration MSE    R: population MSE     U: issued upper bound  Certificate status   Coverage")
    for row in result["table"]:
        upper = row["issued_upper_endpoint_decimal_18"]
        print(f"{row['candidate_id']:12s}  {row['empirical_mse_decimal_18_approximate']:20s}  "
              f"{row['population_mse_decimal_18_approximate']:20s}  {str(upper):20s}  "
              f"{row['certificate_status']:20s} {row['realized_coverage']}")
    print("q and R are rounded displays. U is displayed upward; saved exact fractions govern decisions.")
    print("Under the stipulated iid evaluation premises, the three bounds have at least 95% simultaneous coverage, conditional on development.")
    print("The coverage column is a retained benchmark comparison, not an estimate of long-run coverage frequency.")
    print("\nSelected history and recorded forecasts | " + mode)
    print(json.dumps({key: result["history"][key] for key in (
        "state_labels", "observations", "recorded_predictions")}, indent=2))
    if result["model_replayed"]:
        print("Current-runtime replay matched the recorded forecast outputs.")
    features = result["history"]["recorded_encoder"]
    print("\nRecorded encoder features before decoder scaling | " + mode)
    print(json.dumps({key: features[key] for key in (
        "bank_id", "raw", "raw_meaning", "memberships", "enriched", "enriched_order")}, indent=2))
    print("Each partition normalizes separately. These memberships are geometry-induced features; they are not established latent-state posterior probabilities.")
    print("\nTo inspect another supported history, change --history. To run a new experiment explicitly:")
    print("python -m kahm_demo reproduce --output runs/tutorial --master-seed 20261110 --run-id tutorial-v0.1")
    print("See docs/TUTORIAL.md and docs/ADAPTING_TO_NEW_DATA.md. Fresh experiments retain every outcome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
