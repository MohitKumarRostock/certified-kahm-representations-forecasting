"""Load and inspect a verified frozen predictor without touching calibration.

From the extracted project root:
    python -m examples.frozen_forecast_walkthrough

With kahm_demo installed, this file can also be run directly. The default
freeze is frozen/primary/freeze.json; the default candidate is kahm_local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze", type=Path, default=project / "frozen/primary/freeze.json")
    parser.add_argument("--candidate", choices=("raw", "kahm_mem", "kahm_local"), default="kahm_local")
    args = parser.parse_args()
    if not args.freeze.is_file():
        parser.error("a completed model freeze is required; supply --freeze or create the freeze first")

    from kahm_demo.freeze import load_frozen_forecaster

    # Loading verifies the full self-contained frozen snapshot and returns the
    # actual model instance whose state and supported outputs were checked.
    predictor = load_frozen_forecaster(args.freeze, args.candidate)
    history = [0.25, 0.75]
    result = predictor.inspect_one(history)
    repeated = predictor.predict([history, history])
    if any(float(value).hex() != result.prediction.hex() for value in repeated):
        raise RuntimeError("frozen scalar and batch prediction replay disagree")
    print(json.dumps({
        "status": "frozen",
        "candidate_id": predictor.candidate_id,
        "bank_id": predictor.bank_id,
        "freeze_record_sha256": predictor.freeze_record_sha256,
        "calibration_protocol_sha256": predictor.calibration_protocol_sha256,
        "history_order": ["previous_observation", "current_observation"],
        "history": history,
        "prediction": result.prediction,
        "prediction_hex": result.prediction.hex(),
        "unclipped_prediction": result.unclipped_prediction,
        "clipped": result.clipped,
        "encoder_fallbacks": list(result.encoder_fallbacks),
        "repeated_row_predictions": [float(value) for value in repeated],
        "scalar_batch_exact_match": True,
        "calibration_outcomes_accessed": False,
        "risk_certificate_checked": False,
        "scope": "Frozen model inference only; use verify-certificate to inspect the risk assessment.",
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
