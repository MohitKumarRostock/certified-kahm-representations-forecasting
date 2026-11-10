"""Inspect the three saved development forecasters without fitting or certification.

From the project root:
    python -m examples.forecast_walkthrough

With kahm_demo installed, this file can also be run as a script:
    python examples/forecast_walkthrough.py

The default inputs are models/primary/{raw,kahm_mem,kahm_local}.json.
Use --models-dir to inspect another saved development bundle. Development selects
forecasters; the final freeze and independent calibration are later steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project / "models" / "primary",
        help="directory containing raw.json, kahm_mem.json, and kahm_local.json",
    )
    args = parser.parse_args()
    candidates = ("raw", "kahm_mem", "kahm_local")
    paths = {name: args.models_dir / f"{name}.json" for name in candidates}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(
            "saved development models are required; complete development fitting "
            "or supply --models-dir. Missing: " + ", ".join(missing)
        )

    from kahm_demo.forecast_state import load_forecaster

    # Observed values, ordered as previous observation then current observation.
    history = [0.25, 0.75]
    records = []
    for candidate_id, path in paths.items():
        model = load_forecaster(path)
        if model.candidate_id != candidate_id:
            raise ValueError(f"{path} contains the wrong forecasting candidate")

        inspected = model.inspect_one(history)
        prediction = model.predict_one(history)
        # A repeated row illustrates that batch inference preserves occurrences
        # and uses exactly the same scalar prediction calculation.
        batch = model.predict([history, history])
        same_predictions = (
            prediction.hex() == inspected.prediction.hex()
            and all(float(value).hex() == prediction.hex() for value in batch)
        )
        if not same_predictions:
            raise RuntimeError("scalar and batch prediction replay disagree")
        records.append({
            "candidate_id": candidate_id,
            "bank_id": model.bank_id,
            "prediction": prediction,
            "prediction_hex": prediction.hex(),
            "unclipped_prediction": inspected.unclipped_prediction,
            "clipped": inspected.clipped,
            "encoder_fallbacks": list(inspected.encoder_fallbacks),
            "repeated_row_batch_predictions": [float(value) for value in batch],
            "scalar_and_batch_identical": same_predictions,
        })

    print(json.dumps({
        "status": "selected development forecasters; final model freeze is a separate operation",
        "history_order": ["previous_observation", "current_observation"],
        "history": history,
        "forecasters": records,
        "scope": "prediction inspection only; no fitting, oracle risk, or calibration",
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
