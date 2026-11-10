"""Complete deterministic computational forecasters for RAW, MEM, and LOCAL.

These objects are fitted and selected during development. The model freeze binds their
artifacts and canonical predictions into the final model-freeze protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder import KAHMEncoder
from .geometry import Standardizer
from .readout_features import ReadoutFeatures
from .ridge import RidgeModel

CANONICAL_FORECAST_METHOD = "single_history_numpy_float64_forecast_v1"
CANDIDATE_IDS = ("raw", "kahm_mem", "kahm_local")


@dataclass(frozen=True, slots=True)
class ForecastResult:
    prediction: float
    unclipped_prediction: float
    clipped: bool
    encoder_fallbacks: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class ForecastModel:
    candidate_id: str
    bank_id: str | None
    raw_standardizer: Standardizer | None
    encoder: KAHMEncoder | None
    readout: ReadoutFeatures
    ridge: RidgeModel

    def __post_init__(self):
        if self.candidate_id not in CANDIDATE_IDS:
            raise ValueError("unknown forecasting candidate")
        if self.candidate_id == "raw":
            if self.bank_id is not None or self.encoder is not None or not isinstance(self.raw_standardizer, Standardizer):
                raise ValueError("RAW must contain only its input standardizer")
            representation_width = self.raw_standardizer.input_dimension
        else:
            if self.raw_standardizer is not None or not isinstance(self.encoder, KAHMEncoder):
                raise ValueError("KAHM forecaster must contain its shared encoder")
            if self.bank_id != f"cap_{self.encoder.config.cap}":
                raise ValueError("bank identity differs from encoder cap")
            representation_width = self.encoder.feature_widths[self.candidate_id]
        if not isinstance(self.readout, ReadoutFeatures) or self.readout.input_width != representation_width:
            raise ValueError("readout feature map has the wrong representation width")
        if not isinstance(self.ridge, RidgeModel) or self.ridge.feature_count != self.readout.output_width:
            raise ValueError("ridge coefficients have the wrong feature width")
        if self.ridge.training_count != self.readout.training_count:
            raise ValueError("decoder training counts are inconsistent")

    @property
    def input_width(self) -> int:
        return self.raw_standardizer.input_dimension if self.candidate_id == "raw" else self.encoder.input_width

    def inspect_one(self, history) -> ForecastResult:
        if self.candidate_id == "raw":
            representation = self.raw_standardizer.transform_one(history)
            fallbacks = ()
        else:
            encoded = self.encoder.encode_one(history)
            representation = encoded.features(self.candidate_id)
            fallbacks = tuple({"model_index": index, **diagnostic}
                              for index, diagnostic in enumerate(encoded.model_diagnostics)
                              if diagnostic.get("used_fallback"))
        features = self.readout.transform_one(representation)
        unclipped = self.ridge.predict_unclipped_one(features)
        # Finite checking occurs in the ridge readout before range clipping.
        prediction = float(min(1.0, max(0.0, unclipped)))
        return ForecastResult(prediction, unclipped, prediction != unclipped, fallbacks)

    def predict_one(self, history) -> float:
        return self.inspect_one(history).prediction

    def predict_unclipped_one(self, history) -> float:
        return self.inspect_one(history).unclipped_prediction

    def predict(self, histories) -> np.ndarray:
        rows = np.asarray(histories)
        if rows.dtype.kind not in "iuf":
            raise TypeError("forecast histories must contain real integer or floating-point values")
        if rows.ndim != 2 or rows.shape[1] != self.input_width:
            raise ValueError("forecast histories have the wrong input width")
        rows = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(rows).all():
            raise ValueError("forecast histories must be finite")
        values = []
        cache = {}
        for row in rows:
            key = row.tobytes()
            if key not in cache:
                cache[key] = self.predict_one(row)
            values.append(cache[key])
        array = np.asarray(values, dtype=np.float64)
        return np.frombuffer(array.tobytes(), dtype=np.float64)
