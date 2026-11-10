"""Inspect the saved Gaussian KAHM features; run from the project directory.

    python examples/encoder_walkthrough.py

Install the package first, or use:
    python -m pip install .
"""

from pathlib import Path

import numpy as np

from kahm_demo.datasets import load_development
from kahm_demo.encoder_state import load_encoder


def main():
    project = Path(__file__).resolve().parents[1]
    bank = load_encoder(project / "encoders" / "primary" / "cap_2.json")
    history = [0.25, 0.75]
    row = bank.encode_one(history)
    print("Observed history:", history)
    print("Standardized history:", row.raw)
    print("Memberships, partitions of size 2 and 4:", row.kahm_mem)
    print("Local enrichment:", row.kahm_local)
    print("MEM equals the LOCAL prefix:", np.array_equal(row.kahm_mem, row.kahm_local[:6]))
    print("Partition membership sums:", row.kahm_mem[:2].sum(), row.kahm_mem[2:].sum())

    training = load_development(project / "data" / "primary", "training")
    x = [episode.x for episode in training.episodes]
    print("Training feature shapes:", {kind: bank.transform(x, kind).shape
                                       for kind in ("raw", "kahm_mem", "kahm_local")})
    print("These are encoder features. Forecast decoders and certificates are later steps.")


if __name__ == "__main__":
    main()
