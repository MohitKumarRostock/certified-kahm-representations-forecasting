"""Exact specification of the v0.1 four-state forecasting benchmark.

The probability law is analytical. Sampling lives in a separate module so that
the integer-ticket sampler and these rational probabilities can be checked
against one another independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


BENCHMARK_ID = "four_state_order2_modular_v1"
OBSERVATIONS = (Fraction(0), Fraction(1, 4), Fraction(3, 4), Fraction(1))
OBSERVATION_FLOATS = tuple(float(value) for value in OBSERVATIONS)
HISTORY_KEYS = tuple((a, b) for a in range(4) for b in range(4))


def _validate_state(value: object, name: str) -> int:
    """Validate a state label without accepting booleans as integers."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer state label, not {type(value).__name__}")
    if not 0 <= value < 4:
        raise ValueError(f"{name} must be one of 0, 1, 2, 3")
    return value


def stationary_history_probability(a: int, b: int) -> Fraction:
    """Return the exact stationary mass of an ordered two-state history."""
    _validate_state(a, "previous_state")
    _validate_state(b, "current_state")
    return Fraction(1, 16)


def transition_probability(a: int, b: int, c: int) -> Fraction:
    """Return P(next_state=c | previous_state=a, current_state=b)."""
    _validate_state(a, "previous_state")
    _validate_state(b, "current_state")
    _validate_state(c, "next_state")
    return Fraction(17, 20) if c == (a + b) % 4 else Fraction(1, 20)


def joint_probability(a: int, b: int, c: int) -> Fraction:
    """Return the exact probability of one independent episode's state triple."""
    return stationary_history_probability(a, b) * transition_probability(a, b, c)


def conditional_mean(a: int, b: int) -> Fraction:
    """Return the analytical, rational Bayes prediction for a history.

    This is not the floating-point approximation that an implemented predictor
    might return. That distinction matters when checking exact risk equality.
    """
    _validate_state(a, "previous_state")
    _validate_state(b, "current_state")
    return Fraction(1, 10) + Fraction(4, 5) * OBSERVATIONS[(a + b) % 4]


@dataclass(frozen=True)
class Episode:
    """One sampled history and successor, with a nonnegative local row ID."""

    episode_id: int
    previous_state: int
    current_state: int
    next_state: int

    def __post_init__(self) -> None:
        if type(self.episode_id) is not int:
            raise TypeError("episode_id must be an integer, not a boolean or other type")
        if self.episode_id < 0:
            raise ValueError("episode_id must be nonnegative")
        _validate_state(self.previous_state, "previous_state")
        _validate_state(self.current_state, "current_state")
        _validate_state(self.next_state, "next_state")

    @property
    def x(self) -> tuple[float, float]:
        """The previous and current observations, in that order."""
        return (
            OBSERVATION_FLOATS[self.previous_state],
            OBSERVATION_FLOATS[self.current_state],
        )

    @property
    def y(self) -> float:
        return OBSERVATION_FLOATS[self.next_state]

    @property
    def history_id(self) -> int:
        return 4 * self.previous_state + self.current_state


__all__ = [
    "BENCHMARK_ID",
    "OBSERVATIONS",
    "OBSERVATION_FLOATS",
    "HISTORY_KEYS",
    "Episode",
    "transition_probability",
    "joint_probability",
    "conditional_mean",
    "stationary_history_probability",
]
