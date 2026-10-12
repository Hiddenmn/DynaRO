"""Pure, count-matched action schedules for prospective TRAIN-only generation.

This module reads no project data and starts no simulation, training, or evaluation.
The paired sequences contain exactly the same number of applications of each
measure; only temporal order differs.  Each schedule is a sequence of simulator
steps, not a calibrated physical injection programme.
"""

from __future__ import annotations

from collections import Counter


def paired_action_orders(
    action_a: str, action_b: str, horizon: int, *, pattern: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a schedule and its time reversal at a supported horizon.

    ``single_switch`` reproduces the existing 2:3 exposure design.
    ``alternating`` retains the older four-segment, 1:1 exposure design.
    ``dose_matched_multiswitch`` uses three switches while preserving 2:3
    exposure, so switch count is not confounded with total measure counts.
    Nothing in this construction claims that a simulator step is physical time.
    """
    if not isinstance(action_a, str) or not isinstance(action_b, str):
        raise TypeError("Action names must be strings")
    if not action_a or not action_b or action_a == action_b:
        raise ValueError("Two distinct, nonempty measure names are required")
    if type(horizon) is not int or horizon < 25 or horizon % 5:
        raise ValueError("Horizon must be an integer >=25 divisible by 5")
    if pattern == "single_switch":
        segments = ((action_a, 2 * horizon // 5),
                    (action_b, 3 * horizon // 5))
    elif pattern == "alternating":
        if horizon % 10:
            raise ValueError("Four-segment schedule needs a horizon divisible by 10")
        segments = ((action_a, horizon // 5),
                    (action_b, 3 * horizon // 10),
                    (action_a, 3 * horizon // 10),
                    (action_b, horizon // 5))
    elif pattern == "dose_matched_multiswitch":
        count_a, count_b = 2 * horizon // 5, 3 * horizon // 5
        segments = ((action_a, count_a // 2),
                    (action_b, count_b // 2),
                    (action_a, count_a - count_a // 2),
                    (action_b, count_b - count_b // 2))
    else:
        raise ValueError("Unknown schedule pattern")
    forward = tuple(name for name, count in segments for _ in range(count))
    reversed_order = forward[::-1]
    if (len(forward) != horizon or forward == reversed_order
            or Counter(forward) != Counter(reversed_order)):
        raise AssertionError("Count-matched reversal failed")
    return forward, reversed_order


def paired_tail_extensions(
    action_a: str, action_b: str, horizon: int, *, prefix_horizon: int = 50
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep a known paired prefix, then hold each schedule's terminal measure.

    This changes rollout length without moving the original switch or adding
    switches. Beyond the prefix, the two schedules no longer have identical
    measure counts. Their contrast is a schedule response, not an equal-dose
    order effect. No simulator, dataset or model is used here.
    """
    prefix_a, prefix_b = paired_action_orders(
        action_a, action_b, prefix_horizon, pattern="single_switch"
    )
    if type(horizon) is not int or horizon < prefix_horizon:
        raise ValueError("Extension horizon must be an integer >= prefix horizon")
    extra = horizon - prefix_horizon
    return (prefix_a + (prefix_a[-1],) * extra,
            prefix_b + (prefix_b[-1],) * extra)


def action_segments(actions: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    """Run-length description for a schedule, without changing any step."""
    if (not isinstance(actions, tuple) or not actions
            or any(not isinstance(name, str) or not name for name in actions)):
        raise ValueError("Expected a nonempty tuple of nonempty measure names")
    segments: list[tuple[str, int]] = []
    for action in actions:
        if segments and action == segments[-1][0]:
            name, count = segments[-1]
            segments[-1] = (name, count + 1)
        else:
            segments.append((action, 1))
    return tuple(segments)


def _self_test() -> None:
    for horizon in (25, 50, 75, 100):
        first, second = paired_action_orders("S1", "S4", horizon,
                                             pattern="single_switch")
        assert first == ("S1",) * (2 * horizon // 5) + ("S4",) * (3 * horizon // 5)
        assert second == first[::-1]
        assert sum(first[i] != first[i - 1] for i in range(1, horizon)) == 1
        multi_a, multi_b = paired_action_orders(
            "S1", "S4", horizon, pattern="dose_matched_multiswitch"
        )
        assert Counter(multi_a) == Counter(first) == Counter(multi_b)
        assert multi_b == multi_a[::-1]
        assert len(action_segments(multi_a)) == len(action_segments(multi_b)) == 4
    for horizon in (50, 100):
        first, second = paired_action_orders("S1", "S4", horizon,
                                             pattern="alternating")
        assert second == first[::-1]
        assert Counter(first) == {"S1": horizon // 2, "S4": horizon // 2}
        assert sum(first[i] != first[i - 1] for i in range(1, horizon)) == 3
    prefix_a, prefix_b = paired_action_orders("S1", "S4", 50,
                                            pattern="single_switch")
    for horizon in (50, 51, 75, 100):
        extended_a, extended_b = paired_tail_extensions("S1", "S4", horizon)
        assert len(extended_a) == len(extended_b) == horizon
        assert extended_a[:50] == prefix_a and extended_b[:50] == prefix_b
        assert action_segments(extended_a) == (("S1", 20), ("S4", horizon - 20))
        assert action_segments(extended_b) == (("S4", 30), ("S1", horizon - 30))
        if horizon > 50:
            assert Counter(extended_a) != Counter(extended_b)
        else:
            assert (extended_a, extended_b) == (prefix_a, prefix_b)
    for invalid in (49, 100.0, True):
        try:
            paired_tail_extensions("S1", "S4", invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid extension horizon accepted")
    for invalid in (0, 24, 26, 35.0, True):
        try:
            paired_action_orders("S1", "S4", invalid, pattern="single_switch")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid horizon accepted: {invalid}")


if __name__ == "__main__":
    _self_test()
    print("long_horizon_schedules self-test: PASS")
