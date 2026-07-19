import pytest

from auto_round.calibration.sampling import uniform_call_indices


@pytest.mark.parametrize("total_calls", [1, 4, 16])
def test_uniform_call_indices_retains_all_calls_within_limit(total_calls):
    assert uniform_call_indices(total_calls, 16) == tuple(range(total_calls))


def test_uniform_call_indices_selects_midpoint_for_one_call_limit():
    assert uniform_call_indices(10, 1) == (5,)


def test_uniform_call_indices_spans_sequence_deterministically():
    first = uniform_call_indices(160, 16)
    second = uniform_call_indices(160, 16)

    assert first == second
    assert len(first) == 16
    assert first[0] == 0
    assert first[-1] == 159
    assert tuple(sorted(set(first))) == first


@pytest.mark.parametrize("total_calls,max_calls", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_uniform_call_indices_rejects_non_positive_sizes(total_calls, max_calls):
    with pytest.raises(ValueError):
        uniform_call_indices(total_calls, max_calls)
