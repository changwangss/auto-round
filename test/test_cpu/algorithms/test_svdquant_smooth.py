import pytest
import torch


def test_build_alpha_beta_candidates_matches_reference_order():
    from auto_round.algorithms.transforms.svdquant.smooth import build_alpha_beta_candidates

    assert build_alpha_beta_candidates(4) == [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
    ]
    assert len(build_alpha_beta_candidates(20)) == 39


@pytest.mark.parametrize("value", [True, 1, 1.5, None])
def test_build_alpha_beta_candidates_rejects_invalid_grid_count(value):
    from auto_round.algorithms.transforms.svdquant.smooth import build_alpha_beta_candidates

    with pytest.raises(ValueError, match="num_grids"):
        build_alpha_beta_candidates(value)


def test_absmax_channel_span_reduces_every_non_channel_dimension():
    from auto_round.algorithms.transforms.svdquant.smooth import absmax_channel_span

    tensor = torch.tensor(
        [
            [[1.0, -2.0], [3.0, 0.5], [-4.0, 1.0]],
            [[-2.0, 5.0], [0.25, -6.0], [1.5, 3.0]],
        ]
    )

    torch.testing.assert_close(absmax_channel_span(tensor, channels_dim=-1), torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(absmax_channel_span(tensor, channels_dim=1), torch.tensor([5.0, 6.0, 4.0]))


def test_build_smooth_scale_matches_absmax_reference():
    from auto_round.algorithms.transforms.svdquant.smooth import build_smooth_scale

    x_span = torch.tensor([16.0, 9.0, 4.0])
    w_span = torch.tensor([1.0, 4.0, 16.0])

    scale = build_smooth_scale(x_span, w_span, alpha=0.5, beta=0.5)

    torch.testing.assert_close(scale, x_span.pow(0.5) / w_span.pow(0.5))


def test_build_smooth_scale_identity_candidate():
    from auto_round.algorithms.transforms.svdquant.smooth import build_smooth_scale

    scale = build_smooth_scale(
        torch.tensor([0.0, 4.0, 9.0]),
        torch.tensor([0.0, 2.0, 3.0]),
        alpha=0.0,
        beta=0.0,
    )

    torch.testing.assert_close(scale, torch.ones(3))


def test_build_smooth_scale_replaces_zero_entries_with_one():
    from auto_round.algorithms.transforms.svdquant.smooth import build_smooth_scale

    scale = build_smooth_scale(
        torch.tensor([0.0, 4.0]),
        torch.tensor([1.0, 1.0]),
        alpha=0.5,
        beta=0.0,
    )

    torch.testing.assert_close(scale, torch.tensor([1.0, 2.0]))


def test_build_smooth_scale_falls_back_entire_scale_on_nonfinite_value():
    from auto_round.algorithms.transforms.svdquant.smooth import build_smooth_scale

    scale = build_smooth_scale(
        torch.tensor([4.0, 9.0]),
        torch.tensor([0.0, 1.0]),
        alpha=0.5,
        beta=0.5,
    )

    torch.testing.assert_close(scale, torch.ones(2))


def test_build_smooth_scale_clamps_span_bases_with_epsilon():
    from auto_round.algorithms.transforms.svdquant.smooth import build_smooth_scale

    scale = build_smooth_scale(
        torch.tensor([0.0, 4.0]),
        torch.tensor([0.0, 1.0]),
        alpha=0.5,
        beta=0.5,
        eps=1e-4,
    )

    torch.testing.assert_close(scale, torch.tensor([1.0, 2.0]))


def test_validate_smooth_scale_for_deployment_rejects_nonfinite_reciprocal():
    from auto_round.algorithms.transforms.svdquant.smooth import validate_smooth_scale_for_deployment

    with pytest.raises(ValueError, match="deployable.*proj"):
        validate_smooth_scale_for_deployment(
            torch.tensor([1e-40, 1.0]),
            dtype=torch.bfloat16,
            module_name="transformer_blocks.0.proj",
        )


def test_summarize_smooth_scale_reports_range_and_extreme_counts():
    from auto_round.algorithms.transforms.svdquant.smooth import summarize_smooth_scale

    stats = summarize_smooth_scale(torch.tensor([5e-4, 1.0, 25.0]))

    assert stats.minimum == pytest.approx(5e-4)
    assert stats.maximum == 25.0
    assert stats.ratio == pytest.approx(50_000.0)
    assert stats.below_min_count == 1
    assert stats.above_max_count == 1


def test_selected_smooth_candidate_logs_statistics_and_extreme_warning(monkeypatch):
    import auto_round.algorithms.transforms.svdquant.apply as apply_module
    from auto_round.algorithms.transforms.svdquant.smooth import SmoothCandidate

    info_messages = []
    warning_messages = []
    monkeypatch.setattr(apply_module.logger, "info", lambda message, *args: info_messages.append(message % args))
    monkeypatch.setattr(
        apply_module.logger,
        "warning",
        lambda message, *args: warning_messages.append(message % args),
    )
    candidate = SmoothCandidate(alpha=0.5, beta=0.5, scale=torch.tensor([5e-4, 1.0, 25.0]))

    apply_module.SVDQuantTransform._log_selected_smooth_candidate("blocks.0.qkv", candidate, 1.25)

    assert len(info_messages) == 1
    assert "alpha=0.5 beta=0.5 error=1.25" in info_messages[0]
    assert "scale_min=0.0005 scale_max=25" in info_messages[0]
    assert "below_1e-3=1 above_20=1" in info_messages[0]
    assert warning_messages == [
        "SVDQuant smooth scale for blocks.0.qkv contains extreme values: below_1e-3=1 above_20=1"
    ]


def test_select_best_layer_candidate_prefers_later_exact_tie():
    from auto_round.algorithms.transforms.svdquant.smooth import select_best_layer_candidate

    candidates = [("identity", 3.0), ("first", 1.0), ("later", 1.0), ("worse", 2.0)]

    assert select_best_layer_candidate(candidates, module_name="model.layers.0.proj") == "later"


def test_select_best_layer_candidate_skips_nonfinite_errors():
    from auto_round.algorithms.transforms.svdquant.smooth import select_best_layer_candidate

    candidates = [("nan", float("nan")), ("inf", float("inf")), ("valid", 2.0)]

    assert select_best_layer_candidate(candidates, module_name="model.layers.0.proj") == "valid"


def test_select_best_layer_candidate_fails_when_no_finite_error():
    from auto_round.algorithms.transforms.svdquant.smooth import select_best_layer_candidate

    with pytest.raises(ValueError, match="model.layers.0.proj"):
        select_best_layer_candidate(
            [("nan", float("nan")), ("inf", float("inf"))],
            module_name="model.layers.0.proj",
        )
