import logging

from auto_round.utils.model import is_diffusion_model


def test_diffusers_scheduler_directory_does_not_emit_transformers_config_warning(tmp_path, caplog):
    (tmp_path / "scheduler_config.json").write_text('{"_class_name": "FlowMatchEulerDiscreteScheduler"}')

    with caplog.at_level(logging.WARNING, logger="autoround"):
        assert is_diffusion_model(str(tmp_path)) is False

    assert not any("Failed to load config" in record.message for record in caplog.records)


def test_local_diffusion_pipeline_is_detected_from_model_index_without_warning(tmp_path, caplog):
    (tmp_path / "model_index.json").write_text('{"_class_name": "FluxPipeline"}')

    with caplog.at_level(logging.WARNING, logger="autoround"):
        assert is_diffusion_model(str(tmp_path)) is True

    assert not any("Failed to load config" in record.message for record in caplog.records)
