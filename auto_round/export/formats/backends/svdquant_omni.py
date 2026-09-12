# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit canonical SVDQuant MXFP4 format for vLLM-Omni Wan."""

from auto_round.export.formats.backends.svdquant_nunchaku import SVDQuantNunchakuFormat
from auto_round.export.formats.base import OutputFormat


@OutputFormat.register("svdquant_omni")
class SVDQuantOmniFormat(SVDQuantNunchakuFormat):
    format_name = "svdquant_omni"

    def save_quantized(
        self,
        output_dir,
        model=None,
        tokenizer=None,
        layer_config=None,
        inplace=True,
        device="cpu",
        serialization_dict=None,
        *,
        residual_provider=None,
        **kwargs
    ):
        if output_dir is None:
            return model
        from auto_round.export.svdquant_omni import save_svdquant_omni

        self._validate_svd_layer_overrides(model, layer_config)
        save_svdquant_omni(model, output_dir, residual_provider=residual_provider)
        return model
