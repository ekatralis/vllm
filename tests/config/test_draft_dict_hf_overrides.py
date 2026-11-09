# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target dict hf_overrides reach a draft that loads the target's checkpoint."""

import functools
import pickle

from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig

YARN = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}


def _config() -> PretrainedConfig:
    text = PretrainedConfig(
        max_position_embeddings=262144,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "mrope_section": [11, 11, 10],
        },
    )
    return PretrainedConfig(
        text_config=text, model_type="x", architectures=["XForCausalLM"]
    )


def test_apply_dict_hf_overrides_merges_nested():
    overrides = {
        "text_config": {"max_position_embeddings": 1048576, "rope_parameters": YARN}
    }
    cfg = SpeculativeConfig.apply_dict_hf_overrides(overrides, _config())
    assert cfg.text_config.max_position_embeddings == 1048576
    rp = cfg.text_config.rope_parameters
    assert rp["rope_type"] == "yarn" and rp["factor"] == 4.0
    # Keys the override does not mention are kept.
    assert rp["rope_theta"] == 1e7 and rp["mrope_section"] == [11, 11, 10]


def test_composed_draft_override_is_picklable_and_applies():
    overrides = {"text_config": {"rope_parameters": YARN}}
    fn = SpeculativeConfig.compose_draft_hf_overrides(
        functools.partial(SpeculativeConfig.apply_dict_hf_overrides, overrides)
    )
    fn = pickle.loads(pickle.dumps(fn))  # sent to engine-core processes
    cfg = fn(_config())
    assert cfg.text_config.rope_parameters["rope_type"] == "yarn"
    assert cfg.text_config.rope_parameters["factor"] == 4.0
