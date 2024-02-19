import json
import sys
from pathlib import Path

import pytest
import torch

from conftest import RunIf
from unittest import mock

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def test_sliding_window_parameter_in_config():
    from lit_gpt import Config

    config = Config()
    assert config.name == ""
    # Check sliding window parameters are default
    assert config.use_sliding_window is False
    assert config.sliding_window_size is None

    config = Config(use_sliding_window=True)
    assert config.use_sliding_window is True
    assert config.sliding_window_size == config.n_embd

    # The following config will generate a warning.
    config = Config(sliding_window_size=1024)
    assert config.use_sliding_window is False
    assert config.sliding_window_size == 1024

    config = Config(
        use_sliding_window=True,
        sliding_window_size=1024
    )
    assert config.use_sliding_window is True
    assert config.sliding_window_size == 1024


@pytest.mark.parametrize(
    ("max_seq_length", "window_size", "cache_opt", "expected_output"),
    [
        (5, None, False,
         torch.Tensor([[[[True, False, False, False, False],
                         [True, True, False, False, False],
                         [True, True, True, False, False],
                         [True, True, True, True, False],
                         [True, True, True, True, True]]]])
         ),  # base case
        (5, 3, False,
         torch.Tensor([[[[True, False, False, False, False],
                         [True, True, False, False, False],
                         [True, True, True, False, False],
                         [False, True, True, True, False],
                         [False, False, True, True, True]]]])
         ),  # use of sliding window
        (12, 3, True,
         torch.Tensor([[[[True, False, False, False, False, False],
                         [True, True, False, False, False, False],
                         [True, True, True, False, False, False],
                         [False, True, True, True, False, False],
                         [False, False, True, True, True, False],
                         [False, False, False, True, True, True],
                         [False, True, True, True, False, False],
                         [False, False, True, True, True, False],
                         [False, False, False, True, True, True],
                         [False, True, True, True, False, False],
                         [False, False, True, True, True, False],
                         [False, False, False, True, True, True]]]])
         )  # use of sliding window
    ],
)
def test_causal_mask_generation(
        max_seq_length, window_size, cache_opt, expected_output
):
    from lit_gpt.model import build_mask_cache

    mask = build_mask_cache(
        max_seq_length=max_seq_length,
        window_size=window_size,
        optimise_cache_size=cache_opt
    )
    torch.testing.assert_close(mask, expected_output, check_dtype=False)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("block_size", "use_sliding_window", "window_size", "opt_cache"),
   [
       (32, False, None, None),  # basic check that the changes did not break the original behaviour.
       (32, True, 32, None),  # window size equal to block and embedding size (no window size)
       # (64, True, 32, None),  # block size of size n_embed*n_layer with window_size=n_embed (default mistral)
       # (32, True, 16, None),  # window size < n_embed (extra case)
       # (64, True, 32, True),  # introducing optimised cache (here with same length as original one)
       # (64, True, 16, True),  # try introducing optimised cache with size smaller than block size
       # (256, True, 16, True),  # as above with bigger block size
       # (256, True, 4, True),  # as above with smaller window size

        # Note that the commented-out tests have been temporarily disabled since HF version of Mistral with SPDA
        # does not support SWA (that is the default for CPU).
        # todo: find a way to enable the two tests above conditionally on cpu/gpu.
   ]
)
def test_against_hf_mistral(device, dtype, block_size, use_sliding_window, window_size, opt_cache):
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM

    from lit_gpt import GPT, Config
    from scripts.convert_hf_checkpoint import copy_weights_hf_llama

    torch.set_default_dtype(dtype)

    ours_config = Config.from_name(
        "Mistral-7B-Instruct-v0.1",
        padded_vocab_size=10000,
        block_size=block_size,
        n_layer=2,
        n_embd=32,
        n_head=8,
        n_query_groups=2,
        intermediate_size=86,
        use_sliding_window=use_sliding_window,
        sliding_window_size=window_size,
        optimise_cache_for_sliding_window = opt_cache
    )

    theirs_config = MistralConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=block_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        sliding_window=ours_config.sliding_window_size
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size
    assert ours_config.sliding_window_size == theirs_config.sliding_window

    theirs_model = MistralForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    if block_size == 32:
        T = 24
        x = torch.tensor(
            [[9856, 23, 491, 1536, 32, 43, 1982, 86,
                178, 12, 46, 89, 1293, 43, 41, 72,
                2, 23, 64, 12, 21, 9, 6, 304]],
            dtype=torch.int32, device=device
        )
    else:  # block_size == 64
        T = 48
        x = torch.tensor(
            [[9856, 23, 491, 1536, 32, 43, 1982, 86,
              178, 12, 46, 89, 1293, 43, 41, 72,
              123, 43, 12, 46, 89, 8, 56, 12,
              178, 12, 9, 10, 11, 43, 41, 72,
              5, 98, 123, 4, 76, 87, 88, 89,
              2, 23, 64, 12, 21, 9, 6, 304]],
            dtype=torch.int32, device=device
        )
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@pytest.mark.parametrize(
   ("block_size", "use_sliding_window", "window_size", "opt_cache"),
   [
       (32, False, None, None),  # basic check that the changes did not break the original behaviour.
       (32, True, 32, None),  # window size equal to block and embedding size (no window size)
       (64, True, 32, None),  # block size of size n_embed*n_layer with window_size=n_embed (default mistral)
       (32, True, 16, None),  # window size < n_embed (extra case)
       (64, True, 32, True),  # introducing optimised cache (here with same length as original one)
       (64, True, 16, True),  # try introducing optimised cache with size smaller than block size
       (256, True, 16, True),  # as above with bigger block size
       (256, True, 4, True),  # as above with smaller window size
   ]
)
def test_generate(block_size, use_sliding_window, window_size, opt_cache):
    import generate.base as generate
    from lit_gpt import GPT, Config

    T = 5
    input_idx = torch.randint(10, size=(T,))

    config = Config(
        block_size=block_size,
        vocab_size=16,
        n_layer=1,
        n_head=4,
        n_embd=8,
        use_sliding_window=use_sliding_window,
        sliding_window_size=window_size,
        optimise_cache_for_sliding_window=opt_cache
    )
    model = GPT(config)
    model.set_kv_cache(batch_size=1)
    max_new_tokens = 25

    multinomial_results = []

    def multinomial(*args, **kwargs):
        out = torch.multinomial(*args, **kwargs, num_samples=1)
        multinomial_results.append(out)
        return out

    with mock.patch("generate.base.multinomial_num_samples_1", multinomial):
        out = generate.generate(model, input_idx, T + max_new_tokens, top_k=4)

    assert out.size(0) == T + max_new_tokens
    multinomial_results = torch.hstack(multinomial_results)
    expected = torch.cat((input_idx, multinomial_results))
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)
