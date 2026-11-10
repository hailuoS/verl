"""Exercise the entropy autograd implementation without importing Megatron/CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def entropy_functions():
    source = Path(__file__).resolve().parents[2] / "verl/utils/megatron/tensor_parallel.py"
    names = {"_VocabParallelEntropy", "vocab_parallel_entropy", "vocab_parallel_entropy_with_chunking"}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    # Single-rank collectives are identities. The separate two-NPU test exercises TP reductions.
    torch_api = SimpleNamespace(**{name: getattr(torch, name) for name in dir(torch)})
    torch_api.compile = lambda **kwargs: lambda fn: fn
    scope = {
        "torch": torch_api,
        "dist": SimpleNamespace(all_reduce=lambda *args, **kwargs: None, ReduceOp=SimpleNamespace(MAX=None)),
        "mpu": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
    }
    exec(compile(tree, str(source), "exec"), scope)
    return scope


@pytest.mark.parametrize("chunk_size", [1, 3, 32])
def test_chunked_entropy_gradient_and_saved_inputs(entropy_functions, chunk_size):
    torch.manual_seed(17)
    logits = torch.randn(2, 7, 19, requires_grad=True)
    original = logits.detach().clone()
    reference_logits = original.clone().requires_grad_()
    expected = -(reference_logits.softmax(-1) * reference_logits.log_softmax(-1)).sum(-1)
    actual = entropy_functions["vocab_parallel_entropy_with_chunking"](logits, chunk_size)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream, retain_graph=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    torch.testing.assert_close(logits, original, atol=0, rtol=0)
    # Saved probabilities must remain usable for another backward over the same graph.
    logits.grad = None
    actual.backward(upstream)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
