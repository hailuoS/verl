import torch

from verl.models.transformers.dense_common import torch_fused_lm_head


def test_fused_head_uses_module_hooks_and_preserves_linear_forward():
    torch.manual_seed(123)
    head = torch.nn.Linear(8, 19, bias=False)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.randint(0, 19, (2, 5))
    calls = []
    head.register_forward_pre_hook(lambda *args: calls.append("pre"))
    head.register_forward_hook(lambda *args: calls.append("post"))
    log_probs, entropy = torch_fused_lm_head(head, hidden, labels, 0.7)
    assert calls == ["pre", "post"]
    logits = head._verl_linear_forward(hidden) / 0.7
    expected = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(log_probs, expected)
    (log_probs.sum() + entropy.sum()).backward()
    assert hidden.grad is not None and head.weight.grad is not None
    torch.testing.assert_close(head(hidden.detach()), hidden.detach() @ head.weight.t())
    assert calls == ["pre", "post", "pre", "post"]
    assert list(head.state_dict()) == ["weight"]
