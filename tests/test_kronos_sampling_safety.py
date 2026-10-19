import torch

from model.kronos import sample_from_logits


def test_sample_from_logits_handles_nan_and_inf_values():
    logits = torch.tensor([[float("nan"), float("-inf"), 1.0, float("inf")]])

    sample = sample_from_logits(logits, temperature=1.0, top_k=1, top_p=1.0, sample_logits=True)

    assert sample.shape == (1, 1)
    assert torch.isfinite(sample.float()).all()
    assert 0 <= int(sample.item()) < logits.shape[-1]


def test_sample_from_logits_falls_back_when_all_logits_invalid():
    logits = torch.tensor([[float("nan"), float("-inf"), float("-inf")]])

    sample = sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True)

    assert sample.shape == (1, 1)
    assert 0 <= int(sample.item()) < logits.shape[-1]


def test_sample_from_logits_argmax_path_uses_torch_topk():
    logits = torch.tensor([[0.1, 0.2, 9.0]])

    sample = sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=False)

    assert int(sample.item()) == 2


def test_sample_from_logits_top_k_one_breaks_ties_deterministically():
    logits = torch.tensor([[4.0, 4.0, 1.0]])

    samples = [
        int(sample_from_logits(logits, temperature=1.0, top_k=1, top_p=1.0, sample_logits=True).item())
        for _ in range(20)
    ]

    assert samples == [0] * 20


def test_sample_from_logits_top_k_one_does_not_consume_rng():
    logits = torch.tensor([[0.1, 9.0, 0.2]])
    torch.manual_seed(123)
    before = torch.random.get_rng_state()

    sample = sample_from_logits(logits, temperature=1.0, top_k=1, top_p=1.0, sample_logits=True)
    after = torch.random.get_rng_state()

    assert int(sample.item()) == 1
    assert torch.equal(before, after)
