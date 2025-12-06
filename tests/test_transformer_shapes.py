import torch

from recsys_transformer_seq.models.transformer_seq import TransformerSeqModel


def test_transformer_forward_shapes():
    num_items = 100
    max_len = 20
    model = TransformerSeqModel(
        num_items=num_items,
        max_len=max_len,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.1,
    )

    batch_size = 4
    seqs = torch.randint(low=1, high=num_items, size=(batch_size, max_len))
    logits = model(seqs)
    assert logits.shape == (batch_size, num_items)