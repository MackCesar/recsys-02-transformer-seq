from __future__ import annotations

import torch
from torch import nn, optim

from recsys_transformer_seq.config import load_config, PROJECT_ROOT
from recsys_transformer_seq.data.seq_dataset import build_dataloader
from recsys_transformer_seq.models.transformer_seq import TransformerSeqModel


def train_one_epoch(
    model: TransformerSeqModel,
    loader,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    for seqs, targets in loader:
        seqs = seqs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    cfg = load_config()
    max_len = cfg["data"]["max_sequence_length"]
    batch_size = cfg["train"]["batch_size"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    model_cfg = cfg["model"]

    seqs_csv = PROJECT_ROOT / "data" / "processed" / "movielens_sequences.csv"

    loader, num_items = build_dataloader(
        csv_path=seqs_csv,
        max_len=max_len,
        batch_size=batch_size,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = TransformerSeqModel(
        num_items=num_items,
        max_len=max_len,
        d_model=model_cfg["embedding_dim"],
        nhead=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"[INFO] Starting training on {device} for {epochs} epochs")
    for epoch in range(epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"[INFO] Epoch {epoch + 1}/{epochs}: loss={loss:.4f}")

    out_dir = PROJECT_ROOT / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "transformer_seq_movielens.pt"
    torch.save(
        {"state_dict": model.state_dict(), "num_items": num_items, "max_len": max_len},
        ckpt_path,
    )
    print(f"[INFO] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()