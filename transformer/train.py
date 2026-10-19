import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                    DIM_FEEDFORWARD, DROPOUT, BATCH_SIZE, EPOCHS, LR, CLIP)
from dataset import RAW_PAIRS, Vocab, TranslationDataset, collate_fn, PAD, EOS
from model import Transformer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    random.seed(42)
    torch.manual_seed(42)

    # 1) 构建词表
    src_vocab = Vocab([zh for zh, _ in RAW_PAIRS])
    tgt_vocab = Vocab([en for _, en in RAW_PAIRS])
    print(f"中文词表大小: {len(src_vocab)}, 英文词表大小: {len(tgt_vocab)}")

    # 2) 数据
    dataset = TranslationDataset(RAW_PAIRS, src_vocab, tgt_vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 3) 模型 / 优化器 / 损失
    model = Transformer(
        len(src_vocab), len(tgt_vocab),
        d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, pad_idx=PAD,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # 4) 训练
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 25 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:3d} | loss = {avg_loss:.4f}")
            # 每 25 轮抽查一个句子
            sample = "我 有 一个 苹果"
            src_ids = torch.tensor([src_vocab.encode(sample)], dtype=torch.long, device=device)
            pred_ids = model.greedy_decode(src_ids)
            print(f"    示例: {sample}  ->  {tgt_vocab.decode(pred_ids)}")

    # 5) 保存
    torch.save({
        "model": model.state_dict(),
        "src_word2idx": src_vocab.word2idx,
        "tgt_word2idx": tgt_vocab.word2idx,
    }, "checkpoint.pt")
    print("模型已保存到 checkpoint.pt")

    # 6) 最终逐句测试
    print("\n===== 最终测试 =====")
    for zh, en in RAW_PAIRS:
        src_ids = torch.tensor([src_vocab.encode(zh)], dtype=torch.long, device=device)
        pred_ids = model.greedy_decode(src_ids)
        pred = tgt_vocab.decode(pred_ids)
        print(f"{zh:12s} -> {pred:20s} (真值: {en})")


if __name__ == "__main__":
    main()