import torch

from config import (D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                    DIM_FEEDFORWARD, DROPOUT)
from dataset import Vocab, PAD
from model import Transformer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("checkpoint.pt", map_location=device)

    src_vocab = Vocab(word2idx=ckpt["src_word2idx"])
    tgt_vocab = Vocab(word2idx=ckpt["tgt_word2idx"])

    model = Transformer(
        len(src_vocab), len(tgt_vocab),
        d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, pad_idx=PAD,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("输入中文（用空格分词，例如：我 有 一个 苹果），输入 q 退出。")
    while True:
        text = input("\n中文> ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue
        src_ids = torch.tensor([src_vocab.encode(text)], dtype=torch.long, device=device)
        pred_ids = model.greedy_decode(src_ids)
        print("英文>", tgt_vocab.decode(pred_ids))


if __name__ == "__main__":
    main()