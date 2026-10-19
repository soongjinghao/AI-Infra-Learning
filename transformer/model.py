import math
import torch
import torch.nn as nn

from dataset import PAD, BOS, EOS


class PositionalEncoding(nn.Module):
    """经典正弦位置编码，形状 (1, max_len, d_model)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def make_padding_mask(seq, pad_idx=PAD):
    """True 表示该位置是 padding，需要被 mask 掉"""
    return seq == pad_idx


def make_causal_mask(sz, device=None):
    """True 表示不可见；上三角为 True，防止看到未来 token"""
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=256, dropout=0.1, pad_idx=PAD):
        super().__init__()
        self.pad_idx = pad_idx

        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        # 用 PyTorch 内置的 Encoder/Decoder Layer，保证 mask 逻辑正确、代码简洁。
        # 核心的：Embedding -> 位置编码 -> 多头自注意力 -> 交叉注意力 -> FFN -> 残差+LN
        # 都在这些 Layer 内部，训练循环和 mask 构造是手写的。
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt_in):
        """训练前向：src (B,S), tgt_in (B,T) -> logits (B,T,V)"""
        src_pad_mask = make_padding_mask(src, self.pad_idx)
        tgt_pad_mask = make_padding_mask(tgt_in, self.pad_idx)
        tgt_causal_mask = make_causal_mask(tgt_in.size(1), device=tgt_in.device)

        src_emb = self.dropout(self.pos_enc(self.src_emb(src)))
        tgt_emb = self.dropout(self.pos_enc(self.tgt_emb(tgt_in)))

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(out)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=32):
        """贪心解码：一次一个 token，遇到 EOS 停止。返回不含 BOS 的 id 列表。"""
        self.eval()
        device = src.device
        src_pad_mask = make_padding_mask(src, self.pad_idx)
        memory = self.encoder(
            self.pos_enc(self.src_emb(src)),
            src_key_padding_mask=src_pad_mask,
        )

        ys = torch.full((1, 1), BOS, dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_mask = make_causal_mask(ys.size(1), device=device)
            out = self.decoder(
                self.pos_enc(self.tgt_emb(ys)), memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            next_id = self.fc_out(out[:, -1, :]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == EOS:
                break
        return ys[0, 1:].tolist()