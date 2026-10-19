import torch
from torch.utils.data import Dataset

# 特殊 token 的 id（约定固定）
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# 玩具语料：中文已经用空格分好词，英文本身以空格分词
RAW_PAIRS = [
    ("我 有 一个 苹果", "i have an apple"),
    ("我 有 一个 香蕉", "i have a banana"),
    ("我 有 一只 猫", "i have a cat"),
    ("我 有 一只 狗", "i have a dog"),
    ("我 喜欢 苹果", "i like apples"),
    ("我 喜欢 香蕉", "i like bananas"),
    ("我 喜欢 猫", "i like cats"),
    ("我 喜欢 狗", "i like dogs"),
    ("他 有 一个 苹果", "he has an apple"),
    ("他 有 一只 猫", "he has a cat"),
    ("他 喜欢 苹果", "he likes apples"),
    ("他 喜欢 猫", "he likes cats"),
    ("她 有 一个 香蕉", "she has a banana"),
    ("她 有 一只 狗", "she has a dog"),
    ("她 喜欢 香蕉", "she likes bananas"),
    ("她 喜欢 狗", "she likes dogs"),
]


def build_vocab(sentences):
    """从句子列表构建词表：特殊 token 固定在前，其余按出现顺序编号。"""
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for s in sentences:
        for tok in s.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


class Vocab:
    def __init__(self, sentences=None, word2idx=None):
        if word2idx is not None:
            self.word2idx = word2idx
        else:
            self.word2idx = build_vocab(sentences)
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, sentence):
        """句子 -> id 列表（OOV 用 UNK）"""
        return [self.word2idx.get(tok, UNK) for tok in sentence.split()]

    def decode(self, ids, skip_special=True):
        """id 列表 -> 句子字符串"""
        special = {PAD, BOS, EOS} if skip_special else set()
        return " ".join(self.idx2word.get(i, "<unk>") for i in ids if i not in special)

    def __len__(self):
        return len(self.word2idx)


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = []
        for zh, en in pairs:
            self.data.append((src_vocab.encode(zh), tgt_vocab.encode(en)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_idx=PAD):
    """
    返回：
      src:      (batch, src_len)      源语言输入
      tgt_in:   (batch, tgt_len)      Decoder 输入 = [BOS] + tgt
      tgt_out:  (batch, tgt_len)      训练标签   = tgt + [EOS]
    """
    src_list, tgt_in_list, tgt_out_list = [], [], []
    for src, tgt in batch:
        src_list.append(src)
        tgt_in_list.append([BOS] + tgt)
        tgt_out_list.append(tgt + [EOS])

    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        padded = [s + [pad_idx] * (max_len - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long)

    return pad(src_list), pad(tgt_in_list), pad(tgt_out_list)