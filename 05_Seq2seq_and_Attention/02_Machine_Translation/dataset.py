import re
import random
import unicodedata
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip().split()

class Vocab:
    def __init__(self, name, min_freq=1):
        self.name = name
        self.min_freq = min_freq
        self.word2id = {PAD_TOKEN: PAD_ID, SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID, UNK_TOKEN: UNK_ID}
        self.id2word = {PAD_ID: PAD_TOKEN, SOS_ID: SOS_TOKEN, EOS_ID: EOS_TOKEN, UNK_ID: UNK_TOKEN}
        self.counts = Counter()

    def build_vocab(self, tokenized_sentences):
        for s in tokenized_sentences:
            self.counts.update(s)
        for w, c in self.counts.items():
            if c >= self.min_freq and w not in self.word2id:
                new_id = len(self.word2id)
                self.word2id[w] = new_id
                self.id2word[new_id] = w

    def __len__(self):
        return len(self.word2id)

    def encode(self, tokens):
        return [self.word2id.get(w, UNK_ID) for w in tokens]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == EOS_ID:
                break
            if i not in (PAD_ID, SOS_ID):
                words.append(self.id2word.get(i, UNK_TOKEN))
        return words


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, reverse=False):
        """
        pairs: list of (src_tokens, tgt_tokens)
        reverse: whether to reverse source sentence tokens
        """
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.reverse = reverse

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]
        
        if self.reverse:
            src_tokens = src_tokens[::-1]
            
        src_ids = self.src_vocab.encode(src_tokens)
        tgt_ids = self.tgt_vocab.encode(tgt_tokens)
        
        # decoder input: [SOS, y1, y2, ...]
        dec_in = [SOS_ID] + tgt_ids
        # decoder target: [y1, y2, ..., EOS]
        dec_tgt = tgt_ids + [EOS_ID]
        
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(dec_in, dtype=torch.long), \
               torch.tensor(dec_tgt, dtype=torch.long), \
               src_tokens, tgt_tokens


def collate_fn(batch):
    src_list, dec_in_list, dec_tgt_list, raw_src, raw_tgt = zip(*batch)
    
    src_lens = [len(s) for s in src_list]
    tgt_lens = [len(t) for t in dec_in_list]
    
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    
    padded_src = torch.full((len(batch), max_src_len), PAD_ID, dtype=torch.long)
    padded_dec_in = torch.full((len(batch), max_tgt_len), PAD_ID, dtype=torch.long)
    padded_dec_tgt = torch.full((len(batch), max_tgt_len), PAD_ID, dtype=torch.long)
    
    for i in range(len(batch)):
        padded_src[i, :len(src_list[i])] = src_list[i]
        padded_dec_in[i, :len(dec_in_list[i])] = dec_in_list[i]
        padded_dec_tgt[i, :len(dec_tgt_list[i])] = dec_tgt_list[i]
        
    return padded_src, padded_dec_in, padded_dec_tgt, list(raw_src), list(raw_tgt)


def get_translation_data(file_path=None, num_samples=15000, train_ratio=0.9, seed=42, min_len=2, max_len=9):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'eng-fra.txt')
    random.seed(seed)
    pairs = []

    
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                src = normalize_string(parts[0])
                tgt = normalize_string(parts[1])
                if min_len <= len(src) <= max_len and min_len <= len(tgt) <= max_len:
                    pairs.append((src, tgt))
                    
    # Randomly shuffle and take num_samples
    random.shuffle(pairs)
    pairs = pairs[:num_samples]
    
    # Train / Test split
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    # Build vocabs only on train pairs
    src_vocab = Vocab('eng', min_freq=2)
    tgt_vocab = Vocab('fra', min_freq=2)
    
    src_vocab.build_vocab([p[0] for p in train_pairs])
    tgt_vocab.build_vocab([p[1] for p in train_pairs])
    
    return train_pairs, test_pairs, src_vocab, tgt_vocab


def get_translation_loaders(train_pairs, test_pairs, src_vocab, tgt_vocab, batch_size=128, reverse=False):
    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab, reverse=reverse)
    test_ds = TranslationDataset(test_pairs, src_vocab, tgt_vocab, reverse=reverse)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader
