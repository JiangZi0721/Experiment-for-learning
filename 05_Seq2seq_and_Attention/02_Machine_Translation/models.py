import torch
import torch.nn as nn

class TranslationEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, hidden_size, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns: h_n, c_n of shape (1, batch_size, hidden_size)
        """
        embedded = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embedded)
        return h_n, c_n


class TranslationStandardDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, hidden_size, pad_id=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, dec_in, h_0, c_0):
        embedded = self.embedding(dec_in)
        out, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))
        logits = self.fc(out)
        return logits, (h_n, c_n)

    def step(self, x_t, h_t, c_t, h_enc=None):
        embedded = self.embedding(x_t)
        out, (h_next, c_next) = self.lstm(embedded, (h_t, c_t))
        logits = self.fc(out)
        return logits, (h_next, c_next)


class TranslationPeekyDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, hidden_size, pad_id=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + hidden_size, tgt_vocab_size)

    def forward(self, dec_in, h_0, c_0):
        batch_size, seq_len = dec_in.shape
        embedded = self.embedding(dec_in)
        
        h_enc = h_0[0]  # (B, H)
        h_expanded = h_enc.unsqueeze(1).expand(-1, seq_len, -1)
        
        lstm_in = torch.cat([embedded, h_expanded], dim=-1)
        out, (h_n, c_n) = self.lstm(lstm_in, (h_0, c_0))
        
        fc_in = torch.cat([out, h_expanded], dim=-1)
        logits = self.fc(fc_in)
        return logits, (h_n, c_n)

    def step(self, x_t, h_t, c_t, h_enc):
        embedded = self.embedding(x_t)
        h_expanded = h_enc.unsqueeze(1)
        
        lstm_in = torch.cat([embedded, h_expanded], dim=-1)
        out, (h_next, c_next) = self.lstm(lstm_in, (h_t, c_t))
        
        fc_in = torch.cat([out, h_expanded], dim=-1)
        logits = self.fc(fc_in)
        return logits, (h_next, c_next)


class TranslationSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, is_peeky=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_peeky = is_peeky

    def forward(self, src, dec_in):
        h_n, c_n = self.encoder(src)
        logits, _ = self.decoder(dec_in, h_n, c_n)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=15, sos_id=1, eos_id=2):
        self.eval()
        batch_size = src.shape[0]
        h_t, c_t = self.encoder(src)
        h_enc = h_t[0]
        
        cur_token = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src.device)
        preds = []
        
        for _ in range(max_len):
            if self.is_peeky:
                logits, (h_t, c_t) = self.decoder.step(cur_token, h_t, c_t, h_enc)
            else:
                logits, (h_t, c_t) = self.decoder.step(cur_token, h_t, c_t)
                
            pred = logits.argmax(dim=-1)  # (B, 1)
            preds.append(pred)
            cur_token = pred
            
        return torch.cat(preds, dim=1)
