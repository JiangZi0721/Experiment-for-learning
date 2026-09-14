import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns:
            h_n: (1, batch_size, hidden_size)
            c_n: (1, batch_size, hidden_size)
        """
        embedded = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embedded)
        return h_n, c_n


class StandardDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_0, c_0):
        """
        x: (batch_size, seq_len)
        h_0, c_0: (1, batch_size, hidden_size)
        """
        embedded = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))
        logits = self.fc(out)  # (batch_size, seq_len, vocab_size)
        return logits, (h_n, c_n)

    def step(self, x_t, h_t, c_t, h_enc=None):
        """
        Single step forward for autoregressive generation
        x_t: (batch_size, 1)
        """
        embedded = self.embedding(x_t)
        out, (h_next, c_next) = self.lstm(embedded, (h_t, c_t))
        logits = self.fc(out)  # (batch_size, 1, vocab_size)
        return logits, (h_next, c_next)


class PeekyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Input to LSTM: embed_size + hidden_size (concatenated context vector h)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        # Input to FC: hidden_size + hidden_size (concatenated context vector h)
        self.fc = nn.Linear(hidden_size + hidden_size, vocab_size)

    def forward(self, x, h_0, c_0):
        """
        x: (batch_size, seq_len)
        h_0, c_0: (1, batch_size, hidden_size)
        """
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)
        
        # Context vector from encoder's final hidden state
        h_enc = h_0[0]  # (batch_size, hidden_size)
        h_expanded = h_enc.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_size)
        
        # 1. Concatenate h with embedding before LSTM
        lstm_in = torch.cat([embedded, h_expanded], dim=-1)  # (batch_size, seq_len, embed_size + hidden_size)
        out, (h_n, c_n) = self.lstm(lstm_in, (h_0, c_0))  # (batch_size, seq_len, hidden_size)
        
        # 2. Concatenate h with LSTM output before Linear layer
        fc_in = torch.cat([out, h_expanded], dim=-1)  # (batch_size, seq_len, 2 * hidden_size)
        logits = self.fc(fc_in)  # (batch_size, seq_len, vocab_size)
        
        return logits, (h_n, c_n)

    def step(self, x_t, h_t, c_t, h_enc):
        """
        Single step forward for autoregressive generation
        x_t: (batch_size, 1)
        h_enc: (batch_size, hidden_size)
        """
        embedded = self.embedding(x_t)  # (batch_size, 1, embed_size)
        h_expanded = h_enc.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        lstm_in = torch.cat([embedded, h_expanded], dim=-1)
        out, (h_next, c_next) = self.lstm(lstm_in, (h_t, c_t))
        
        fc_in = torch.cat([out, h_expanded], dim=-1)
        logits = self.fc(fc_in)
        return logits, (h_next, c_next)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, is_peeky=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_peeky = is_peeky

    def forward(self, src, tgt):
        """
        src: (batch_size, src_len)
        tgt: (batch_size, tgt_len) where tgt[:, 0] is start token '_'
        Teacher forcing forward pass.
        Returns: logits of shape (batch_size, tgt_len - 1, vocab_size)
        """
        dec_in = tgt[:, :-1]  # Exclude last token for teacher forcing
        h_n, c_n = self.encoder(src)
        logits, _ = self.decoder(dec_in, h_n, c_n)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=4, start_id=12):
        """
        Autoregressive greedy generation.
        src: (batch_size, src_len)
        max_len: number of tokens to generate
        start_id: ID of start token '_'
        Returns: tensor of shape (batch_size, max_len) with generated IDs
        """
        self.eval()
        batch_size = src.shape[0]
        h_t, c_t = self.encoder(src)
        h_enc = h_t[0]  # (batch_size, hidden_size)
        
        cur_token = torch.full((batch_size, 1), start_id, dtype=torch.long, device=src.device)
        preds = []
        
        for _ in range(max_len):
            if self.is_peeky:
                logits, (h_t, c_t) = self.decoder.step(cur_token, h_t, c_t, h_enc)
            else:
                logits, (h_t, c_t) = self.decoder.step(cur_token, h_t, c_t)
            
            # Greedy choice
            pred = logits.argmax(dim=-1)  # (batch_size, 1)
            preds.append(pred)
            cur_token = pred
            
        return torch.cat(preds, dim=1)  # (batch_size, max_len)
