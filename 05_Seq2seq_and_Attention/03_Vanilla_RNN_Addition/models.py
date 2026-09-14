import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True, nonlinearity='tanh')

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns: h_n: (1, batch_size, hidden_size)
        """
        embedded = self.embedding(x)
        out, h_n = self.rnn(embedded)
        return h_n


class RNNStandardDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_0):
        embedded = self.embedding(x)
        out, h_n = self.rnn(embedded, h_0)
        logits = self.fc(out)
        return logits, h_n

    def step(self, x_t, h_t, h_enc=None):
        embedded = self.embedding(x_t)
        out, h_next = self.rnn(embedded, h_t)
        logits = self.fc(out)
        return logits, h_next


class RNNPeekyDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size + hidden_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size + hidden_size, vocab_size)

    def forward(self, x, h_0):
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        
        h_enc = h_0[0]  # (batch_size, hidden_size)
        h_expanded = h_enc.unsqueeze(1).expand(-1, seq_len, -1)
        
        rnn_in = torch.cat([embedded, h_expanded], dim=-1)
        out, h_n = self.rnn(rnn_in, h_0)
        
        fc_in = torch.cat([out, h_expanded], dim=-1)
        logits = self.fc(fc_in)
        return logits, h_n

    def step(self, x_t, h_t, h_enc):
        embedded = self.embedding(x_t)
        h_expanded = h_enc.unsqueeze(1)
        
        rnn_in = torch.cat([embedded, h_expanded], dim=-1)
        out, h_next = self.rnn(rnn_in, h_t)
        
        fc_in = torch.cat([out, h_expanded], dim=-1)
        logits = self.fc(fc_in)
        return logits, h_next


class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, is_peeky=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_peeky = is_peeky

    def forward(self, src, tgt):
        dec_in = tgt[:, :-1]
        h_n = self.encoder(src)
        logits, _ = self.decoder(dec_in, h_n)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=4, start_id=12):
        self.eval()
        batch_size = src.shape[0]
        h_t = self.encoder(src)
        h_enc = h_t[0]
        
        cur_token = torch.full((batch_size, 1), start_id, dtype=torch.long, device=src.device)
        preds = []
        
        for _ in range(max_len):
            if self.is_peeky:
                logits, h_t = self.decoder.step(cur_token, h_t, h_enc)
            else:
                logits, h_t = self.decoder.step(cur_token, h_t)
                
            pred = logits.argmax(dim=-1)
            preds.append(pred)
            cur_token = pred
            
        return torch.cat(preds, dim=1)
