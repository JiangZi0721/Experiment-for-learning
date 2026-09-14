import torch
import torch.nn as nn

class AttentionEncoder(nn.Module):
    """
    Saito Ch08: Encoder that outputs all hidden states hs = [h_1, h_2, ..., h_T]
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns:
            hs: (batch_size, seq_len, hidden_size) - all time-step representations
            (h_n, c_n): final state tuple of shape (1, batch_size, hidden_size)
        """
        embedded = self.embedding(x)
        hs, (h_n, c_n) = self.lstm(embedded)
        return hs, (h_n, c_n)


class Attention(nn.Module):
    """
    Saito Ch08: Attention layer composed of AttentionWeight and WeightSum
    Computes alignment score via dot-product: score = hs . s
    """
    def __init__(self):
        super().__init__()

    def forward(self, hs, s):
        """
        hs: (batch_size, T_enc, hidden_size) - all encoder representations
        s:  (batch_size, hidden_size) - current decoder hidden state
        Returns:
            c: (batch_size, hidden_size) - dynamically weighted context vector
            a: (batch_size, T_enc) - attention distribution / alignment weights
        """
        # 1. Attention Weight (Dot product score between hs and s)
        # hs: (B, T_enc, H), s.unsqueeze(2): (B, H, 1) -> score: (B, T_enc, 1)
        score = torch.bmm(hs, s.unsqueeze(2)).squeeze(2)  # (B, T_enc)
        a = torch.softmax(score, dim=-1)                  # (B, T_enc)
        
        # 2. Weight Sum (Context Vector c)
        # a.unsqueeze(1): (B, 1, T_enc), hs: (B, T_enc, H) -> c: (B, 1, H)
        c = torch.bmm(a.unsqueeze(1), hs).squeeze(1)      # (B, H)
        return c, a


class AttentionDecoder(nn.Module):
    """
    Saito Ch08: Attention Decoder
    At each step:
      1. Embed current token e(y_{t-1})
      2. Update LSTM: s_t, c_t = LSTM(e(y_{t-1}), (s_{t-1}, c_{t-1}))
      3. Attention: c_t, a_t = Attention(hs, s_t)
      4. Affine projection: logits_t = Linear([s_t; c_t])
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.attention = Attention()
        self.fc = nn.Linear(hidden_size + hidden_size, vocab_size)

    def forward(self, dec_in, hs, h_0, c_0):
        """
        Teacher forcing forward pass.
        dec_in: (batch_size, T_dec)
        hs: (batch_size, T_enc, hidden_size)
        h_0, c_0: (1, batch_size, hidden_size)
        """
        batch_size, T_dec = dec_in.shape
        embedded = self.embedding(dec_in)  # (B, T_dec, embed_size)
        
        logits_list = []
        attns_list = []
        h_t, c_t = h_0, c_0
        
        for t in range(T_dec):
            x_t = embedded[:, t:t+1, :]  # (B, 1, embed_size)
            out_t, (h_t, c_t) = self.lstm(x_t, (h_t, c_t))  # out_t: (B, 1, H)
            s_t = out_t.squeeze(1)  # (B, H)
            
            c_ctx, a_t = self.attention(hs, s_t)  # c_ctx: (B, H), a_t: (B, T_enc)
            combined = torch.cat([s_t, c_ctx], dim=-1)  # (B, 2H)
            logits_t = self.fc(combined)  # (B, vocab_size)
            
            logits_list.append(logits_t.unsqueeze(1))
            attns_list.append(a_t.unsqueeze(1))
            
        logits = torch.cat(logits_list, dim=1)  # (B, T_dec, vocab_size)
        attns = torch.cat(attns_list, dim=1)    # (B, T_dec, T_enc)
        return logits, attns

    def step(self, cur_token, h_t, c_t, hs):
        """
        Single step forward for autoregressive generation
        """
        embedded = self.embedding(cur_token)  # (B, 1, embed_size)
        out_t, (h_next, c_next) = self.lstm(embedded, (h_t, c_t))
        s_t = out_t.squeeze(1)
        c_ctx, a_t = self.attention(hs, s_t)
        combined = torch.cat([s_t, c_ctx], dim=-1)
        logits_t = self.fc(combined)
        return logits_t, (h_next, c_next), a_t


class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        dec_in = tgt[:, :-1]
        hs, (h_n, c_n) = self.encoder(src)
        logits, attns = self.decoder(dec_in, hs, h_n, c_n)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=4, start_id=12):
        """
        Autoregressive greedy generation returning both predictions and attention weights
        """
        self.eval()
        batch_size = src.shape[0]
        hs, (h_t, c_t) = self.encoder(src)
        
        cur_token = torch.full((batch_size, 1), start_id, dtype=torch.long, device=src.device)
        preds = []
        attns_list = []
        
        for _ in range(max_len):
            logits_t, (h_t, c_t), a_t = self.decoder.step(cur_token, h_t, c_t, hs)
            pred = logits_t.argmax(dim=-1, keepdim=True)  # (B, 1)
            preds.append(pred)
            attns_list.append(a_t.unsqueeze(1))  # (B, 1, T_enc)
            cur_token = pred
            
        gen_ids = torch.cat(preds, dim=1)        # (B, max_len)
        all_attns = torch.cat(attns_list, dim=1) # (B, max_len, T_enc)
        return gen_ids, all_attns

