import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import math
import torch.nn.functional as F


class EmbModel(nn.Module):
    def __init__(self, num_classes, emb_dim=None, lstm_dim=None, lstm_layers=1,
                 dropout=0.0, vocab_size=16000):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=3)

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=lstm_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.linear = nn.Linear(2 * lstm_dim, num_classes, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.leaky_relu = nn.LeakyReLU()

    def run(self, x, x_len, stock_pos):
        embedded = self.embedding(x)
        x_len_cpu = x_len.to("cpu").long()
        packed = pack_padded_sequence(embedded, x_len_cpu, 
                                    batch_first=True,
                                    enforce_sorted=False)
        output, _ = self.lstm(packed)
        output_unpacked, _ = pad_packed_sequence(output, batch_first=True)

        stock_pos = stock_pos[:, :output_unpacked.size(1)].unsqueeze(2)
        mask = torch.zeros_like(stock_pos)
        weight = 1 / stock_pos.float().sum(1, keepdim=True)
        mask = torch.where(stock_pos, weight, mask.float())
        return (output_unpacked * mask).sum(1)

    def forward(self, x, x_len, stock_pos, classify=False):
        h = self.run(x, x_len, stock_pos)
        if classify:
            return self.linear(h)
        return h

    def stock_emb(self):
        return self.linear.weight


class LearnableTwoStreamContext(nn.Module):
    def __init__(self, d, use_cosine=True, init_method="bert"):
        super().__init__()
        self.use_cosine = use_cosine
        self.q_news  = nn.Parameter(torch.empty(d))
        self.q_tweet = nn.Parameter(torch.empty(d))
        self.reset_parameters(init_method)

    def reset_parameters(self, init="bert"):
        if init == "bert":
            nn.init.normal_(self.q_news,  mean=0.0, std=0.02)
            nn.init.normal_(self.q_tweet, mean=0.0, std=0.02)
        elif init == "xavier_uniform":
            fan_in = self.q_news.numel()
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.q_news,  -bound, bound)
            nn.init.uniform_(self.q_tweet, -bound, bound)
        elif init == "unit_norm":
            nn.init.normal_(self.q_news,  0.0, 1.0)
            nn.init.normal_(self.q_tweet, 0.0, 1.0)
            with torch.no_grad():
                self.q_news.data  /= self.q_news.data.norm(p=2).clamp_min(1e-12)
                self.q_tweet.data /= self.q_tweet.data.norm(p=2).clamp_min(1e-12)
        else:
            raise ValueError(f"Unknown init: {init}")

    def _attention_pool(self, query, keys, values, mask=None):
        B, N, d = keys.shape
        scores = (keys @ query) / math.sqrt(d)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, N)
        return (weights.unsqueeze(-1) * values).sum(dim=1)  # (B, d_v)

    def forward(self, news_embs, tweet_embs, news_mask=None, tweet_mask=None):
        B, _, d = news_embs.shape
        assert tweet_embs.size(0) == B and tweet_embs.size(2) == d

        s_news  = self._attention_pool(self.q_news,  news_embs,  news_embs,  news_mask)   # (B, d)
        s_tweet = self._attention_pool(self.q_tweet, tweet_embs, tweet_embs, tweet_mask)  # (B, d)

        c_news_n  = F.normalize(s_news,  dim=-1)
        c_tweet_n = F.normalize(s_tweet, dim=-1)
        cos = (c_news_n * c_tweet_n).sum(-1, keepdim=True)  # (B, 1)
        divergence = 1.0 - cos                               # (B, 1)
        return s_news, s_tweet, cos, divergence              # (B,d), (B,d), (B,1), (B,1)

# ---------- NEW: Agreement-aware MLP gate ----------
class AgreementGate(nn.Module):
    """
    Learnable gate g in [0,1] using MLP over [c_news; c_tweet; cos].
    If scalar=False -> feature-wise gate (B, d); else scalar (B, 1).
    """
    def __init__(self, d, hidden=None, scalar=False, use_cos=True):
        super().__init__()
        self.scalar = scalar
        self.use_cos = use_cos
        hidden = hidden or max(64, d // 2)
        in_dim = 2 * d + (1 if use_cos else 0)
        out_dim = 1 if scalar else d
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, c_news, c_tweet, cos=None):
        xs = [c_news, c_tweet]
        if self.use_cos:
            assert cos is not None, "cos required when use_cos=True"
            xs.append(cos)  # (B,1)
        x = torch.cat(xs, dim=-1)  # (B, 2d + [1])
        g = self.mlp(x)            # (B, 1) or (B, d)
        # print("gate value", g)
        mix = g * c_news + (1.0 - g) * c_tweet
        return mix, g  # return gate too for analysis



class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, query, keys, values):
        # Shapes:
        #   `query`: (B, M)
        #   `keys`: (N, M)
        #   `values`: (N, X)
        #    return: (B, X)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        return self.softmax(query.matmul(keys.t())).matmul(values)

class MainModel(nn.Module):
    """
    gate_usage: 'pre' | 'post' | 'both' | 'none'
      - 'pre'  : concat projected mixed context (and optionally divergence) to every timestep before LSTM
      - 'post' : concat mixed context (and optionally divergence) to state after LSTM
      - 'both' : do both
    gate_scalar: if True, gate is scalar; else feature-wise gate
    add_div: use divergence (1 - cos) as an explicit scalar feature in pre/post, depending on gate_usage
    """
    def __init__(self,
                 num_features,
                 hidden_dim,
                 d_text,
                 lstm_layers=1,
                 dropout=0.0,
                 gate_usage='post',
                 gate_scalar=False,
                 add_div=True,
                 gate_hidden=None, 
                 div_proj_dim: int = 16 ):
        super().__init__()
        assert gate_usage in ('pre', 'post', 'both', 'none')
        self.gate_usage = gate_usage
        self.add_div = add_div
        self.d_text = d_text
        self.div_proj_dim = div_proj_dim


        # Context + Gate
        self.text_ctx = LearnableTwoStreamContext(d=d_text, use_cosine=True, init_method="xavier_uniform")
        self.gate = AgreementGate(d=d_text, hidden=gate_hidden, scalar=gate_scalar, use_cos=True)

        # === PRE injection settings ===
        
        self.use_pre = gate_usage in ('pre', 'both')
        self.pre_include_div = self.use_pre and self.add_div    # div를 pre에도 붙일지
        self.pre_mix_dim = d_text  #50
        self.pre_dim = self.pre_mix_dim + (1 if self.pre_include_div else 0)


        # if self.use_pre:
        #     self.proj_pre = nn.Linear(d_text, self.pre_mix_dim)
        #     # div: (B,1) -> (B, div_proj_dim)
        #     if self.pre_include_div:
        #         self.proj_div = nn.Sequential(
        #             nn.Linear(1, self.div_proj_dim),
        #             nn.Tanh(),
        #             nn.LayerNorm(self.div_proj_dim),
        #         )

        # LSTM input dim depends on pre features
        # pre_extra = (self.pre_mix_dim + (self.div_proj_dim if self.pre_include_div else 0)) if self.use_pre else 0
        
        # lstm_in = num_features + pre_extra
        lstm_in = num_features + (self.pre_dim if self.use_pre else 0)
        self.lin_x = nn.Linear(lstm_in, lstm_in)
        self.lstm = nn.LSTM(input_size=lstm_in,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout)

        # Temporal attention over LSTM hidden states
        self.lin_a1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_a2 = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.use_post = gate_usage in ('post')
        post_extra = 0
        if self.use_post:
            post_extra += d_text
            if self.add_div:
                post_extra += 1
        if self.add_div:
            post_extra += 1
        self.in_dim = 2 * hidden_dim + post_extra
        self.lin_out = nn.Linear(self.in_dim, 1)

        self.initialize()

    def initialize(self):
        nn.init.zeros_(self.lin_a1.bias)
        lstm_params = list(self.lstm.parameters())
        if len(lstm_params) >= 4:
            nn.init.zeros_(lstm_params[2])  # bias_ih_l0
            nn.init.zeros_(lstm_params[3])  # bias_hh_l0
        for layer in [self.lin_x, self.lin_a1, self.lin_a2, self.lin_out]:
            nn.init.xavier_uniform_(layer.weight)

    def l2_norm(self):
        l2_norm = 0
        for weight in [self.lin_out.weight, self.lin_out.bias]:
            l2_norm = l2_norm + (weight ** 2).sum()
        return l2_norm / 2

    def _temporal_state(self, x):
        # x: (B, T, F_eff)
        x2 = torch.tanh(self.lin_x(x))
        h, _ = self.lstm(x2)            # (B, T, H)
        h_last = h[:, -1, :]            # (B, H)
        h2 = torch.tanh(self.lin_a1(h)) # (B, T, H)
        score = self.softmax(self.lin_a2(h2)) # (B, T, 1)
        out = torch.bmm(score.transpose(1, 2), h).squeeze(1)  # (B, H)
        base = torch.cat([h_last, out], dim=1)                # (B, 2H)
        return base

    def to_state(self, x, news_seq=None, tweet_seq=None, news_mask=None, tweet_mask=None):
        """
        x: (B, T, F)
        news_seq/tweet_seq: (B, N, d_text) with masks (B, N) [True=keep]
        """
        print("x.shape", x.shape)
        B, T, _ = x.shape
        print("B", B, "T", T)
        use_ctx = (news_seq is not None) and (tweet_seq is not None)

        # (1) Build contexts + cosine/divergence
        if use_ctx:
            c_news, c_tweet, cos, div = self.text_ctx(news_seq, tweet_seq, news_mask, tweet_mask)  # (B,d),(B,d),(B,1),(B,1)
            # (2) Agreement-aware gate -> mixed vector
            mix, g = self.gate(c_news, c_tweet, cos)  # mix: (B,d) g is scaler
        else:
            mix = x.new_zeros((B, self.d_text))
            div = x.new_zeros((B, 1))
            dim_g = 1 if self.gate.scalar else self.d_text
            g = x.new_full((B, dim_g), 0.5, device=x.device)
        self.last_gate_values = g.detach()
        # (3) PRE injection (concat to every timestep)  <<< NEW: div can be included here too
        if self.use_pre:
            z = torch.cat([mix, g], dim=-1) if self.pre_include_div else mix  # (B, d_text [+1])
            # (sanity) ensure LSTM input matches __init__-time sizing
            assert z.size(-1) == self.pre_dim, f"Expected pre_dim={self.pre_dim}, got {z.size(-1)}"
            z_rep = z.unsqueeze(1).expand(B, T, z.size(-1))  # (B,T,pre_dim)
            x = torch.cat([x, z_rep], dim=-1)                                                        # (B, T, F + pre_dim)

        # (4) Temporal modeling
        base = self._temporal_state(x)  # (B, 2H)

        # (5) POST injection (concat to state)
        if self.use_post:
            extras = [mix]              # (B, d_text)
            if self.add_div:
                extras.append(div)      # (B, 1)
            base = torch.cat([base] + extras, dim=-1)  # (B, 2H + d_text + [1])
        extras = []
        if self.add_div:
            extras.append(div)
            base = torch.cat([base] + extras, dim=-1)
        return base

    def forward(self, x, news_seq=None, news_mask=None, tweet_seq=None, tweet_mask=None):
        state = self.to_state(x, news_seq, tweet_seq, news_mask, tweet_mask)
        return self.lin_out(state).view(-1)



class DualModel(nn.Module):
    def __init__(self, num_features, hidden_dim, lstm_layers=1, dropout=0.0):
        super().__init__()
        num_features = num_features // 2
        self.model1 = MainModel(num_features, hidden_dim, lstm_layers, dropout)
        self.model2 = MainModel(num_features, hidden_dim, lstm_layers, dropout)
        self.lin_out = nn.Linear(4 * hidden_dim, 1)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)

    def l2_norm(self):
        l2_norm = 0
        for weight in [self.lin_out.weight, self.lin_out.bias]:
            l2_norm = l2_norm + (weight ** 2).sum()
        return l2_norm / 2

    def forward(self, x):
        num_features = x.size(2) // 2
        out1 = self.model1.to_state(x[:, :, :num_features])
        out2 = self.model2.to_state(x[:, :, num_features:])
        out = torch.cat([out1, out2], dim=1)
        return self.lin_out(out).view(-1)


