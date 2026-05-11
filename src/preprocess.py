import ast
import os
from os import path as osp
import pickle
from typing import Optional, Tuple
import math

import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn.functional as F
from pretrain_news import NewsPriceModel
from models import Attention
import utils

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from db import read_news_data
import torch.nn as nn

BASE_MODEL = "bert-base-uncased"


class NewsGuidedTweetAttention(nn.Module):
    def __init__(self, d_model=768, d_kv=64, n_heads=8, d_out=256, dropout=0.1, pool_method="dot", n_pool=1, pool_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.q = nn.Linear(d_model, n_heads * d_kv)
        self.k = nn.Linear(d_model, n_heads * d_kv)
        self.v = nn.Linear(d_model, n_heads * d_kv)
        self.proj = nn.Linear(n_heads * d_kv, d_out)
        self.drop = nn.Dropout(dropout)
        self.pool = AttnPool1D(d_out, n_seeds=n_pool, method=pool_method, dropout=pool_dropout)


    def forward(self, news_embs, tweet_embs, tweet_mask=None, news_mask=None, return_weights: bool = False):
        """
        news_embs: [J, d_model] or [B, J, d_model]
        tweet_embs:[I, d_model] or [B, I, d_model]
        tweet_mask: [B, I] or [I] (True for keep / 1)
        returns g_t: [d_out] or [B, d_out]
        """
        if news_embs.dim() == 2:
            news_embs = news_embs.unsqueeze(0)      # [1, J, d]
            tweet_embs = tweet_embs.unsqueeze(0)    # [1, I, d]

        B, J, d = news_embs.shape
        _, I, _ = tweet_embs.shape

        Q = self.q(news_embs).view(B, J, self.n_heads, self.d_kv).transpose(1,2) # [B,H,J,D]
        K = self.k(tweet_embs).view(B, I, self.n_heads, self.d_kv).transpose(1,2)# [B,H,I,D]
        V = self.v(tweet_embs).view(B, I, self.n_heads, self.d_kv).transpose(1,2)# [B,H,I,D]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_kv ** 0.5)         # [B,H,J,I]
        if tweet_mask is not None:
            if tweet_mask.dim() == 1: tweet_mask = tweet_mask.unsqueeze(0)       # [1,I]
            tweet_mask = tweet_mask.unsqueeze(1).unsqueeze(1)                    # [B,1,1,I]
            attn = attn.masked_fill(~tweet_mask.bool(), float('-inf'))

        A = torch.softmax(attn, dim=-1)                                          # [B,H,J,I]
        A = self.drop(A)
        Z = torch.matmul(A, V)                                                   # [B,H,J,D]
        Z = Z.transpose(1,2).contiguous().view(B, J, self.n_heads * self.d_kv)   # [B,J,H*D]

        Z = self.proj(Z)                                                         # [B,J,d_out]
        # NEW: attention pooling over news dimension
        g_t, news_alpha = self.pool(Z, news_mask)                                 # [B,d_out], [B,J]  (n_pool=1)

        if return_weights:
            return g_t, {"news_weights": news_alpha, "tweet_weights": A}         # 가시화/디버깅용
        return g_t

def inject_noise_vectors_with_mask(embs, noise_ratio, device):
    """
    Returns: 
      - noisy_embs: Tensor with noise injected
      - noise_mask: Boolean tensor (True = Noise, False = Real)
    """
    if embs is None:
        return None, None
    
    n_tweets, dim = embs.shape
    noise_mask = torch.zeros(n_tweets, dtype=torch.bool, device=device)
    
    if noise_ratio <= 0.0:
        return embs, noise_mask

    num_noise = int(n_tweets * noise_ratio)
    if num_noise == 0:
        return embs, noise_mask

    noisy_embs = embs.clone()
    noise = torch.randn(num_noise, dim, device=device)
    noise = F.normalize(noise, p=2, dim=1)

    # Select indices
    perm = torch.randperm(n_tweets, device=device)
    noise_idx = perm[:num_noise]
    
    # Apply noise and set mask
    noisy_embs[noise_idx] = noise
    noise_mask[noise_idx] = True
    
    return noisy_embs, noise_mask

class FeatureGenerator:
    def __init__(self, features, mask, window, mode='base'):
        self.features = features
        self.mask = mask
        self.window = window
        self.mode = mode

    def min_index(self):
        if self.mode == 'base':
            return self.window
        elif self.mode == 'dual-sampling':
            return 5 * self.window - 4
        else:
            raise ValueError(self.mode)

    def get_x(self, stock, *index):
        x_list, m_list = [], []
        for idx in index:
            x_list.append(self.features[idx, stock])
            m_list.append(self.mask[idx, stock])
        x_out = torch.cat(x_list, dim=1)
        m_out = torch.stack(m_list).all(0)
        return x_out, m_out

    def to_features(self, data_idx, stock_idx):
        if self.mode == 'base':
            idx1 = list(range(data_idx - self.window, data_idx))
            return self.get_x(stock_idx, idx1)
        elif self.mode == 'dual-sampling':
            idx1 = list(range(data_idx - self.window, data_idx))
            idx2 = [data_idx - 1 - 5 * j for j in range(self.window)]
            idx2.reverse()
            return self.get_x(stock_idx, idx1, idx2)
        else:
            raise ValueError(self.mode)

@torch.no_grad()
def embed_texts(tokenizer, texts, model, device, batch_size=32, max_len=512):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # ✅ NewsPriceModel에는 encode_news 사용
        z = model.encode_news(enc["input_ids"], enc["attention_mask"])  # [B, dim]
        all_vecs.append(z.detach())
    return torch.cat(all_vecs, dim=0)   # [N, dim]
    
def alpha_from_agree(agree: torch.Tensor, t: float = 0.2, tau: float = 0.15):
    # agree is scalar tensor in [-1, 1] (cosine sim)
    x = (agree - t) / tau
    return torch.sigmoid(x)

class NewsTweetGating(nn.Module):
    def __init__(self, input_dim, tweet_dim, output_dim=1):
        super(NewsTweetGating, self).__init__()
        # Linear layer for learning the gate
        self.gate_layer = nn.Linear(input_dim + tweet_dim, output_dim)
    
    def forward(self, c_news, c_tweet):
        # Combine both contexts and compute the gating score
        combined = torch.cat([c_news, c_tweet], dim=-1)  # (d,)
        gate = torch.sigmoid(self.gate_layer(combined))  # (1,)
        return gate

def _attention_pool(query: torch.Tensor,
                    keys: torch.Tensor,
                    values: torch.Tensor) -> torch.Tensor:
    """
    Simple scaled dot-product attention over a *set* (no batch).

    query:  (d,)
    keys:   (N, d)
    values: (N, d_v)
    return: (d_v,)
    """
    if keys is None or keys.numel() == 0:
        return values.new_zeros(values.size(-1))

    d = keys.size(-1)
    # scores: (N,)
    scores = (keys @ query) / math.sqrt(d)
    weights = torch.softmax(scores, dim=0)  # (N,)
    # (N, 1) * (N, d_v) -> (N, d_v) -> (d_v,)
    return (weights.unsqueeze(-1) * values).sum(dim=0)



def two_stream_news_tweet_context(
    news_embs: Optional[torch.Tensor],   # (N_news, d)
    tweet_embs: Optional[torch.Tensor],  # (N_tweet, d)
    use_news_filter: bool = True,        # <- kept for signature, but unused now
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      c_news:  (d,)   - news-only context
      c_tweet: (d,)   - tweet-only context (simple attention pooling)
    """
    has_news = news_embs is not None and news_embs.numel() > 0
    has_tweets = tweet_embs is not None and tweet_embs.numel() > 0

    if not has_news and not has_tweets:
        raise ValueError("Both news_embs and tweet_embs are empty.")

    any_emb = news_embs if has_news else tweet_embs
    device = any_emb.device
    d = any_emb.size(-1)

    # ---------- News context ----------
    if has_news:
        q_news = news_embs.mean(dim=0)                    # (d,)
        c_news = _attention_pool(q_news, news_embs, news_embs)  # (d,)
    else:
        c_news = torch.zeros(d, device=device)

    # ---------- Tweet context ----------
    if has_tweets:
        q_tweet = c_news if has_news else tweet_embs.mean(dim=0)
        tweet_vals = tweet_embs
        # NOTE: no reliability weighting for now
        c_tweet = _attention_pool(q_tweet, tweet_embs, tweet_vals)  # (d,)
    else:
        c_tweet = torch.zeros(d, device=device)

    return c_news, c_tweet


class AttnPool1D(nn.Module):
    """
    X: [B, J, d] -> pooled: [B, d]  (n_seeds=1) or [B, S, d] (n_seeds=S)
    method='dot': 학습 가능한 쿼리 Q와 점곱 어텐션
    method='mlp': 항목별 점수 MLP로 산출
    """
    def __init__(self, d, n_seeds: int = 1, method: str = "dot", hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        assert method in ("dot", "mlp")
        self.method = method
        self.n_seeds = n_seeds
        self.d = d
        self.drop = nn.Dropout(dropout)
        if method == "dot":
            # S개의 학습 쿼리
            self.Q = nn.Parameter(torch.randn(n_seeds, d))  # [S, d]
        else:
            h = hidden or max(64, d // 2)
            self.mlp = nn.Sequential(
                nn.Linear(d, h),
                nn.Tanh(),
                nn.Linear(h, n_seeds),
            )

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        """
        X: [B, J, d], mask: [B, J] (True=유효)
        returns:
          pooled: [B, d] (S=1) 또는 [B, S, d]
          attn_w: [B, J] (S=1) 또는 [B, J, S]
        """
        B, J, d = X.shape
        if self.method == "dot":
            # logits: [B, J, S]  (각 시드 쿼리에 대한 항목 가중치)
            logits = (X @ self.Q.t()) / math.sqrt(d)  # [B, J, S]
        else:
            logits = self.mlp(X)  # [B, J, S]

        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, J, 1]
            logits = logits.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(logits, dim=1)  # J축 정규화 -> [B, J, S]
        attn = self.drop(attn)
        # (B, J, S) * (B, J, d) -> (B, S, d)
        pooled = (attn.transpose(1, 2) @ X)  # [B, S, d]

        if self.n_seeds == 1:
            return pooled.squeeze(1), attn.squeeze(-1)  # [B, d], [B, J]
        return pooled, attn  # [B, S, d], [B, J, S]

def compress_sequence_pooling(seq_tensor: torch.Tensor, max_len: int, mode='mean') -> torch.Tensor:
    """
    Compresses a sequence of length N to max_len by pooling chunks.
    Args:
        seq_tensor: (N, dim)
        max_len: target length (e.g., 128)
        mode: 'mean' or 'max'
    Returns:
        (max_len, dim)
    """
    N, dim = seq_tensor.shape
    if N <= max_len:
        return seq_tensor

    # Calculate chunk size (e.g., 1000 / 128 = 7.8 -> 8)
    chunk_size = math.ceil(N / max_len)
    
    # We need to pad the sequence so it can be reshaped perfectly
    target_full_len = chunk_size * max_len
    pad_len = target_full_len - N
    
    if pad_len > 0:
        # Pad with the last vector to avoid zero-bias
        padding = seq_tensor[-1].unsqueeze(0).expand(pad_len, -1)
        seq_tensor = torch.cat([seq_tensor, padding], dim=0)

    # Reshape: (max_len, chunk_size, dim)
    seq_reshaped = seq_tensor.view(max_len, chunk_size, dim)

    # Pool
    if mode == 'mean':
        compressed = seq_reshaped.mean(dim=1) # (max_len, dim)
    elif mode == 'max':
        compressed, _ = seq_reshaped.max(dim=1)
    else:
        raise ValueError("mode must be 'mean' or 'max'")
        
    return compressed

MAX_NEWS_PER_DAY = 256      # 필요하면 더 줄여서 256, 128, 64 등으로
MAX_TWEETS_PER_DAY = 256    
def make_reliable_tweet_features(
    data,
    out_path,
    device,
    dates,
    start_idx,
    end_idx,
    price_features,
    window,
    noise_ratio,
    use_news_filter: bool = True,
):
    """
    price_features: (B, N_stocks, F_price)
    Returns:
      reliable_tweet_features: (B, N_stocks, F_price)  # day-level global feature
      news_seq_tensor:  (len(dates), max_news,  d_text)
      news_mask_tensor: (len(dates), max_news)
      tweet_seq_tensor: (len(dates), max_tweet, d_text)
      tweet_mask_tensor:(len(dates), max_tweet)
    """
    print(dates, start_idx, end_idx)
    print(f"Generating features... Noise Ratio: {noise_ratio}")
    price_features = price_features.to(device)
    B, num_stocks, F_price = price_features.shape

    reliable_tweet_features = torch.full_like(price_features, np.nan, device=device)
    reliable_news_features = torch.full_like(price_features, np.nan, device=device)
    #Settings for pretrain news model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # bert_model = BertContrastive(base_model_name=BASE_MODEL).to(device)
    # ✅ 2) NewsPriceModel 생성 (학습 때와 같은 하이퍼파라미터)
    price_dim = window * F_price
    news_model = NewsPriceModel(
        num_stocks=num_stocks,
        base_model_name=BASE_MODEL,
        news_proj_dim=128,      # 학습 때 쓴 값과 동일해야 함
        price_dim=price_dim,    # window * F_price
        window=window,
        price_hidden=32,
        lstm_hidden=32,
        lstm_layers=1,
        lstm_dropout=0.0,
        fusion_hidden=128,
        pretrained_ckpt=None,   # 여기서는 ckpt 안 씀, 아래에서 직접 로드
    ).to(device)

    model_path = utils.news_emb_model_path(data)
    print("model_path", model_path)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    news_model.load_state_dict(state_dict)
    news_model.eval()
    # ✅ 3) 이 모델에서 stock embedding 사용
    stock_news_embs = news_model.stock_embeddings().to(device)  # (N_stocks, d_news)
    print("stock_embs for news", stock_news_embs.shape)

    #Settings for tweet pretrain model
    sp_path = utils.sp_model_path(data)
    emb_path = utils.emb_model_path(out_path)
    emb_model = torch.load(emb_path, map_location=device, weights_only=False)
    emb_model.eval()
    stock_tweet_embs = emb_model.stock_emb().to(device)  # (N_stocks, d_tweet)

    print("stock_tweet_embs for tweets", stock_tweet_embs.shape)

    # 공통 텍스트 임베딩 차원 (트윗 차원으로 통일)
    d_text = stock_tweet_embs.size(1)
    print("d_text size", d_text)
    # ---- Read news & tweets ----
    print("dates", dates[0], dates[end_idx])
    news_df = read_news_data(dates[0], dates[end_idx], None, "ALL")
    print("news data fetched successfully from database...")

    news_df["date"] = pd.to_datetime(news_df["date"], utc=True).dt.tz_localize(None)
    news_df["date"] = news_df["date"].dt.normalize()

    if news_df.empty:
        print("news_df is empty")
    else:
        # make sure it's datetime (you already normalized above)
        d = pd.to_datetime(news_df["date"], errors="coerce")

        start_date = d.min()
        end_date   = d.max()
        total_news = len(news_df)           # or: news_df.shape[0]

        print("Start date:", start_date)
        print("End date  :", end_date)
        print("Total news:", total_news)

        # (optional) coverage in days
        if pd.notna(start_date) and pd.notna(end_date):
            print("Coverage (days):", (end_date - start_date).days + 1)

        # (optional) daily counts
        daily_counts = d.value_counts().sort_index()
        print("daily_counts", daily_counts.head())    
    try:
        tweets = torch.from_numpy(np.load(osp.join(sp_path, f"{data}_out.npy"))).to(device)
        df_info = pd.read_csv(osp.join(sp_path, f"{data}_out.csv"))
        df_info["positions"] = df_info["positions"].apply(
            lambda x: np.array(ast.literal_eval(x))
        )
        df_info["date"] = pd.to_datetime(df_info["date"], utc=True).dt.tz_localize(None)
        df_info["date"] = df_info["date"].dt.normalize()
        print("tweet data fetched successfully...")
    except Exception as e:
        print("Failed to load tweet features:", e)
        tweets, df_info = None, None

    attention = Attention()
    news_guided_attention = None
    
    # news / tweet 를 d_text 로 프로젝트하기 위한 Linear들
    news_to_text_proj = None   # news_dim -> d_text
    tweet_to_text_proj = None  # (optional) tweet_dim -> d_text (dim 안 맞을 때만)

    # === day별 raw(프로젝션 된) sequence 저장용 리스트 ===
    all_news_seq = [None] * len(dates)
    all_tweet_seq = [None] * len(dates)
    all_real_weights = []
    all_noise_weights = []
    for idx in tqdm(range(start_idx, end_idx), total=end_idx - start_idx,
                    desc="Making reliable tweet features"):
        current_date = dates[idx]
        # print("current_date",current_date)
        # ----- NEWS -----
        last_news_date = news_df.loc[news_df["date"] <= current_date, "date"].max()
        # print("last_news_date", last_news_date)
        if pd.isna(last_news_date):
            day_news_embs = None
        else:
            day_news_df = news_df[news_df["date"] == last_news_date]
            print("news len", len(day_news_df))
            day_news_texts = day_news_df["text"].dropna().tolist()
            if len(day_news_texts) == 0:
                day_news_embs = None
            else:
                day_news_embs = embed_texts(
                    tokenizer,
                    day_news_texts,
                    news_model,
                    device,
                    batch_size=8,
                    max_len=512,
                )  # (N_news, d_news)

        # ----- TWEETS -----
        if tweets is not None and df_info is not None:
            
            last_tweet_date = df_info.loc[df_info["date"] <= current_date, "date"].max()
            if pd.isna(last_tweet_date):
                day_tweet_embs = None
                print("tweets day_tweet_embs is not found", current_date)
            else:
                tweet_index = df_info["date"] == last_tweet_date
                curr_info = df_info.loc[tweet_index]
                curr_tweets = tweets[tweet_index]
                print("tweets len",current_date, len(curr_tweets))
                if len(curr_info) == 0:
                    print("tweets day_tweet_embs is not found inside>", current_date)
                    day_tweet_embs = None
                else:
                    tweet_len = torch.from_numpy(curr_info["length"].values).to(device)
                    stock_pos = torch.zeros(
                        curr_tweets.shape, dtype=torch.bool, device=device
                    )
                    for i, jl in enumerate(curr_info["positions"]):
                        stock_pos[i, jl] = 1
                    day_tweet_embs = emb_model(curr_tweets, tweet_len, stock_pos)  # (N_tweet, d_tweet_raw)
                    # noise_mask = None
                    # if day_tweet_embs is not None and noise_ratio >= 0.0:
                    #     # Inject noise and get a mask (True=Noise, False=Real)
                    #     day_tweet_embs, noise_mask = inject_noise_vectors_with_mask(
                    #         day_tweet_embs, noise_ratio, device
                    #     )
                    
        else:
            print("tweets is not found", current_date)
            day_tweet_embs = None
        
        
        # print("day_news_embs", day_news_embs.shape, "day_tweet_embs", day_tweet_embs.shape)
        # 둘 다 비어 있으면 이 날은 스킵
        if (
            day_news_embs is None or day_news_embs.numel() == 0
        ) and (
            day_tweet_embs is None or day_tweet_embs.numel() == 0
        ):
            print("day_news_embs.size(1)", day_news_embs.size(1), current_date)
            print("day_tweet_embs.size(1)", day_tweet_embs.size(1), current_date)
            continue
        stock_news_emb_proj = None 
        
        if stock_news_emb_proj is None:
            news_dim = day_news_embs.size(1)
            linear_proj = nn.Linear(news_dim, d_text, bias=False).to(device)
            stock_news_emb_proj = linear_proj(stock_news_embs)


        # ----- 공통 차원 d_text 로 projection -----
        day_news_proj = None
        if day_news_embs is not None and day_news_embs.numel() > 0:
            if day_news_embs.size(1) == d_text:
                day_news_proj = day_news_embs
            else:
                if news_to_text_proj is None:
                    news_dim = day_news_embs.size(1)
                    news_to_text_proj = nn.Linear(news_dim, d_text, bias=False).to(device)
                    print(f"Created news_to_text_proj: {news_dim} -> {d_text}")
            day_news_proj = news_to_text_proj(day_news_embs)  # (N_news, d_text)

        day_tweet_proj = None
        if day_tweet_embs is not None and day_tweet_embs.numel() > 0:
            tdim = day_tweet_embs.size(1)
            if tdim != d_text:
                if tweet_to_text_proj is None:
                    tweet_to_text_proj = nn.Linear(tdim, d_text, bias=False).to(device)
                    print(f"Created tweet_to_text_proj: {tdim} -> {d_text}")
                day_tweet_proj = tweet_to_text_proj(day_tweet_embs)
            else:
                day_tweet_proj = day_tweet_embs  # 이미 d_text
        
        if day_news_proj is not None and day_news_proj.size(0) > MAX_NEWS_PER_DAY:
            # Instead of slicing, we pool (compress) the 1000 items into 128 representations
            day_news_proj = compress_sequence_pooling(day_news_proj, MAX_NEWS_PER_DAY, mode='mean')

        if day_tweet_proj is not None and day_tweet_proj.size(0) > MAX_TWEETS_PER_DAY:
            # Same for tweets
            day_tweet_proj = compress_sequence_pooling(day_tweet_proj, MAX_TWEETS_PER_DAY, mode='mean')
        # === truncate by MAX_*_PER_DAY ===
        # if day_news_proj is not None and day_news_proj.size(0) > MAX_NEWS_PER_DAY:
        #     day_news_proj = day_news_proj[:MAX_NEWS_PER_DAY]

        # if day_tweet_proj is not None and day_tweet_proj.size(0) > MAX_TWEETS_PER_DAY:
        #     day_tweet_proj = day_tweet_proj[:MAX_TWEETS_PER_DAY]
            # if noise_mask is not None:
            #     noise_mask = noise_mask[:MAX_TWEETS_PER_DAY]

        # === projection 된 것만 저장 ===
        if day_news_proj is not None:
            all_news_seq[idx] = day_news_proj.detach().cpu()   # (N_news, d_text)
        if day_tweet_proj is not None:
            all_tweet_seq[idx] = day_tweet_proj.detach().cpu() # (N_tweet, d_text)

        # 둘 다 None이면 더 할 게 없음
        if (day_news_proj is None or day_news_proj.numel() == 0) and \
           (day_tweet_proj is None or day_tweet_proj.numel() == 0):
            continue
        
        # ---- News-guided Tweet Attention 준비 ----
        if news_guided_attention is None:
            d_model = d_text
            d_kv = max(1, d_model)
            news_guided_attention = NewsGuidedTweetAttention(
                d_model=d_model,
                d_kv=d_kv,
                n_heads=8,
                d_out=d_model,  # keep same dim
                dropout=0.2,
                pool_method="dot"
            ).to(device)
            print(f"Created NewsGuidedTweetAttention with d_model={d_model}, d_kv={d_kv}")
            # news_tweet_gate = NewsTweetGating(d_text, d_text) 
        news_mask_b = torch.ones(1, day_news_proj.size(0), dtype=torch.bool, device=device)
        tweet_mask_b = torch.ones(1, day_tweet_proj.size(0), dtype=torch.bool, device=device)


        # ---- day-level context vector (tweet_cxt) ----
        if day_news_proj is None:
            tweet_cxt = day_tweet_embs.mean(dim=0)          # (d_text,)
        elif day_tweet_embs is None:
            tweet_cxt = day_news_proj.mean(dim=0)           # (d_text,)
        else:
            tweet_cxt, w_dict = news_guided_attention(
                                day_news_proj, day_tweet_proj,
                                tweet_mask=None, news_mask=None,
                                return_weights=True
                            ) # (d_text,)
            news_cxt, weights = news_guided_attention(day_tweet_proj, day_news_proj, tweet_mask=None, news_mask=None,
                                return_weights=True)

            # w_dict['tweet_weights'] shape: [1, 8, J, I] -> Average heads & news
            # Result shape: [I] (Importance of each tweet)
            # attn_scores = w_dict['tweet_weights'].mean(dim=(0, 1, 2))
            # if noise_mask is not None:
            #     # Ensure sizes match (in case of truncation)
            #     valid_len = min(len(noise_mask), len(attn_scores))
            #     current_mask = noise_mask[:valid_len]
            #     current_scores = attn_scores[:valid_len]

            #     # Separate weights
            #     noise_vals = current_scores[current_mask].detach().cpu().tolist()
            #     real_vals = current_scores[~current_mask].detach().cpu().tolist()
                
            #     all_noise_weights.extend(noise_vals)
            #     all_real_weights.extend(real_vals)
        # print("tweet_cxt shape before attention", tweet_cxt.shape, news_cxt.shape)

        # Save weights analysis to disk
        # if len(all_noise_weights) >= 0:
        #     print(f"Saving Analysis: {len(all_noise_weights)} noise vs {len(all_real_weights)} real weights")
        #     analysis_path = osp.join(osp.dirname(out_path), f"noise_analysis_{noise_ratio}.pkl")
        #     with open(analysis_path, "wb") as f:
        #         pickle.dump({"real": all_real_weights, "noise": all_noise_weights}, f)

        # ---- context -> price feature 공간으로 맵핑 ----
        global_x = attention(
            query=tweet_cxt,
            keys=stock_tweet_embs,
            values=price_features[idx],  # (N_stocks, F_price)
        )  # (F_price,) or (1, F_price)
        global_x = global_x.view(-1)  # (F_price,)
        
        global_y = attention(
            query=news_cxt,
            keys=stock_news_emb_proj,
            values=price_features[idx],  # (N_stocks, F_price)
        )
        global_y = global_y.view(-1) 
        # broadcast to all stocks for that day
        for idx_stock in range(num_stocks):
            reliable_tweet_features[idx, idx_stock] = global_x
        for idx_stock in range(num_stocks):
            reliable_news_features[idx, idx_stock] = global_y    

    # === day별 sequence를 패딩해서 텐서로 만들기 ===
    max_news = max((x.size(0) for x in all_news_seq if x is not None), default=1)
    max_tweet = max((x.size(0) for x in all_tweet_seq if x is not None), default=1)

    news_seq_tensor  = torch.zeros(len(dates), max_news,  d_text)
    news_mask_tensor = torch.zeros(len(dates), max_news,  dtype=torch.bool)
    tweet_seq_tensor  = torch.zeros(len(dates), max_tweet, d_text)
    tweet_mask_tensor = torch.zeros(len(dates), max_tweet, dtype=torch.bool)
    # print("tweet_seq_tensor.shape", tweet_seq_tensor.shape)
    # print("tweet_mask_tensor.shape", tweet_mask_tensor.shape)
    # print("news_seq_tensor.shape", news_seq_tensor.shape)
    for idx in range(len(dates)):
        if all_news_seq[idx] is not None:
            n = all_news_seq[idx].size(0)
            news_seq_tensor[idx, :n, :] = all_news_seq[idx]
            news_mask_tensor[idx, :n] = True
        if all_tweet_seq[idx] is not None:
            m = all_tweet_seq[idx].size(0)
            tweet_seq_tensor[idx, :m, :] = all_tweet_seq[idx]
            tweet_mask_tensor[idx, :m] = True

    # If everything was NaN (no news/tweets in range), return None for consistency
    if torch.isnan(reliable_tweet_features).all():
        reliable_tweet_features = None
    else:
        reliable_tweet_features = reliable_tweet_features.cpu()

    if torch.isnan(reliable_news_features).all():
        reliable_news_features = None
    else:
        reliable_news_features = reliable_news_features.cpu()
    print("reliable_news_features and tweet_features", reliable_news_features.shape)
    # === 최종 반환 ===
    return reliable_tweet_features, reliable_news_features, news_seq_tensor, news_mask_tensor, tweet_seq_tensor, tweet_mask_tensor




@torch.no_grad()
def make_tweet_features(data, out_path, device, dates, start_idx, end_idx,
                        price_features, global_trend=True, local_trend=True):
    if not (global_trend or local_trend):
        return None

    price_features = price_features.to(device)
    global_features = torch.full_like(price_features, np.nan, device=device)
    local_features = torch.full_like(price_features, np.nan, device=device)

    num_stocks = price_features.size(1)
    emb_path = utils.emb_model_path(out_path)
    print("emb_path of twitter", emb_path)
    emb_model = torch.load(emb_path, map_location=device, weights_only=False)
    emb_model.eval()
    stock_embs = emb_model.stock_emb()
    print('stock_embs in twitter', stock_embs.shape)
    attention = Attention()

    sp_path = utils.sp_model_path(data)
    df_info = pd.read_csv(osp.join(sp_path, f'{data}_out.csv'))
    df_info['positions'] = df_info['positions'] \
        .apply(lambda x: np.array(ast.literal_eval(x)))
    tweets = torch.from_numpy(np.load(osp.join(sp_path, f'{data}_out.npy'))).to(device)

    for idx in tqdm(range(start_idx, end_idx), total=end_idx - start_idx,
                desc="Making Tweets features"):
        last_date = df_info.loc[df_info['date'] <= dates[idx], 'date'].max()
        index = df_info['date'] == last_date
        curr_info = df_info.loc[index]
        curr_tweets = tweets[index]
        
        tweet_len = torch.from_numpy(curr_info['length'].values).to(device)
        stock_pos = torch.zeros(curr_tweets.shape, dtype=torch.bool, device=device)
        for i, jl in enumerate(curr_info['positions']):
            stock_pos[i, jl] = 1
        tweet_embs = emb_model(curr_tweets, tweet_len, stock_pos)
        print("tweet_embs", tweet_embs.shape)
        if global_trend:
            global_x = attention(query=tweet_embs.mean(axis=0),
                                 keys=stock_embs,
                                 values=price_features[idx])
            global_x = global_x.view(-1)
            for idx_stock in range(num_stocks):
                global_features[idx, idx_stock] = global_x

        if local_trend:
            local_tweet_embs = attention(query=stock_embs,
                                         keys=tweet_embs,
                                         values=tweet_embs)
            local_x = attention(query=local_tweet_embs,
                                keys=stock_embs,
                                values=price_features[idx])
            for idx_stock in range(num_stocks):
                local_features[idx, idx_stock] = local_x[idx_stock]

    if global_trend and local_trend:
        return torch.cat([global_features, local_features], dim=2).cpu()
    elif global_trend:
        return global_features.cpu()
    elif local_trend:
        return local_features.cpu()
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='base')

    parser.add_argument('--global-trend', type=utils.str2bool, default=True)
    parser.add_argument('--local-trend', type=utils.str2bool, default=True)
    parser.add_argument('--reliable-tweet-trend', type=utils.str2bool, default=True)
    parser.add_argument('--global-news-trend', type=utils.str2bool, default=True)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--noise-ratio', type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    data = args.data
    print("data", data)
    seed = args.seed
    window = args.window
    noise_ratio = args.noise_ratio
    device = utils.to_device(args.gpu)
    out_path = utils.default_path(data, seed) if args.out is None else args.out
    news_model_path = utils.news_emb_model_path(data)
    trn_date, val_date, test_date, end_date = utils.get_date_info(data)

    data_path = os.path.join(utils.ROOT_PATH, 'data', data, 'price')
    dates, features, labels, mask, prices = utils.read_price_data(data_path)
  
    trn_idx, val_idx, test_idx, end_idx = \
        utils.get_date_index(dates, trn_date, val_date, test_date, end_date)
    num_stocks = features.size(1)
    print("Start to make News Global trend")
    (reliable_tweet_features,
     reliable_news_features,
     news_seq_tensor,
     news_mask_tensor,
     tweet_seq_tensor,
     tweet_mask_tensor) = make_reliable_tweet_features(
        data, out_path, device, dates, trn_idx, end_idx, features, window, noise_ratio
    )
    print("reliable_tweet_features shape", reliable_news_features.shape)
    print("Start to make tweet features")
    tweet_features = make_tweet_features(
        data, out_path, device, dates, trn_idx, end_idx, features,
        args.global_trend, args.local_trend)
    
    if tweet_features is not None:
        features = torch.cat([features, tweet_features], dim=2)
    #New-guided tweet attention module done
    if reliable_tweet_features is not None:
        features = torch.cat([features, reliable_tweet_features], dim=2)
    #tweet driven news attention module
    if reliable_news_features is not None:
        features = torch.cat([features, reliable_news_features], dim=2)
        
    print("features shape in preprocess", features.shape)
    generator = FeatureGenerator(features, mask, window, args.mode)
    start_idx = trn_idx + generator.min_index()

    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    # === NEW: sample별 day index 저장 ===
    train_day_idx, valid_day_idx, test_day_idx = [], [], []
    # New lists for raw prices
    p_train, p_valid, p_test = [], [], []

    for stock_idx in tqdm(range(num_stocks), desc="Creating features...", total=num_stocks):
        for data_idx in range(start_idx, end_idx + 1):
            x_out, x_mask = generator.to_features(data_idx, stock_idx)
            y = labels[data_idx, stock_idx]
            
            p = prices[data_idx, stock_idx]
            if not x_mask.all() or y == 0:
                continue
            y = (y > 0).long()

            if data_idx < val_idx:
                x_train.append(x_out)
                y_train.append(y)
                train_day_idx.append(data_idx)
                p_train.append(p)
            elif data_idx < test_idx:
                x_valid.append(x_out)
                y_valid.append(y)
                valid_day_idx.append(data_idx)
                p_valid.append(p)
            else:
                x_test.append(x_out)
                y_test.append(y)
                test_day_idx.append(data_idx)
                p_test.append(p)

    x_train = torch.stack(x_train)
    x_valid = torch.stack(x_valid)
    x_test = torch.stack(x_test)
    y_train = torch.stack(y_train)
    y_valid = torch.stack(y_valid)
    y_test = torch.stack(y_test)

    # Stack prices
    p_train = torch.tensor(p_train, dtype=torch.float32)
    p_valid = torch.tensor(p_valid, dtype=torch.float32)
    p_test = torch.tensor(p_test, dtype=torch.float32)

    train_day_idx = torch.tensor(train_day_idx, dtype=torch.long)
    valid_day_idx = torch.tensor(valid_day_idx, dtype=torch.long)
    test_day_idx = torch.tensor(test_day_idx, dtype=torch.long)

    out_path = utils.feature_path(out_path)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump([
            x_train, y_train, train_day_idx, p_train,
            x_valid, y_valid, valid_day_idx, p_valid, 
            x_test, y_test, test_day_idx, p_test,
            news_seq_tensor, news_mask_tensor,
            tweet_seq_tensor, tweet_mask_tensor,
        ], f)
    print("saved features pickle file..")

if __name__ == "__main__":
    main()

#python src/preprocess.py --data acl18 --gpu 0 --seed 0 --out src/out/acl18-0 --window 10