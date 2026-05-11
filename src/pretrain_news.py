import os
import re
import json
import random
from os import path as osp
import argparse
from tqdm import tqdm
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import utils  # 방금 보여주신 utils.py

from db import news_multisource_chunks, count_rows
# =========================
# CONFIG
# =========================
MAX_LEN = 512
BASE_MODEL = "bert-base-uncased"
EPOCHS = 5
LR = 3e-5
MAX_NEWS_PER_DAY = 50   # 필요하면 4, 8, 16 등으로 조절


# =========================
# (선택) 금융 뉴스 augmentation 관련 유틸
# 이미 다른 파일에 있으면 이 부분은 생략해도 됩니다.
# =========================
def load_fin_sent_words(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    return set()

def load_fin_synonyms(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

RE_NUMBER = re.compile(r"\b\d+(\.\d+)?%?\b")
RE_DATE   = re.compile(r"\b(20\d{2}|19\d{2})[-/\.]\d{1,2}([-/\.]\d{1,2})?\b")
RE_TICKER = re.compile(r"\$[A-Za-z]{1,5}")

FIN_SENT_WORDS = set()
FIN_SYNONYMS = {}

def safe_word_drop(tokens, p=0.1):
    new_tokens = []
    for tok in tokens:
        low = tok.lower()
        if low in FIN_SENT_WORDS or low.startswith("$"):
            new_tokens.append(tok)
            continue
        if random.random() < p:
            continue
        new_tokens.append(tok)
    return new_tokens or tokens

def mask_finance_spans(text, mask_num_prob=0.5, mask_date_prob=0.5, mask_ticker_prob=0.3):
    def _mask_num(m):
        return "[NUM]" if random.random() < mask_num_prob else m.group(0)
    def _mask_date(m):
        return "[DATE]" if random.random() < mask_date_prob else m.group(0)
    def _mask_ticker(m):
        return "[TICKER]" if random.random() < mask_ticker_prob else m.group(0)

    text = RE_NUMBER.sub(_mask_num, text)
    text = RE_DATE.sub(_mask_date, text)
    text = RE_TICKER.sub(_mask_ticker, text)
    return text

def synonym_replace(text, p=0.15):
    tokens = text.split()
    out = []
    for tok in tokens:
        low = tok.lower()
        if low in FIN_SENT_WORDS:
            out.append(tok)
            continue
        if low in FIN_SYNONYMS and random.random() < p:
            repl = random.choice(FIN_SYNONYMS[low])
            if tok and tok[0].isupper():
                repl = repl.capitalize()
            out.append(repl)
        else:
            out.append(tok)
    return " ".join(out)

def sentence_level_noise(text, p_drop=0.25):
    sents = re.split(r'(?<=[.!?])\s+', text)
    if len(sents) <= 1:
        return text
    kept = []
    for s in sents:
        low = s.lower()
        if any(w in low for w in FIN_SENT_WORDS):
            kept.append(s)
        else:
            if random.random() > p_drop:
                kept.append(s)
    if not kept:
        kept = [sents[0]]
    return " ".join(kept)

def finance_augment(text):
    text = mask_finance_spans(text, 0.5, 0.4, 0.3)
    text = sentence_level_noise(text, 0.25)
    text = synonym_replace(text, 0.2)
    toks = text.split()
    toks = safe_word_drop(toks, 0.1)
    return " ".join(toks)


def nt_xent(z1, z2, temperature=0.07):
    """
    z1, z2: [B, d]
    InfoNCE / NT-Xent loss (SimCLR 스타일)
    """
    bsz = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, d]
    sim = torch.matmul(z, z.t())    # [2B, 2B]

    # 자기 자신은 제외
    mask = torch.eye(2 * bsz, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)

    # positive index: i ↔ i+bsz, i+bsz ↔ i
    targets = torch.cat([
        torch.arange(bsz, 2 * bsz),
        torch.arange(0, bsz)
    ]).to(z.device)

    sim = sim / temperature
    loss = F.cross_entropy(sim, targets)
    return loss

# =========================
# Dataset
# =========================

class NewsPriceDataset(Dataset):
    """
    하나의 sample = (date_idx = t, stock_id = s)에 해당.
    - 그 날짜 t의 뉴스들 중 최대 MAX_NEWS_PER_DAY개를 뽑아서
      각각 augment → BERT → 평균 내서 day-level news embedding으로 사용.
    - price window는 pfeat_* 컬럼에 flatten 형태로 들어있음.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        daily_news_by_t: dict,      # {date_idx(int): [text1, text2, ...]}
        tokenizer,
        max_len: int = 128,
        use_augment: bool = True,
        max_news_per_day: int = MAX_NEWS_PER_DAY,
    ):
        self.df = df.reset_index(drop=True)
        self.daily_news_by_t = daily_news_by_t
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_augment = use_augment
        self.max_news_per_day = max_news_per_day

        # price feature 컬럼 미리 캐시
        self.price_cols = [c for c in self.df.columns if c.startswith("pfeat_")]
        if not self.price_cols:
            raise RuntimeError("Price feature columns (pfeat_*) not found in df.")

    def __len__(self):
        return len(self.df)

    def _sample_news_for_day(self, date_idx: int):
        texts = self.daily_news_by_t.get(date_idx, None)
        if not texts:
            # 뉴스가 아예 없는 날이면 dummy 텍스트
            return ["[NO_NEWS]"] * self.max_news_per_day

        if len(texts) >= self.max_news_per_day:
            # 랜덤으로 max_news_per_day개 선택
            return random.sample(texts, self.max_news_per_day)
        else:
            # 개수가 부족하면 중복 sampling으로 맞춰줌
            extra = random.choices(texts, k=self.max_news_per_day - len(texts))
            return list(texts) + extra

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        date_idx = int(row["date_idx"])
        stock_id = int(row["stock_id"])
        label = float(row["label"])
        price_vec = row[self.price_cols].values.astype("float32")

        # 이 날짜의 뉴스들 중 max_news_per_day개 선택
        base_texts = self._sample_news_for_day(date_idx)

        if self.use_augment:
            texts1 = [finance_augment(t) for t in base_texts]
            texts2 = [finance_augment(t) for t in base_texts]
        else:
            texts1 = base_texts
            texts2 = base_texts

        # 각각 [K, L] 텐서로 토크나이즈 (K = max_news_per_day)
        enc1 = self.tokenizer(
            texts1,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            texts2,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids_1": enc1["input_ids"],          # [K, L]
            "attention_mask_1": enc1["attention_mask"],
            "input_ids_2": enc2["input_ids"],          # [K, L]
            "attention_mask_2": enc2["attention_mask"],
            "stock_id": torch.tensor(stock_id, dtype=torch.long),
            "price": torch.tensor(price_vec, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float),
        }



# =========================
# Model
# =========================

class NewsPriceModel(nn.Module):
    """
    - 하루 t에 대해 K개의 뉴스 텍스트를 받아서 (각각 BERT→proj),
      평균을 내어 day-level news embedding z_news[t]를 만든다.
    - 종목 embedding z_stock[s], price window 시퀀스 z_price_seq[t,s]를 LSTM에 넣어서
      최종 price 임베딩을 얻고, 셋을 concat하여 다음날 상승/하락 logit을 예측.
    - contrastive: 같은 날 뉴스 set에 대해 view1 vs view2 임베딩(z1, z2)에 NT-Xent 적용.
    """
    def __init__(
        self,
        num_stocks: int,
        base_model_name: str = "bert-base-uncased",
        news_proj_dim: int = 128,
        price_dim: int = 16,       # = window * F_price
        window: int = 10,
        price_hidden: int = 32,    # per-step price hidden dim
        lstm_hidden: int = 32,     # LSTM hidden size
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        fusion_hidden: int = 128,
        pretrained_ckpt: str | None = None,
    ):
        super().__init__()

        # ===== BERT (news encoder) =====
        self.bert = AutoModel.from_pretrained(base_model_name)
        hid = self.bert.config.hidden_size

        self.news_proj = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, news_proj_dim),
        )

        # ===== Stock embedding =====
        self.stock_emb = nn.Embedding(num_stocks, news_proj_dim)

        # ===== Price sequence encoder (MLP + LSTM) =====
        self.window = window
        assert price_dim % window == 0, \
            f"price_dim ({price_dim}) must be divisible by window ({window})"
        self.f_price = price_dim // window   # feature dim per time step

        # per-time-step price feature MLP
        self.price_mlp = nn.Sequential(
            nn.Linear(self.f_price, price_hidden),
            nn.ReLU(),
        )

        # LSTM over the window sequence
        self.lstm = nn.LSTM(
            input_size=price_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # ===== Classifier (news + stock + price_LSTM) =====
        fusion_in = news_proj_dim + news_proj_dim + lstm_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1),
        )

        if pretrained_ckpt is not None:
            self._load_contrastive_init(pretrained_ckpt)

    def _load_contrastive_init(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Pretrained checkpoint {ckpt_path} not found, skipping load.")
            return
        state = torch.load(ckpt_path, map_location="cpu")
        own = self.state_dict()
        for k in list(state.keys()):
            if k.startswith("bert.") or k.startswith("proj."):
                if k in own:
                    own[k] = state[k]
        self.load_state_dict(own, strict=False)
        print("Loaded contrastive init for bert & proj")

    def encode_news(self, input_ids, attention_mask):
        """
        input_ids: [B, K, L] 또는 [B, L]
        attention_mask: [B, K, L] 또는 [B, L]
        return: [B, d]  (각 sample에 대해 K개 뉴스의 평균 임베딩)
        """
        if input_ids.dim() == 3:
            # [B, K, L]
            B, K, L = input_ids.shape
            flat_ids = input_ids.view(B * K, L)
            flat_mask = attention_mask.view(B * K, L)

            out = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
            cls = out.last_hidden_state[:, 0, :]          # [B*K, hid]
            z = self.news_proj(cls)                       # [B*K, d]
            z = F.normalize(z, dim=-1)
            z = z.view(B, K, -1).mean(dim=1)              # [B, d]  (뉴스 K개 평균)
            return z
        else:
            # [B, L]
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            z = self.news_proj(cls)
            z = F.normalize(z, dim=-1)
            return z

    def encode_price(self, price_feats):
        """
        price_feats: [B, price_dim] where price_dim = window * F_price

        1) reshape -> [B, W, F_price]
        2) per-step MLP -> [B, W, price_hidden]
        3) LSTM over W -> last hidden -> [B, lstm_hidden]
        """
        B, D = price_feats.shape
        assert D == self.window * self.f_price

        price_seq = price_feats.view(B, self.window, self.f_price)     # [B, W, F_price]
        price_h_seq = self.price_mlp(price_seq)                        # [B, W, price_hidden]

        _, (h_n, _) = self.lstm(price_h_seq)                           # h_n: [layers, B, lstm_hidden]
        price_z = h_n[-1]                                              # [B, lstm_hidden] (마지막 layer)
        return price_z

    def stock_embeddings(self):
        """
        Learned stock embedding matrix: [num_stocks, news_proj_dim]
        """
        return self.stock_emb.weight

    def forward(
        self,
        input_ids_1,
        attention_mask_1,
        stock_ids,
        price_feats,
        input_ids_2=None,
        attention_mask_2=None,
    ):
        """
        input_ids_1:    [B, K, L] (view1 for K news)
        attention_mask_1: same
        input_ids_2:    [B, K, L] (view2) or None
        price_feats:    [B, price_dim = window * F_price]
        stock_ids:      [B]
        return:
          logit: [B]
          z1:    [B, d]
          z2:    [B, d] or None
        """
        # 뉴스 임베딩
        news_z1 = self.encode_news(input_ids_1, attention_mask_1)   # [B, d]

        news_z2 = None
        if input_ids_2 is not None and attention_mask_2 is not None:
            news_z2 = self.encode_news(input_ids_2, attention_mask_2)  # [B, d]

        # 종목 임베딩
        stock_z = self.stock_emb(stock_ids)                         # [B, d]

        # 가격 시퀀스 LSTM 임베딩
        price_z = self.encode_price(price_feats)                    # [B, lstm_hidden]

        # 최종 fusion
        x = torch.cat([news_z1, stock_z, price_z], dim=-1)          # [B, 2d + lstm_hidden]
        logit = self.classifier(x).squeeze(-1)                      # [B]

        return logit, news_z1, news_z2

# =========================
# (예시) price 파일 + 뉴스 파일로 df_train/df_valid 만드는 뼈대
# =========================

def build_news_price_df(
    data: str,
    news_df: pd.DataFrame,
    window: int = 10,
):
    """
    data: dataset name (e.g. 'acl18')
    news_df: DB에서 읽은 뉴스 (stock 컬럼 없음)
      필요한 컬럼:
        - 'date' : 뉴스 날짜
        - 'text' : 뉴스 텍스트

    아이디어:
      1) price CSV들에서 (dates, features, labels, mask) 읽고
      2) news_df를 날짜별로 모아서
           daily_news_by_t[date_idx] = [news1, news2, ...] 형태로 저장
      3) 각 날짜 t, 각 종목 s에 대해:
           - 입력:   (t-window+1..t)의 price window + 그 날의 뉴스들 (BERT에서 평균)
           - 라벨:   t+1에서의 label (상승/하락) → 1/0

    Returns:
      df_train, df_valid, df_test: 각 row는 (t, s) 샘플
        - stock_id
        - date_idx
        - label
        - pfeat_0 .. pfeat_{price_dim-1}
      num_stocks     : N_stocks
      price_dim      : window * F_price
      daily_news_by_t: dict[int -> list[str]] (t별 뉴스 리스트)
    """
    # 1) 가격 데이터 읽기
    data_path = os.path.join(utils.ROOT_PATH, "data", data, "price")
    dates, features, labels, mask = utils.read_price_data(data_path)
    # features: (T, N_stocks, F_price)
    # labels:   (T, N_stocks) in {-1,0,1}
    T, N_stocks, F_price = features.shape
    print("Price features shape:", features.shape, "labels shape:", labels.shape)

    # 2) 날짜 인덱스 매핑 (price 기준)
    date_series = pd.to_datetime(dates).dt.normalize()
    date2idx = {d: i for i, d in enumerate(date_series)}

    # 3) 종목 리스트 / ID 매핑
    stock_list = utils.get_stock_list(data)          # ['AAPL.csv', ...]
    stock_list = [os.path.splitext(s)[0] for s in stock_list]
    assert N_stocks == len(stock_list), \
        f"N_stocks ({N_stocks}) != len(stock_list) ({len(stock_list)})"

    # 4) train/valid/test 경계
    trn_date, val_date, test_date, end_date = utils.get_date_info(data)
    trn_idx, val_idx, test_idx, end_idx = utils.get_date_index(
        dates, trn_date, val_date, test_date, end_date, window=window
    )

    # 5) news_df 전처리: 날짜 normalize
    df = news_df.copy()
    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    print("Creating a list of news accroding to dates...")
    # 날짜별 뉴스 리스트 {date (Timestamp) -> [text1, text2, ...]}
    grouped = df.groupby("date")["text"].apply(
        lambda xs: [str(x) for x in xs if isinstance(x, str) and str(x).strip()]
    )
    date_to_news_list = grouped.to_dict()

    # date_idx 기준으로도 매핑 생성: {t(int) -> [text1, text2, ...]}
    daily_news_by_t = {}
    for d, texts in date_to_news_list.items():
        if d in date2idx:
            t = int(date2idx[d])
            daily_news_by_t[t] = texts

    records = []

    for t in range(T):
        # 이 날짜에 뉴스가 없으면 스킵 (원하면 dummy로 넣어도 됨)
        if t not in daily_news_by_t:
            continue

        # window 만큼 과거가 있고, 다음날 라벨도 있어야 함
        if t + 1 >= T or t - (window - 1) < 0:
            continue

        # 이 날짜 t가 train/valid/test 중 어디에 속하는지
        if t < val_idx:
            split = "train"
        elif t < test_idx:
            split = "valid"
        else:
            split = "test"

        for s in range(N_stocks):
            # 현재 날짜(t)에서 해당 종목 s의 price가 유효한지
            if not bool(mask[t, s]):
                continue

            # 다음날 수익률 sign -> {-1, 0, 1}
            y_raw = int(labels[t + 1, s])
            if y_raw == 0:
                continue  # 중립은 버림
            label = 1 if y_raw > 0 else 0

            # window 가격 피쳐: (window, F_price) -> flatten
            price_window = features[t - window + 1 : t + 1, s]  # (window, F_price)
            price_vec = price_window.reshape(-1).numpy().astype("float32")

            rec = {
                "stock_id": s,
                "date_idx": t,
                "label": label,
                "split": split,
            }
            for j in range(price_vec.shape[0]):
                rec[f"pfeat_{j}"] = price_vec[j]

            records.append(rec)

    df_all = pd.DataFrame(records)
    print("Total supervised samples:", len(df_all))

    df_train = df_all[df_all["split"] == "train"].drop(columns=["split"])
    df_valid = df_all[df_all["split"] == "valid"].drop(columns=["split"])
    df_test  = df_all[df_all["split"] == "test"].drop(columns=["split"])

    price_dim = window * F_price
    return df_train, df_valid, df_test, N_stocks, price_dim, daily_news_by_t





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-path", type=str, default="files")
    parser.add_argument("--data", type=str, default="cikm18")
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_len", type=int, default=512)
    
    return parser.parse_args()
# =========================
# Train loop
# =========================

def main():
    args = parse_args()
    data = args.data
    seed = args.seed
    files_path = args.files_path

    fin_sent = load_fin_sent_words(utils.fin_sent_path(files_path))
    fin_syn = load_fin_synonyms(utils.fin_synonyms_path(files_path))

    out_path = utils.default_path(data, seed) if args.out is None else args.out
    save_path = utils.news_emb_model_path(data)
    print("save model path for news", save_path)
    # # 🔴 if model already exists, skip training
    if os.path.exists(save_path):
        print(f"[INFO] Found existing news emb model at {save_path}, skipping pretraining.")
        return
     # if you’re not always running with torchrun, guard DDP init
    if "LOCAL_RANK" in os.environ:  # launched with torchrun
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_main = (rank == 0)
    else:  # normal single-process python
        local_rank = None
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        is_distributed = False
        rank = 0
        world_size = 1
        is_main = True

    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}, distributed={is_distributed}, rank={rank}")
    
    if is_main:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
    trn_date, val_date, test_date, end_date = utils.get_date_info(data)
    print("Reading News Data from Database...")
    chunksize = 200_000
    total_rows = count_rows("fin_news_multisource")
    total_chunks = math.ceil(total_rows / chunksize)
    dfs = []
    for i, chunk in tqdm(enumerate(news_multisource_chunks("fin_news_multisource", trn_date, end_date), start=1),
    total=total_chunks, unit="chunks"):
        dfs.append(chunk)
    
    news_df = pd.concat(dfs, ignore_index=True)
    print(f"Reading News completed. Total fetched row={len(news_df)}")
    # ---- supervised df + 날짜별 뉴스 리스트 만들기 ----
    df_train, df_valid, df_test, num_stocks, price_dim, daily_news_by_t = \
        build_news_price_df(data, news_df, window=args.window)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = NewsPriceDataset(
        df_train,
        daily_news_by_t,
        tokenizer,
        max_len=args.max_len,
        use_augment=True,
        max_news_per_day=MAX_NEWS_PER_DAY,
    )
    valid_ds = NewsPriceDataset(
        df_valid,
        daily_news_by_t,
        tokenizer,
        max_len=args.max_len,
        use_augment=False,
        max_news_per_day=MAX_NEWS_PER_DAY,
    )
    test_ds = NewsPriceDataset(
        df_test,
        daily_news_by_t,
        tokenizer,
        max_len=args.max_len,
        use_augment=False,              # eval이니까 augmentation 끔
        max_news_per_day=MAX_NEWS_PER_DAY,
    )
    # ---- Sampler 설정 ----
    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=False
        )
        # validation은 shard 안 나누고 전체를 각 rank가 돌게 (간단한 방식)
        valid_sampler = None
    else:
        train_sampler = None
        valid_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=4,
        pin_memory=True,
    )
    # ---- Model ----
    # (선택) contrastive-bert.pt 있으면 로드, 없으면 None
    pretrained_ckpt = "contrastive-bert.pt"
    if not os.path.exists(pretrained_ckpt):
        pretrained_ckpt = None

    model = NewsPriceModel(
        num_stocks=num_stocks,
        base_model_name=BASE_MODEL,
        news_proj_dim=128,
        price_dim=price_dim,          # = window * F_price
        window=args.window,           # <= 꼭 넣어줘야 함
        price_hidden=32,
        lstm_hidden=32,
        lstm_layers=1,
        lstm_dropout=0.0,
        fusion_hidden=128,
        pretrained_ckpt=pretrained_ckpt,
).to(device)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce_criterion = nn.BCEWithLogitsLoss()
    lambda_contrast = 0.1

    # ---- Train loop ----
    for epoch in range(args.epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch} (rank {rank})",
            disable=not is_main,
        )

        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            logits, z1, z2 = model(
                batch["input_ids_1"],      # [B, K, L]
                batch["attention_mask_1"],
                batch["stock_id"],         # [B]
                batch["price"],            # [B, price_dim]
                input_ids_2=batch["input_ids_2"],
                attention_mask_2=batch["attention_mask_2"],
            )

            bce_loss = bce_criterion(logits, batch["label"])
            contrast_loss = nt_xent(z1, z2, temperature=0.07)
            loss = bce_loss + lambda_contrast * contrast_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * logits.size(0)

        if is_main:
            avg_loss = total_loss / len(train_ds)
            print(f"Epoch {epoch} train loss: {avg_loss:.4f}")

            # ===== Validation (main rank만 출력) =====
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(valid_loader, total=len(valid_loader), desc="Validation"):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    logits, _, _ = model(
                        batch["input_ids_1"],
                        batch["attention_mask_1"],
                        batch["stock_id"],
                        batch["price"],
                        input_ids_2=batch["input_ids_2"],
                        attention_mask_2=batch["attention_mask_2"],
                    )
                    all_logits.append(logits.cpu())
                    all_labels.append(batch["label"].cpu())
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            preds = (all_logits > 0).int()
            acc = (preds == all_labels.int()).float().mean().item()
            print(f"Epoch {epoch} valid acc: {acc:.4f}")

        if is_main:
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    # test에서는 contrastive view2 안 써도 됨 → input_ids_2=None
                    logits, _, _ = model(
                        batch["input_ids_1"],
                        batch["attention_mask_1"],
                        batch["stock_id"],
                        batch["price"],
                        input_ids_2=batch["input_ids_2"],
                        attention_mask_2=batch["attention_mask_2"],
                    )
                    all_logits.append(logits.cpu())
                    all_labels.append(batch["label"].cpu())

            all_logits = torch.cat(all_logits)   # [N]
            all_labels = torch.cat(all_labels)   # [N]
            preds = (all_logits > 0).int()
            test_acc = (preds == all_labels.int()).float().mean().item()
            print(f"Test acc: {test_acc:.4f}")


        if is_distributed:
            dist.barrier()

    # ---- 저장 ----
    if is_main:
        if isinstance(model, (nn.DataParallel, DDP)):
            to_save = model.module.state_dict()
        else:
            to_save = model.state_dict()
        torch.save(to_save, save_path)
        print("Saved supervised news+stock model to:", save_path)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()











#python src/news_driven_pretrain.py --data acl18 --gpu 0 --seed 0 --out src/out/acl18-0 --window 10













# 1. 같이 하면 진짜 도움이 되냐? 어떻게?
# (1) 왜 도움이 될 수 있는지

# 지금 supervised loss(BCE)만 쓰면 모델은:

# “이 뉴스 → 이 종목의 내일 label(상승/하락)”만 맞추도록 학습

# 뉴스를 어떻게 embedding 하든 loss만 낮으면 장땡이라
# 표현 공간이 깔끔하게 구조화된다는 보장은 없음

# 여기에 contrastive(자기지도) loss를 함께 걸면:

# **같은 뉴스(augment된 두 view)**는 서로 가깝게

# 다른 뉴스들은 멀어지게

# → 임베딩 공간에 군집 구조 / 의미 구조가 생김

# 동시에 BCE가 “그 중에서 어느 방향이 상승/하락과 align 되는지”를 가르침

# 즉:

# contrastive → representation quality & robustness

# BCE → 그 표현을 ‘수익 방향’에 맞게 회전시키는 지도 신호

# 둘을 같이 쓰면

# 라벨이 조금 noisy해도 덜 흔들리고

# augmentation 노이즈에 대해 더 robust해지고

# 뉴스 텍스트 사이의 semantic 관계와
# **“어떤 종류의 뉴스가 실제 price up/down과 align되는지”**를 함께 배우게 됩니다.

# (2) “positive 뉴스 vs price up”을 더 직접적으로 align 하고 싶다면?

# 지금 구조는:

# contrastive: 같은 뉴스의 두 뷰(z1, z2)를 가깝게 (라벨 안 씀)

# supervised(BCE): z1 + stock_emb + price_feat → 상승/하락

# 이렇게만 해도 충분히 **“가격이 실제로 오른 케이스에서 관찰된 뉴스 패턴”**이
# z 공간에 녹아들어요 (BCE가 그 방향으로 gradient를 계속 밀기 때문에).

# 더 적극적으로 가려면 나중에:

# supervised contrastive도 고려 가능

# 같은 stock + 같은 label → positive pair

# 다른 label → negative pair

# 근데 구현이 확 튀어나가서 일단은
# **“news self-contrastive + supervised BCE 동시학습”**을 먼저 안정적으로 돌려보는 걸 추천드립니다.

# 2. 지금 코드에 “joint contrastive + supervised” 넣어준 버전

# 질문 주신 코드에 맞춰서:

# NewsPriceDataset:
# → 한 뉴스에서 두 개의 augmented view를 뽑아서 토크나이즈

# NewsPriceModel:
# → view1으로는 price movement 예측 (BCE)
# → view1, view2 둘 다 임베딩 뽑아서 nt_xent contrastive loss 계산

# train loop:
# → loss = bce_loss + lambda_contrast * contrast_loss