import os
from os import path as osp
from distutils.util import strtobool

import numpy as np
import pandas as pd
import torch

ROOT_PATH = osp.dirname(osp.dirname(osp.realpath(__file__)))
print("ROOT_PATH", ROOT_PATH)

def sp_model_path(data):
    return osp.join(ROOT_PATH, f'out/{data}-sentencepiece')

def news_emb_model_path(data):
    return osp.join(ROOT_PATH, f'out/{data}-newsmodel/pretrain_news_model.pt')

def default_path(data, seed):
    return osp.join(ROOT_PATH, f'out/{data}-results/{seed}')

def emb_model_path(out_path):
    return osp.join(out_path, 'tweet_model.pth')

# def news_emb_model_path(out_path):
#     return osp.join(out_path, 'pretrain_news_model.pt')

def news_csv_path(csv_path):
    return osp.join(ROOT_PATH, 'data', csv_path, 'financial_news.csv')

def fin_sent_path(csv_path):
    return osp.join(ROOT_PATH, 'data', csv_path, 'fin_sent_words.txt')

def fin_synonyms_path(csv_path):
    return osp.join(ROOT_PATH, 'data', csv_path, 'fin_synonyms.json')


def emb_log_path(out_path):
    return osp.join(out_path, 'tweet_log.tsv')


def feature_path(out_path):
    return osp.join(out_path, 'features.pkl')


def pred_model_path(out_path):
    return osp.join(out_path, 'pred_model.pth')


def pred_log_path(out_path):
    return osp.join(out_path, 'pred_log.tsv')

def pred_res_path(out_path):
    return osp.join(out_path, 'pred_result.csv')
    
def get_stock_list(data):
    return sorted(os.listdir(os.path.join(ROOT_PATH, 'data', data, 'tweet')))


def str2bool(x):
    return bool(strtobool(x))


def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def read_price_data(data_path):
    """
    Returns a list of features and labels which are NOT aligned. The i-th label
    corresponds to the (i-1)-th feature vector.
    """

    def to_tensor(data):
        return torch.from_numpy(np.array(data)).transpose(0, 1)

    files = [f for f in sorted(os.listdir(data_path)) if f.endswith('.csv')]
    dates, x_list, y_list, mask_list, price_list = None, [], [], [], []
    for f in files:
        df = pd.read_csv(os.path.join(data_path, f))
        dates = df['date']
        # Features: columns 1 to 11 (indices)
        curr_x = df.iloc[:, 1:12].values.astype(np.float32)
        x_mask = (curr_x != -123321).all(axis=1)
        curr_x[~x_mask, :] = 0
        # Label: column 12
        curr_y = df.iloc[:, 12].values
        # Check if the column exists, otherwise handle it (e.g. use n_adj_close or raise error)
        if df.shape[1] > 13:
            curr_price = df.iloc[:, 13].values.astype(np.float32)
        else:
            # Fallback if column 13 is missing, though your appl.csv example has it.
            # You might need to adjust index depending on your exact csv structure
            curr_price = np.zeros_like(curr_y, dtype=np.float32)
        x_list.append(curr_x)
        y_list.append(curr_y)
        mask_list.append(x_mask)
        price_list.append(curr_price)

    features = to_tensor(x_list)
    labels = to_tensor(y_list).long()  # in {-1, 0, 1}
    mask = to_tensor(mask_list)
    prices = to_tensor(price_list)
    return dates, features, labels, mask, prices


def get_date_info(data):
    if data == 'cikm18':
        trn_date = '2017-01-03'
        val_date = '2017-09-01'
        test_date = '2017-11-01'
        end_date = '2017-12-29'
    elif data == 'acl18':
        trn_date = '2014-01-02'
        val_date = '2015-08-03'
        test_date = '2015-10-01'
        end_date = '2015-12-31'
    elif data == 'icdm22':
        trn_date = '2019-07-05'
        val_date = '2020-03-02'
        test_date = '2020-05-01'
        end_date = '2020-07-02'
    else:
        raise ValueError

    return trn_date, val_date, test_date, end_date


def get_date_index(dates, trn_date, val_date, test_date, end_date, window=30):
    trn_idx = int(dates[dates == trn_date].index[0])
    trn_idx += window - 1
    val_idx = int(dates[dates == val_date].index[0])
    test_idx = int(dates[dates == test_date].index[0])
    end_idx = int(dates[dates == end_date].index[0])
    return trn_idx, val_idx, test_idx, end_idx

def set_deterministic():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def set_seed(seed=42):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False