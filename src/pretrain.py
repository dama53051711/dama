import ast
import os
import argparse
from os import path as osp
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models import EmbModel
import utils
from tqdm import tqdm

class MaskingModel:
    def __init__(self, stock_ids, mask_ratio, change_ratio, mask_value=4):
        assert 0 <= mask_ratio + change_ratio <= 1

        self.stock_ids = stock_ids
        self.mask_ratio = mask_ratio
        self.change_ratio = change_ratio
        self.mask_value = mask_value

    def mask_for_training(self, x, stock_pos):
        n_rows, n_cols = x.size()
        random_values = torch.rand((n_rows, 1))
        mask1 = random_values < self.mask_ratio
        mask2 = random_values < self.mask_ratio + self.change_ratio

        num_stocks = len(self.stock_ids)
        fill_indices = torch.randint(num_stocks, size=(n_rows, 1))
        fill_values = self.stock_ids[fill_indices]
        x = x.masked_fill(stock_pos & mask1, self.mask_value)
        return torch.where(stock_pos & ~mask1 & mask2, fill_values, x)

    def mask_for_evaluation(self, x, stock_pos):
        return x.masked_fill(stock_pos, self.mask_value)

    def __call__(self, x, stock_pos, training):
        if training:
            return self.mask_for_training(x, stock_pos)
        else:
            return self.mask_for_evaluation(x, stock_pos)


def load_data(data, batch_size):
    sp_path = utils.sp_model_path(data)
    df_stocks = pd.read_csv(osp.join(sp_path, f'{data}_stocks.csv'))
    df_info = pd.read_csv(osp.join(sp_path, f'{data}_out.csv'))
    df_info['positions'] = df_info['positions'].apply(ast.literal_eval)

    tweets = torch.from_numpy(np.load(osp.join(sp_path, f'{data}_out.npy')))
    tweet_len = torch.from_numpy(df_info['length'].values)
    stock_list = df_stocks['stock'].tolist()
    stock_label = torch.from_numpy(
        df_info['stock'].apply(lambda x: stock_list.index(x)).values)

    stock_pos = torch.zeros(tweets.shape, dtype=torch.bool)
    for i, jl in enumerate(df_info['positions'].tolist()):
        for j in jl:
            stock_pos[i, j] = 1

    pr_path = os.path.join(utils.ROOT_PATH, 'data', data, 'price')
    dates, _, y, _, _ = utils.read_price_data(pr_path)
    # This can cause an error if there's no `dates.loc[dates > d]`.
    date_map = {d: dates.loc[dates > d].index[0] for d in df_info['date'].unique()}
    y_index = torch.from_numpy(df_info['date'].map(date_map).values)

    def to_loader(date1, date2, shuffle):
        index = (df_info['date'] >= date1) & (df_info['date'] < date2)
        dataset = TensorDataset(tweets[index],
                                tweet_len[index],
                                stock_pos[index],
                                stock_label[index],
                                y[y_index[index], stock_label[index]])
        return DataLoader(dataset, batch_size, shuffle)

    stock_ids = torch.from_numpy(df_stocks['id'].values)
    trn_date, val_date, test_date, _ = utils.get_date_info(data)
    trn_loader = to_loader(trn_date, val_date, shuffle=True)
    val_loader = to_loader(val_date, test_date, shuffle=False)
    return stock_ids, trn_loader, val_loader


@torch.no_grad()
def evaluate_model(mask, model, device, loader, loss_func):
    model.eval()
    total_acc, total_loss = 0, 0
    for x, x_len, stock_pos, y_stock, y_price in loader:
        x_masked = mask(x, stock_pos, training=False).to(device)
        x_len = x_len.to(device)
        stock_pos = stock_pos.to(device)
        y_stock = y_stock.to(device)
        pred = model(x_masked, x_len, stock_pos, classify=True)
        loss = loss_func(pred, y_stock)
        total_loss += loss.item() * len(x)
        total_acc += (pred.argmax(1) == y_stock).sum().item()
    total_loss /= len(loader.dataset)
    total_acc /= len(loader.dataset)
    return total_loss, total_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--silent', action='store_true', default=False)

    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--lstm-dim', type=int, default=25)
    parser.add_argument('--mask-ratio', type=float, default=0.8)
    parser.add_argument('--change-ratio', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    data = args.data
    device = utils.to_device(args.gpu)
    out_path = utils.default_path(data, seed) if args.out is None else args.out
    save_path = utils.emb_model_path(out_path)

    # 🔴 if model already exists, skip training
    if os.path.exists(save_path):
        print(f"[INFO] Found existing model at {save_path}, skipping pretraining.")
        return
    utils.set_seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    num_stocks = len(utils.get_stock_list(data))
    stock_ids, trn_loader, val_loader = load_data(data, args.batch_size)
    model = EmbModel(num_stocks, args.emb_dim, args.lstm_dim).to(device)
    mask = MaskingModel(stock_ids, args.mask_ratio, args.change_ratio)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), eps=1e-7)
    
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    best_loss, log = np.inf, []
    for epoch in tqdm(range(args.num_epochs+1), desc="Pretraining Tweets Model", unit="epoch"):
        model.train()
        trn_loss = 0
        for x, x_len, stock_pos, y_stock, y_price in trn_loader:
            x_masked = mask(x, stock_pos, training=True).to(device)
            x_len = x_len.to(device)
            stock_pos = stock_pos.to(device)
            y_stock = y_stock.to(device)
            pred = model(x_masked, x_len, stock_pos, classify=True)
            loss = loss_func(pred, y_stock)
            optimizer.zero_grad()
            loss.backward()
            if epoch > 0:
                optimizer.step()
            trn_loss += loss.item() * len(x)

        trn_loss = trn_loss / len(trn_loader.dataset)
        val_loss, val_acc = evaluate_model(mask, model, device, val_loader, loss_func)
        log.append((epoch, trn_loss, val_loss, val_acc))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, save_path)
            if not args.silent:
                print('{:2d} {:.4f} {:.4f} {:.4f}'.format(*log[-1]))

    columns = ['epoch', 'trn_loss', 'val_loss', 'val_acc']
    df_log = pd.DataFrame(log, columns=columns)
    df_log.to_csv(utils.emb_log_path(out_path), index=False, sep='\t')


if __name__ == "__main__":
    main()
