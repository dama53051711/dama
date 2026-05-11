import os
from os import path as osp
import pickle
import csv
import numpy as np
import pandas as pd
import argparse

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from models import DualModel, MainModel
from losses import HingeLoss, CrossEntropyLoss, SmoothedHingeLoss
import utils
import seaborn as sns
import matplotlib.pyplot as plt


import torch
import numpy as np

def run_investment_simulation(dates, prices, predictions, labels, output_path):
    """
    Run backtesting simulation.
    dates: list or array of date indices (or actual dates) corresponding to predictions
    prices: array of raw stock prices for each prediction
    predictions: model predictions (0 or 1)
    labels: actual ground truth (0 or 1) - used for perfect oracle comparison
    """
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'day_idx': dates,
        'price': prices,
        'pred': predictions,
        'label': labels
    })
    
    # Sort by day
    df = df.sort_values('day_idx')
    unique_days = df['day_idx'].unique()
    
    # Portfolio Values
    portfolio_val = [1.0]
    market_val = [1.0] # Equal weight buy-and-hold all available stocks
    perfect_val = [1.0]
    lstm_val = [1.0]   # Placeholder if you had LSTM results

    # Iterate through days
    for i in range(len(unique_days) - 1):
        day = unique_days[i]
        next_day = unique_days[i+1]
        
        # Get stocks available today AND tomorrow (to calc return)
        # Note: In this simplified setup, we assume we buy at 'price' on 'day' 
        # and sell at 'price' on 'next_day'. 
        # BUT 'price' in your CSV usually corresponds to the closing price of that day.
        # The label y_t predicts movement from t to t+1. 
        # So Return = (Price[t+1] - Price[t]) / Price[t]
        
        # We need to match stocks across days.
        current_data = df[df['day_idx'] == day].set_index(df.index[df['day_idx'] == day]) 
        # This part is tricky because 'index' in df is not Stock ID.
        # Since x_test is stacked (stock1_t1, stock1_t2... stock2_t1...), 
        # retrieving the exact next price for the SAME stock requires stock_id.
        # However, we lost stock_id in the flat lists.
        # 
        # OPTION A: Accurate way -> Save stock_id in preprocess.py too.
        # OPTION B: Approximate/Batch way -> Assuming the test set is ordered chronologically per stock
        # or ordered by date then stock.
        #
        # Given your code structure, x_test is created by:
        # for stock_idx in range(num_stocks):
        #    for data_idx in range(...):
        #        x_test.append(...)
        # So it is blocked by Stock ID.
        pass

    # --- SIMPLIFIED SIMULATION LOGIC ---
    # Since we know x_test is ordered by Stock, then Time:
    # We can reconstruct returns easily.
    # Return[i] = (Price[i+1] - Price[i]) / Price[i] 
    # IF Day[i+1] == Day[i] + 1 (roughly).
    
    # Let's calculate returns per row first.
    # Shift prices within each stock block to get next day's price.
    # Since we don't explicitly have stock IDs, we detect stock boundaries 
    # using day indices (day index should increase; if it drops, it's a new stock).
    
    df['next_price'] = df['price'].shift(-1)
    df['next_day'] = df['day_idx'].shift(-1)
    
    # Valid trade if next_day > day_idx (monotonic increase)
    # If next_day < day_idx, it means we wrapped to the next stock.
    mask = df['next_day'] > df['day_idx']
    
    # Calculate daily return for each sample
    df.loc[mask, 'return'] = (df['next_price'] - df['price']) / df['price']
    df.loc[~mask, 'return'] = 0.0 # No trade on last day of stock
    
    # Now group by day to simulate portfolio
    daily_groups = df[mask].groupby('day_idx')
    
    p_val = 1.0
    m_val = 1.0
    
    plot_dates = []
    plot_p_vals = []
    plot_m_vals = []
    
    for day, group in daily_groups:
        # Market Return: Average return of ALL stocks available
        mkt_ret = group['return'].mean()
        
        # Strategy Return: Average return of stocks predicted as 1 (Long)
        # If prediction is 0, we assume cash (0 return) or Short (negative return).
        # Common paper setting: Long top K or Long all positives.
        # Let's use Long All Positives.
        
        long_stocks = group[group['pred'] == 1]
        if len(long_stocks) > 0:
            strat_ret = long_stocks['return'].mean()
        else:
            strat_ret = 0.0 # Hold cash
            
        p_val *= (1 + strat_ret)
        m_val *= (1 + mkt_ret)
        
        plot_dates.append(day)
        plot_p_vals.append(p_val)
        plot_m_vals.append(m_val)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(plot_dates, plot_p_vals, label='Proposed (DTML)', linewidth=2)
    plt.plot(plot_dates, plot_m_vals, label='Market Index', linestyle='--', alpha=0.7)
    plt.xlabel('Time (Day Index)')
    plt.ylabel('Portfolio Value')
    plt.title('Investment Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path.replace('.csv', '_simulation.png'))
    plt.close()
    
    print(f"Simulation saved to {output_path.replace('.csv', '_simulation.png')}")
    print(f"Final Portfolio Value: {p_val:.4f} vs Market: {m_val:.4f}")


def to_acc(y_pred, y_true):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    return accuracy_score(y_true_np, y_pred_np)


def to_mcc(y_pred, y_true):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    return matthews_corrcoef(y_true_np, y_pred_np)


def to_loss_func(name):
    if name == 'hinge':
        return HingeLoss()
    elif name in ('smoothed-hinge', 'sh'):
        return SmoothedHingeLoss()
    elif name in ('cross-entropy', 'ce'):
        return CrossEntropyLoss()
    else:
        raise ValueError(name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='base')

    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--l2-norm', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--loss', type=str, default='hinge')
    parser.add_argument('--patience', type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    data = args.data
    seed = args.seed
    device = utils.to_device(args.gpu)
    out_path = utils.default_path(data, seed) if args.out is None else args.out
    utils.set_seed(seed)
    data_path = os.path.join(utils.ROOT_PATH, 'data', data, 'price')
    dates, _, _, _, _ = utils.read_price_data(data_path)
    dates = dates.reset_index(drop=True)
    (x_train, y_train, train_day_idx, p_train, 
        x_valid, y_valid, valid_day_idx, p_valid,
        x_test,  y_test,  test_day_idx, p_test,
        news_seq_tensor, news_mask_tensor,
        tweet_seq_tensor, tweet_mask_tensor,
    ) = pickle.load(open(utils.feature_path(out_path), 'rb'))

    # day index는 LongTensor라고 가정 (혹시 아니면 Long으로 캐스팅)
    if not torch.is_tensor(train_day_idx):
        train_day_idx = torch.tensor(train_day_idx, dtype=torch.long)
        valid_day_idx = torch.tensor(valid_day_idx, dtype=torch.long)
        test_day_idx  = torch.tensor(test_day_idx,  dtype=torch.long)

    # === split별로 news/tweet 시퀀스를 미리 잘라놓기 (N_sample 기준) ===
    # shape:
    #   news_seq_tensor: (num_dates, max_news,  d_text)
    #   tweet_seq_tensor: (num_dates, max_tweet, d_text)
    train_news_seq   = news_seq_tensor[train_day_idx]     # (N_tr, max_news, d_text)
    train_news_mask  = news_mask_tensor[train_day_idx]    # (N_tr, max_news)
    train_tweet_seq  = tweet_seq_tensor[train_day_idx]    # (N_tr, max_tweet, d_text)
    train_tweet_mask = tweet_mask_tensor[train_day_idx]   # (N_tr, max_tweet)

    valid_news_seq   = news_seq_tensor[valid_day_idx]
    valid_news_mask  = news_mask_tensor[valid_day_idx]
    valid_tweet_seq  = tweet_seq_tensor[valid_day_idx]
    valid_tweet_mask = tweet_mask_tensor[valid_day_idx]

    test_news_seq    = news_seq_tensor[test_day_idx]
    test_news_mask   = news_mask_tensor[test_day_idx]
    test_tweet_seq   = tweet_seq_tensor[test_day_idx]
    test_tweet_mask  = tweet_mask_tensor[test_day_idx]
    
    # === Train DataLoader: x, y, news_seq, news_mask, tweet_seq, tweet_mask ===
    trn_dataset = TensorDataset(
        x_train, y_train,
        train_news_seq, train_news_mask,
        train_tweet_seq, train_tweet_mask,
    )
    trn_loader = DataLoader(trn_dataset, args.batch_size, shuffle=False)

    x_valid         = x_valid.to(device)
    y_valid         = y_valid.to(device)
    valid_news_seq  = valid_news_seq.to(device)
    valid_news_mask = valid_news_mask.to(device)
    valid_tweet_seq = valid_tweet_seq.to(device)
    valid_tweet_mask = valid_tweet_mask.to(device)

    x_test          = x_test.to(device)
    y_test_torch    = y_test.to(device)   # 이름만 바꿔서 헷갈리지 않게
    test_news_seq   = test_news_seq.to(device)
    test_news_mask  = test_news_mask.to(device)
    test_tweet_seq  = test_tweet_seq.to(device)
    test_tweet_mask = test_tweet_mask.to(device)

    num_features = x_train.size(2)
    print("num_features", num_features)
    if args.mode == 'base':
        d_text = news_seq_tensor.size(-1)
        model = MainModel(num_features, args.hidden_dim, d_text, gate_usage='post', gate_scalar=True, add_div=True,   # 필요시 축소/확장 가능
    div_proj_dim=16)
    elif args.mode.startswith('dual'):
        model = DualModel(num_features, args.hidden_dim)
    else:
        raise ValueError(args.mode)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-07)
    loss_func = to_loss_func(args.loss)

    model_path = utils.pred_model_path(out_path)
    os.makedirs(osp.dirname(model_path), exist_ok=True)

    best_acc, log = 0, []
    epochs_no_improve = 0 
    for epoch in range(args.num_epochs + 1):
        train_loss = 0
        model.train()
        for (batch_x,
             batch_y,
             batch_news_seq,
             batch_news_mask,
             batch_tweet_seq,
             batch_tweet_mask) in trn_loader:
            batch_x          = batch_x.to(device)
            batch_y          = batch_y.to(device)
            batch_news_seq   = batch_news_seq.to(device)
            batch_news_mask  = batch_news_mask.to(device)
            batch_tweet_seq  = batch_tweet_seq.to(device)
            batch_tweet_mask = batch_tweet_mask.to(device)
            out = model(
                batch_x,
                batch_news_seq,
                batch_news_mask,
                batch_tweet_seq,
                batch_tweet_mask,
            )
            # print(model.last_gate_values)
            loss = loss_func(out, batch_y) + args.l2_norm * model.l2_norm()
            optimizer.zero_grad()
            loss.backward()
            if epoch > 0:
                optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(trn_loader.dataset)

        model.eval()
        with torch.no_grad():
            
            out_valid = model(
                x_valid,
                valid_news_seq,
                valid_news_mask,
                valid_tweet_seq,
                valid_tweet_mask,
            )
            out_test = model(
                x_test,
                test_news_seq,
                test_news_mask,
                test_tweet_seq,
                test_tweet_mask,
            )

        y_pred_valid = (out_valid > 0).int()

        y_pred_test  = (out_test  > 0).int()

        val_acc = to_acc(y_pred_valid, y_valid)
        val_mcc = to_mcc(y_pred_valid, y_valid)
        test_acc = to_acc(y_pred_test, y_test)
        test_mcc = to_mcc(y_pred_test, y_test)
        log.append((epoch, train_loss, val_acc, val_mcc, test_acc, test_mcc))

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0 
            torch.save(model, model_path)
            if not args.silent:
                print('{:3d} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}'.format(*log[-1]))
        else:
            epochs_no_improve += 1            # 👈 개선 없으면 +1
            if not args.silent:
                print(f'Epoch {epoch}: no improvement ({epochs_no_improve}/{args.patience})')

        if epochs_no_improve >= args.patience:
            if not args.silent:
                print(f'Early stopping triggered at epoch {epoch}')
            break
    columns = ['epoch', 'trn_loss', 'val_acc', 'val_mcc', 'test_acc', 'test_mcc']
    df_log = pd.DataFrame(log, columns=columns)
    df_log.to_csv(utils.pred_log_path(out_path), index=False, sep='\t')
    ####
    nb_classes = 2

    # with torch.no_grad():
    #     for i, (inputs, classes) in enumerate(trn_loader):
    #         inputs = inputs.to(device)
    #         classes = classes.to(device)
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #         for t, p in zip(classes.view(-1), preds.view(-1)):
    #             confusion_matrix[t.long(), p.long()] += 1
    model.eval()
    
    with torch.no_grad():
        out_test = model(
            x_test,
            test_news_seq,
            test_news_mask,
            test_tweet_seq,
            test_tweet_mask,
        )
        y_pred_test = (out_test > 0).int()

    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred_test.cpu().numpy()
    test_day_idx_np = test_day_idx.cpu().numpy()
    p_test_np = p_test.cpu().numpy()
    day_idx_np = test_day_idx.numpy()
    with open(utils.pred_res_path(out_path), "w") as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['index', 'Prediction', 'Label', 'Hit', 'DayIdx', 'Price'])
        for i in range(len(y_test_np)):
            right = 1 if y_test_np[i] == y_pred_np[i] else 0
            # Lookup Real Date string
            day_idx = day_idx_np[i]
            date_str = dates[day_idx]
            
            csvwriter.writerow([
                i, 
                y_pred_np[i], 
                y_test_np[i], 
                right, 
                date_str, 
                p_test_np[i]
            ])
         
    # 2. RUN SIMULATION
    print("Running Investment Simulation...")
    # run_investment_simulation(
    #     dates=test_day_idx_np,
    #     prices=p_test_np,
    #     predictions=y_pred_np,
    #     labels=y_test_np,
    #     output_path=utils.pred_res_path(out_path)
    # )
    print(confusion_matrix(y_pred_np, y_test_np))
    cf_matrix = confusion_matrix(y_pred_np, y_test_np)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    ####

if __name__ == "__main__":
    main()
