import os
from os import path as osp
import json 
import numpy as np
import sentencepiece as spm 
import pandas as pd 
import argparse
import utils
from tqdm import tqdm

def html_decode(text):
    html_codes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;')
    )
    for code in html_codes:
        text = text.replace(code[1], code[0])
    return text

def train_sp_model(df_tweets, prefix_path, vocab_size):
    os.makedirs(osp.dirname(prefix_path), exist_ok= True)
    input_path = f'{prefix_path}_input.txt'
    with open(input_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(df_tweets['tweet'].tolist()))
    user_defined_symbols = '<s>,</s>,<pad>,<mask>,rt,AT_USER,URL'   
    stock_list = sorted(df_tweets['stock'].unique())
    for stock in stock_list:
        user_defined_symbols+= ',$' + stock.lower()

    command = f'--input={input_path} ' \
              f'--model_prefix={prefix_path} ' \
              f'--vocab_size={vocab_size} ' \
              f'--user_defined_symbols={user_defined_symbols}'
    spm.SentencePieceTrainer.train(command)      

def read_tweets(data, stock_list, start_date, end_date):
    out = [] 
    #Iterate over all the sotcks
    for stock in tqdm(stock_list, desc="Fetching tweets", unit="stock"):
        #get the stock folder under the tweets folder
        stock_path = os.path.join(utils.ROOT_PATH, 'data', data, 'tweet', stock)
        for date in sorted(os.listdir(stock_path)):
            date_path = os.path.join(stock_path, date)
            if not start_date<=date<end_date:
                continue
            with open(date_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
            for line in lines:
                tweet = json.loads(line, strict=False)["text"]
                tweet = html_decode(tweet)
                out.append((stock, date, tweet))
    return pd.DataFrame(out, columns=['stock', 'date', 'tweet'])

def tokenize_tweets(sp_model, df_tweets, padding=3):
    index, positions, out = [],[],[]
    for i, (stock, _, tweet) in df_tweets.iterrows():
        stock_id = sp_model.piece_to_id('$'+ stock.lower())
        id_list = sp_model.encode_as_ids(tweet)
        pos = [i for i, x in enumerate(id_list) if x==stock_id]
        if pos:
            index.append(i)
            positions.append(pos)
            out.append(id_list)
    info = df_tweets.loc[index, ['stock', 'date']].reset_index(drop=True)
    info["index"] = index
    info["length"] = [len(e) for e in out]
    info['positions'] = positions
    max_len = max(info["length"])
    out = np.array([e + [padding] * (max_len-len(e)) for e in out], dtype= np.int64)
    return info, out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='acl18')
    parser.add_argument('--vocab-size', type=int, default=16000)
    return parser.parse_args()

def main():
    #arguments parsing
    args = parse_args()
    #get the data from argument
    data = args.data
    out_path = utils.sp_model_path(data)
    prefix_path = osp.join(out_path, data)
    model_path = f'{prefix_path}.model'
    #Get the stock list 
    stock_list = utils.get_stock_list(data)
    trn_date, val_date, _, end_date = utils.get_date_info(data)
    df_all = read_tweets(data, stock_list, trn_date, end_date)
    df_trn = df_all[(df_all["date"]>=trn_date) & (df_all["date"]<val_date)]
    
    print("-------------df_trn--------------")
    print(df_trn)
    print("df_all----------------------")
    print(df_all)
    if not osp.exists(model_path):
        train_sp_model(df_trn, prefix_path, args.vocab_size)
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(model_path)    
    df_info, tokens = tokenize_tweets(sp_model, df_all)
    print("df_info>>>",df_info)
    df_info.to_csv(osp.join(out_path, f'{data}_out.csv'), index=False)
    np.save(osp.join(out_path, f'{data}_out'), tokens)

    stock_ids = sp_model.piece_to_id([f'${s.lower()}' for s in stock_list])
    print("stock_ids", stock_ids)
    df = pd.DataFrame(list(zip(stock_list, stock_ids)), columns=['stock', 'id'])
    df.to_csv(osp.join(out_path, f'{data}_stocks.csv'), index=False)

if __name__== "__main__":
    main()