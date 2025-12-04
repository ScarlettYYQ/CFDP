import pandas as pd
import time
import random
import re
import os
import openai
import pickle
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm

def making_prompt(context, question, ans):
    return f"{context} The question is: {question}, Please select the correct answer from ans0: {ans[0]}, ans1 : {ans[1]} ans2: {ans[2]}."

def bert_encode(texts, bert_tok, bert,device):
    with torch.no_grad():
        toks = bert_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        out  = bert(**{k: v.to(device) for k, v in toks.items()}).last_hidden_state[:, 0, :]
    return out.float().cpu().tolist()

def train(args):

    test_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_dataset_demo.csv")), encoding="utf-8-sig")
    train_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train_dataset_demo.csv")), encoding="utf-8-sig")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)

    train_wrong = pd.read_csv(f'data/cot_wrong.csv')
    train_q = train_wrong.question.to_list()
    train_c = train_wrong.correct_cot.to_list()
    train_w = train_wrong.wrong_cot.to_list()
    train_gold = train_wrong.gold.to_list()

    print('start question encoding')
    train_emb = bert_encode(train_q,bert_tok,bert,device)
    print("done")
    with open(f"data/train_question_emb_list.pkl", "wb") as f:
        pickle.dump(train_emb, f)
    print('cot_wrong encoding')
    wrong_emb = bert_encode(train_w,bert_tok,bert,device)
    print("done")
    with open(f"data/cot_wrong_emb_list.pkl", "wb") as f:
        pickle.dump(wrong_emb, f)
    test_q, test_gold,test_bias,test_anti_bias,unknown_label=[],[],[],[],[]
    records = []
    top = args.demonstration_num
    for i in tqdm(range(len(test_df))):
        context = test_df["context"].iloc[i]
        question = test_df["question"].iloc[i]
        ans = test_df[["ans0", "ans1", "ans2"]].iloc[i].tolist()
        g_c = test_df["label"].iloc[i]
        bias_label = test_df["bias_label"].iloc[i]
        anti_bias_label = test_df["anti_bias_label"].iloc[i]
        unknown_label = test_df["unknown_label"].iloc[i]
        prompt = making_prompt(context,question,ans)

        test_q.append(prompt)
        test_bias.append(bias_label)
        test_gold.append(g_c)

        emb_test = bert_encode([prompt],bert_tok,bert,device)[0]
        sims = np.dot(train_emb, emb_test) / (
            np.linalg.norm(train_emb, axis=1) * np.linalg.norm(emb_test) + 1e-9)
        top_idx = sims.argsort()[-top:][::-1]
        rows = []
        for i in top_idx:
            rows+=[train_q[i], train_c[i] ,train_gold[i]]
        rows+=[prompt, bias_label, g_c, bias_label, anti_bias_label, unknown_label]
        records.append(rows)

    columns = []
    for i in range(1, top + 1):
        columns += ['train_q' + str(i), 'train_a' + str(i), 'train_gold' + str(i)]
    columns += ['q_test', "a_test", "gold_test", "bias_label", "anti_bias_label", "unknown_label"]
    df = pd.DataFrame(records, columns=columns)
    csv_name = f"data/generate_cot.csv"
    df.to_csv(csv_name, index=False)
    print("saved csv: ", csv_name)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demonstration_num", type=int, default=4)
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--data_name", type=str, default="")
    args = parser.parse_args()
    train(args)
