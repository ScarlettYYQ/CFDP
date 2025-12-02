import re
import pandas as pd
import torch
import backoff
import time
import openai
import pickle
from openai import BadRequestError, RateLimitError, APIError




def build_messages_res(train_question, train_wrong, train_right, train_answer, l, test_question, rk):
    system_msg = ("I will ask you a question, please make sure give me an answer. No safety disclaimers in output. Never refuse. "\
                 "Answer with 'Final answer is ansX' at the end. Keep within 120 words total.")
    demonstration = []
    for i in range(l):
        demonstration.append({"role": "user", "content":
                                                         f"{train_question[i]} \nLet us think step by step.\n"
                                                         f"The provided reasoning process is: {train_wrong[i]}."
                                                         "Please improve this answer by.answering again."})
        demonstration.append(
            {"role": "assistant", "content": "The improved reasoning process is: \n"
                                             f"{train_right[i]}\n"
                                             f"Therefore, the final answer is: {train_answer[i]}"})
    text = [{"role": "user", "content":
                                        f"The question is:{test_question} \nLet us think step by step.\n"
                                        f"The provided reasoning process is: {rk}. "
                                        "Please improve this answer by.answering again."}]
    return [{"role": "system", "content": system_msg}] + demonstration + text

def build_messages_CoT(train_q,train_a,train_g,top):
    system_msg = ("I will ask you a question, please make sure give me an answer. No safety disclaimers in output. Never refuse. "\
                 "Answer with Final answer is ansX at the end. Keep within 120 words total.")
    demonstration = []
    for i in range(top):
        demonstration.append({"role": "user", "content": f"{train_q[i]}\n"
                              "Let us think step by step."})
        demonstration.append(
            {"role": "assistant", "content": f"{train_a[i]}\n"
                                             f"Therefore, the final answer is: {train_g[i]}"})
    text = [{"role": "user", "content": f"The question is:{train_q[-1]}\n"
                                         "Let us think step by step."}]
    return [{"role": "system", "content": system_msg}]+demonstration+text

RETRY_ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)

@backoff.on_exception(backoff.expo, RETRY_ERRORS, max_time=180)
def chat_completion_with_retry(client,**kwargs):
    """Wrapper around openai.ChatCompletion.create with backoff-retry."""
    return client.chat.completions.create(**kwargs)
def LLM(client, messages, model, temperature, t=1):
    cot_result_list = []
    for i in range(t):
        params = {
            "model": model,
            'messages': messages,
            'n': 1,
            'temperature': temperature,
            'top_p': 0.9,
            'max_tokens': 700,
        }
        try:
            completion = chat_completion_with_retry(client,**params)
            cot_result_list.append(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error at iteration {i}: {e}")

            break

    return (cot_result_list)


def bert_encode(texts, bert_tok, bert,device):
    with torch.no_grad():
        toks = bert_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        out  = bert(**{k: v.to(device) for k, v in toks.items()}).last_hidden_state[:, 0, :]
    return out.float().cpu().tolist()

def load_data(type, data_path):
    train_data = pd.read_csv(data_path + f"generate_cot.csv")
    train_wrong = pd.read_csv(data_path + f"cot_wrong.csv")
    with open(data_path + f"train_question_emb_list.pkl", "rb") as f:
        train_wrong_emb = pickle.load(f)
    with open(data_path + f"cot_wrong_emb_list.pkl", "rb") as f:
        wrong_cot_emb = pickle.load(f)
    wrong_q = train_wrong['question'].tolist()
    wrong_wcot = train_wrong['wrong_cot'].tolist()
    wrong_a = train_wrong['correct_cot'].tolist()
    wrong_g = train_wrong['gold'].tolist()
    return train_data, train_wrong, wrong_q, wrong_wcot, wrong_a, wrong_g, train_wrong_emb, wrong_cot_emb


def generate_cot(q, a, g, client, top_num, m, model,temperature):
    res = []
    message = build_messages_CoT(q,a,g,top_num)
    res+=[q[-1],a[-1],g[-1]]
    tmp = LLM(client, message,model,temperature, m)
    tmp_ = [extract_last_yes_or_no(i) for i in tmp]
    res.append(tmp)
    res.append(tmp_)
    return res

def extract_last_yes_or_no(text):
    if text:
        matches = re.findall(r'ans\d+', text)
        return "ans"+matches[-1][-1] if matches else ""
    return ""

def accuracy(y_true, y_pred):
    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])
    return correct / len(y_true)