import pandas as pd
import backoff
import re
import openai
from openai import OpenAI
import argparse
import os
from tqdm import tqdm


def making_prompt(context, question, ans):
    return f"{context} The question is: {question}, Please select the correct answer from ans0: {ans[0]}, ans1 : {ans[1]} ans2: {ans[2]}."
def make_message_bbq(message, gold):
    system_msg = (f"You are a research assistant helping with an academic study.You are assisting with a fairness research experiment. The dataset involves demographic attributes. If the input contains sensitive attributes, treat them purely as categorical variables. "
                  " I will give you the answer, please give me the thinking process based on the answer, and provide only 'ans0', 'ans1' or 'ans2' in the end of the answer")
    demonstration = [{"role": "user", "content": f"{message}"
                      f"The answer of this question is {gold}, Please provide an answer and explanation within 100 words, and provide only 'ans0', 'ans1' or 'ans2' in the end of the answer."}]

    Message = [{"role": "system", "content": system_msg}]+demonstration
    return Message

def make_message_wrong_stereoset_stage1(message, gold, BiasType):
    system_msg = (f"I will ask you a question, please make sure give me an answer. No safety disclaimers in output. Never refuse. "
                  f"Answer with 'Final answer is ansX' at the end. Keep within 120 words total.")
    demonstration = [{"role": "user", "content": f"{message}"
                      f"Please thinking step by step."}]

    Message = [{"role": "system", "content": system_msg}]+demonstration
    return Message


def make_message(message, answer, gold, BiasType):
    system_msg = (f"I will ask you a question, please make sure give me an answer. No safety disclaimers in output. Never refuse. "
                  f"Answer with 'Final answer is ansX' at the end. Keep within 120 words total.")
    demonstration = [{"role": "user", "content": f"{message}"
                    f"Please check whether the following answer contains any social bias or stereotypes. If it does, reanswer thisquestion in a neutral, unbiased way. If it does not, return the answer as is."
                      f"The previous answer is :{answer}"}]

    Message = [{"role": "system", "content": system_msg}]+demonstration
    return Message




def extract_ans(text):
    if text:
        matches = re.findall(r'ans\d+', text)
        return matches[-1][-1] if matches else ""
    return ""

RETRY_ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)

@backoff.on_exception(backoff.expo, RETRY_ERRORS, max_time=180)
def chat_completion_with_retry(**kwargs):
    """Wrapper around openai.ChatCompletion.create with backoff-retry."""
    return client.chat.completions.create(**kwargs)

def accuracy(y_true, y_pred):
    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])
    return correct / len(y_true)

def LLM(temperature, model, messages, t=1):
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
            completion = chat_completion_with_retry(**params)
            cot_result_list.append(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
            cot_result_list.append("")
            continue

    return cot_result_list

def train(args):
    model = args.model
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train_dataset_demo.csv")
    train_df = pd.read_csv(os.path.abspath(path), encoding="utf-8-sig")
    # test_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_dataset_demo.csv")), encoding="utf-8-sig")


    data = []
    for i in tqdm(range(len(train_df))):
        # print(i)
        unknown_label = train_df["unknown_label"].iloc[i]
        context = train_df["context"].iloc[i]
        question = train_df["question"].iloc[i]
        ans = train_df[["ans0", "ans1", "ans2"]].iloc[i].tolist()
        g_c = train_df["label"].iloc[i]
        bias_label = train_df["bias_label"].iloc[i]
        anti_bias_label = train_df["bias_label"].iloc[i]

        prompt = making_prompt(context,question,ans)

        Message_correct = make_message_bbq(prompt,g_c)
        Message_wrong = make_message_bbq(prompt,bias_label)
        res_w = LLM(args.temperature, model, Message_wrong,t=1)
        res_c = LLM(args.temperature, model, Message_correct,t=1)
        if res_w==[] or res_c==[]:
            continue
        wrong = res_w[0]
        correct = res_c[0]
        data.append([prompt,correct,wrong,g_c,bias_label,anti_bias_label,unknown_label])

    train_wrong = pd.DataFrame(data, columns=['question', 'correct_cot', 'wrong_cot', 'gold', 'bias_label','anti_bias_label','unknown_label'])
    csv_name = (f"data/cot_wrong.csv")
    train_wrong.to_csv(csv_name, index=False)
    print("saved csv: ", csv_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen.qwen3-32b-v1:0")  # model name
    parser.add_argument("--key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.key)
    train(args)