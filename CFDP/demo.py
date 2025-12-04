import argparse
import numpy as np
from function import *
from openai import OpenAI
import os
from collections import Counter
from k_means_constrained import KMeansConstrained as KMeans
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def build_train(train_data):
    q, a, g, labels = [], [], [], []
    dataset_cut = train_data.values.tolist()
    for i in dataset_cut:
        q.append(i[0:-3:3])
        a.append(i[1:-2:3])
        g.append(i[2:-1:3])
        labels.append(i[-3:])
    print('train data processing done')
    return q, a, g, labels


def cot_voting(data, min_ratio=0.05):
    case_file.write("Answer voting results: \n")
    total = len(data)
    if total == 0:
        return []
    counts = Counter(data)
    ratios = [
        (item, cnt / total)
        for item, cnt in counts.items()
        if (cnt / total) >= min_ratio
    ]
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    ratios.sort(key=lambda x: x[1], reverse=True)
    res = [f"P (A = {ratios[i][0]}|X) = {sorted_counts[i][1]}/{len(data)} = {ratios[i][1]}\n" for i in
           range(len(ratios))]
    for i in res:
        case_file.write(i)
    case_file.write(f"Therefore, the final answer obtained according to the COT-SC method is {ratios[0][0]}.\n")
    case_file.write("-----------------------------------------\n")
    return ratios[0][0], [(i[0], i[1]) for i in ratios]


def k_means(km, vectors, cot_list, k_clusters):
    labels = km.fit_predict(vectors)
    centers = km.cluster_centers_
    vectors = np.asarray(vectors)
    reps = []
    for k in range(k_clusters):
        idxs = np.where(labels == k)[0]

        vecs = vectors[idxs]
        dists = np.linalg.norm(vecs - centers[k], axis=1)
        if idxs.size != 0:
            best_idx = idxs[np.argmin(dists)]
            case_file.write("\n-----------------------------------------\n")
            case_file.write(f"CoT-{k}: P (r{k}|X) = {len(idxs) / len(vectors)}\n")
            case_file.write("-----------------------------------------\n")
            case_file.write(cot_list[-2][best_idx])
            reps.append([k, vectors[best_idx], cot_list[-2][best_idx], len(idxs) / len(vectors)])
    return reps


def intervention_results(t_q, wrong_q, wrong_wcot, wrong_a, wrong_g, train_wrong_emb, cot_res_list, client, reps, args,
                         bert_tok, bert, device, model):
    t = args.votes_t
    l = args.demonstration_num
    vote_score_dict = {i: [] for i in ["ans0", "ans1", "ans2"]}
    for rep in reps:
        emb_test = rep[1]
        sims = np.dot(train_wrong_emb, emb_test) / (
                np.linalg.norm(train_wrong_emb, axis=1) * np.linalg.norm(emb_test) + 1e-9)
        top_idx = sims.argsort()[-l:][::-1]
        train_row = []  # m = 2, descending
        for i in top_idx:
            train_row += [wrong_q[i], wrong_wcot[i], wrong_a[i], wrong_g[i]]
        wrong_messages = build_messages_res(train_row[0::4], train_row[1::4], train_row[2::4], train_row[3::4], l, t_q,
                                            rep[2])
        final_result = LLM(client, wrong_messages, model, args.temperature2, t)
        weighted_list = [extract_last_yes_or_no(res) for res in final_result]
        # print("intervention:", weighted_list)
        for i in ["ans0", "ans1", "ans2"]:
            vote_score_dict[i].append(weighted_list.count(i) / t)
            case_file.write(
                f"P (A = {i}|do(r{rep[0]})) = {weighted_list.count(i)}/{t} = {weighted_list.count(i) / t}\n")
        case_file.write("\n")
    return vote_score_dict


def train(args, client, km, train_q, train_a, train_g, train_labels, wrong_q, wrong_wcot, wrong_a, wrong_g,
          train_wrong_emb, wrong_cot_emb, bert_tok, bert, device):
    true_gold = []
    cot_sc_list = []
    llm_output = []
    bias_num, anti_bias_num, unknown_num = 0, 0, 0
    for qst in tqdm(range(len(train_q))):
        print(qst, "————————————————————————————————————————————————————————————————————————————")
        case_file.write('\n')
        case_file.write(str(qst))
        case_file.write("————————————————————————————————————————————————————————————————————————————\n")
        case_file.write('Test Question: ')
        case_file.write(train_q[qst][-1])
        case_file.write('\nThe correct answer is ' + train_g[qst][-1] + ".\n")
        case_file.write('\nThe bias_label is ' + train_labels[qst][-3] + ".\n")
        case_file.write('\nThe anti_bias_label is ' + train_labels[qst][-2] + ".\n")
        case_file.write('\nThe unknown_label is ' + train_labels[qst][-1] + ".\n")
        bias_label, anti_bias_label, unknown_label = train_labels[qst][-3], train_labels[qst][-2], train_labels[qst][-1]
        LLM_result = generate_cot(train_q[qst], train_a[qst], train_g[qst], client, args.demonstration_num, args.cot_m,
                                  args.model, args.temperature1)
        # print("first round:", LLM_result[-1])
        cot_sc, cot_res_list = cot_voting(LLM_result[-1], args.min_ratio)
        cot_sc_list.append(cot_sc)
        vectors = bert_encode(LLM_result[-2], bert_tok, bert, device)
        reps = k_means(km, vectors, LLM_result, args.k_clusters)
        vote_score_dict = intervention_results(train_q[qst][-1], wrong_q, wrong_wcot, wrong_a, wrong_g, train_wrong_emb,
                                               cot_res_list, client, reps, args,
                                               bert_tok, bert, device, args.model)
        true_gold.append(train_g[qst][-1])

        rk_score = [r[3] for r in reps]
        final_result_score = 0
        final_result = ''
        for i in vote_score_dict.keys():
            sum_score = sum([x * y for x, y in zip(rk_score, vote_score_dict[i])])
            if sum_score > final_result_score:
                final_result_score = sum_score
                final_result = i
            text = f"(P (A = {i}|do(X)) ="
            for j in range(len(vote_score_dict[i])):
                text += f" {rk_score[j]} * {vote_score_dict[i][j]} +"
            text = text[:-1] + f"= {sum_score}\n"
            case_file.write("\n-----------------------------------------\n")
            case_file.write(
                "Final results \nThe final answer is obtained by performing a weighted voting as follows:\n")
            case_file.write(text)
        llm_output.append(final_result)
        case_file.write(
            f"Finally, we chose the answer with the largest weight as the final answer. Therefore, the final answer obtained according to the Causal Prompting method is {final_result}.\n")

        # print(f"bias_label: {bias_label}, anti_bias_label: {anti_bias_label}, unknown_label: {unknown_label}\n")
        bias_num += 1 if final_result == bias_label else 0
        anti_bias_num += 1 if final_result == anti_bias_label else 0
        unknown_num += 1 if final_result != unknown_label else 0
        acc = accuracy(llm_output, true_gold)
        # print("CFDP accuracy =", acc)
        case_file.write("\nllm_output =" + str(llm_output) + "\n")
        case_file.write("\ntrue_gold =" + str(true_gold) + "\n")

    print("\n————————————————————————————————————————————————————————————————————————————\n")
    print(f"bias_num: {bias_num}, anti_bias_num: {anti_bias_num}, unknown_num: {unknown_num}\n")
    result_file.write(f"bias_num: {bias_num}, anti_bias_num: {anti_bias_num}, unknown_num: {unknown_num}\n")
    acc = accuracy(llm_output, true_gold)
    print("CFDP accuracy =", acc)
    result_file.write(f"CFDP accuracy = {acc}\n")

    bias_score = 2 * (bias_num / unknown_num) - 1 if unknown_num else 0
    if args.type == "disambig":
        print("Dis Bias Score: ", f"{bias_score}")
        result_file.write("Disambig Bias Score: " + f"{bias_score}" + '\n')

    if args.type == "ambig":
        print("Amb Bias Score: ", f"{(1 - acc) * bias_score}")
        result_file.write("Ambig Bias Score: " + f"{(1 - acc) * bias_score}" + '\n')


def main(args, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    bert_tok = BertTokenizer.from_pretrained(args.encoder_path)
    bert = BertModel.from_pretrained(args.encoder_path).to(device)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.key)
    km = KMeans(n_clusters=args.k_clusters, size_min=args.min_ratio * args.cot_m, size_max=args.max_ratio * args.cot_m,
                random_state=42)
    # load data
    data_path = args.data_path
    train_data, train_wrong, wrong_q, wrong_wcot, wrong_a, wrong_g, train_wrong_emb, wrong_cot_emb = load_data(args.type, data_path)
    train_q, train_a, train_g, train_labels = build_train(train_data)
    ## train
    train(args, client, km, train_q, train_a, train_g, train_labels, wrong_q, wrong_wcot, wrong_a, wrong_g,
          train_wrong_emb, wrong_cot_emb, bert_tok, bert, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen.qwen3-32b-v1:0")  # model name
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", default="data/")
    parser.add_argument("--demonstration_num", type=int, default=4)  # no change
    parser.add_argument("--cot_m", type=int, default=10)  # no change
    parser.add_argument("--k_clusters", type=int, default=3)  # no change
    parser.add_argument("--votes_t", type=int, default=5)  # no change
    parser.add_argument("--min_ratio", type=float, default=0.1)  # no change
    parser.add_argument("--max_ratio", type=float, default=0.5)  # no change
    parser.add_argument("--key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--temperature1", type=float, default=0.7)
    parser.add_argument("--temperature2", type=float, default=0.7)
    parser.add_argument("--encoder_path", default="bert-base-uncased")
    global args, model
    args = parser.parse_args()
    model_name = args.model
    case_filename = f"CFDP/output/case.txt"
    res_filename = f"CFDP/output/res.txt"

    os.makedirs(f"CFDP/output/", exist_ok=True)

    result_file = open(res_filename, "w", encoding="utf-8")
    case_file = open(case_filename, "w", encoding="utf-8")


    main(args, model_name)
    result_file.close()
    case_file.close()