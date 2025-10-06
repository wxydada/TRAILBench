import csv
import json
import sys
import os
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parse import ast_parse

def include(x,y):
    if x == 'true' or x == 'false':
        return json.loads(x) in y
    elif isinstance(x, bool):
        return x in y
    elif isinstance(x, str):
        z = [i.lower() for i in y]
        return x.lower() in z
    elif isinstance(x, list):
        x = set([i.lower() for i in x])
        z = [set([i.lower() for i in j]) for j in y]
        return set(x) in z
    elif isinstance(x, int):
        z = y[0]
        return x<=z+0.01 and x>=z-0.01
    return x in y

def evaluate(real, predict):

    real_platform, real_funcs = real
    predict_platform, predict_funcs = predict

    overall_acc = 1
    platform_acc = 1
    funcname_acc = 1
    arg_acc = 1
    func_acc = 1

    if real_platform != predict_platform:
        platform_acc = 0
        overall_acc = 0
    

    #if predict in real
    real_func_names = [list(real_func.keys())[0] for real_func in real_funcs]
    for predict_func in predict_funcs:
        predict_func_name = list(predict_func.keys())[0]
        if predict_func_name in real_func_names:
            pos = real_func_names.index(predict_func_name)
            real_param_names = list(real_funcs[pos][predict_func_name].keys())
            for predict_param_name, predict_param_value in list(predict_func.values())[0].items():
                if predict_param_name in real_param_names:
                    real_param_value = real_funcs[pos][predict_func_name][predict_param_name]
                    if not include(predict_param_value, real_param_value):
                        func_acc = 0
                        overall_acc = 0
                else:
                    arg_acc = 0
                    func_acc = 0
                    overall_acc = 0
        else:
            funcname_acc = 0
            arg_acc = 0
            func_acc = 0
            overall_acc = 0



    #if real in predict
    predict_func_names = [list(predict_func.keys())[0] for predict_func in predict_funcs]
    for real_func in real_funcs:
        real_func_name = list(real_func.keys())[0]
        if real_func_name in predict_func_names:
            pos = predict_func_names.index(real_func_name)
            predict_param_names = list(predict_funcs[pos][real_func_name].keys())
            for real_param_name, real_param_value in list(real_func.values())[0].items():
                if real_param_name in predict_param_names:
                    predict_param_value = predict_funcs[pos][real_func_name][real_param_name]
                    if not include(predict_param_value, real_param_value):
                        func_acc = 0
                        overall_acc = 0
                else:
                    arg_acc = 0
                    func_acc = 0
                    overall_acc = 0
        else:
            funcname_acc = 0
            arg_acc = 0
            func_acc = 0
            overall_acc = 0

    return overall_acc, platform_acc, funcname_acc, arg_acc, func_acc


import os, json, csv,re
from collections import defaultdict

def _count_history_events(uhist):
    """
    统计 user_history 中带有 "time" 键的事件数量。
    - uhist 期望是 list；若不是，返回 0
    - 每个元素若是 dict 且包含键 "time" 则计数 +1
    """
    return len(re.findall(r'{\"time\"\s*:', uhist))

def _bucket_label_by_30(n):
    """
    将事件数 n 映射到区间标签，每 30 为一档：
    0-30, 31-60, 61-90, ...
    """
    if n <= 30:
        return "0-30"
    lower = ((n - 1) // 30) * 30 + 1   # 31, 61, 91, ...
    upper = lower + 30 - 1             # 60, 90, 120, ...
    if lower == 151:
        lower = 121
        upper = 180
    if upper == 150:
        upper = 180
    return f"{lower}-{upper}"

def get_history_bucket_index_map(query_json_path, bin_size_ignored=30):
    with open(query_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bucket_index_map = defaultdict(list)
    current_index = 0
    for i in range(len(data)):
        # 计算该 position 的交互历史长度（含 "time" 的计数）
        uhist = data[i].get("user_history", [])
        hist_len = _count_history_events(uhist)
        bucket = _bucket_label_by_30(hist_len)

        # 按原逻辑展开 sentences 以对齐 predict_all 的扁平索引
        query = data[i].get("query", {})
        for _difficulty, sentences in (query.items() if isinstance(query, dict) else []):
            for _ in sentences:
                bucket_index_map[bucket].append(current_index)
                current_index += 1

    return bucket_index_map

def cal_acc_all_avg(test_type, query_name_list, output_dir, model_name, query_type, answertype="api"):
    total_format = defaultdict(float)
    total_platforms = defaultdict(float)
    total_funcnames = defaultdict(float)
    total_args = defaultdict(float)
    total_funcs = defaultdict(float)
    total_overalls = defaultdict(float)
    total_counts = defaultdict(int)

    for query_name in query_name_list:
        person_name = query_name.split("_")[-1]
        with open(os.path.join(output_dir, "generation_result", query_name, query_type, f"generation_{model_name}_new.jsonl"), "r", encoding="utf-8") as f:
            predict_all = f.readlines()

        # ★ 改为按“交互历史长度区间”分组
        index = get_history_bucket_index_map(f"query/history_query/{person_name}/{query_name}_gpt.json")

        predict = [json.loads(i).get("predict") for i in predict_all]

        for key in index.keys():
            format_acc = 0
            overalls_acc = 0
            platforms_acc = 0
            funcnames_acc = 0
            args_acc = 0
            funcs_acc = 0

            for i in index[key]:
                real = json.loads(predict_all[i])["label"]
                real_answer = [real["toolname"], [{real["apiname"]: {k: [v] for k, v in real["parameters"].items()}}]]
                try:
                    if answertype == "api":
                        predict_answer = predict[i]
                    else:
                        predict_answer = ast_parse(predict[i])  # 假定外部已提供
                    overall_acc, platform_acc, funcname_acc, arg_acc, func_acc = evaluate(real_answer, predict_answer)  # 假定外部已提供
                    format_acc += 1
                    overalls_acc += overall_acc
                    platforms_acc += platform_acc
                    funcnames_acc += funcname_acc
                    args_acc += arg_acc
                    funcs_acc += func_acc
                except Exception:
                    continue

            # 分组平均
            denom = max(len(index[key]), 1)
            total_format[key]    += format_acc   / denom
            total_platforms[key] += platforms_acc/ denom
            total_funcnames[key] += funcnames_acc/ denom
            total_args[key]      += args_acc     / denom
            total_funcs[key]     += funcs_acc    / denom
            total_overalls[key]  += overalls_acc / denom
            total_counts[key]    += 1

    # 写入不同“交互历史长度区间”的结果 CSV
    for dijige,key in enumerate(total_counts):
        if dijige < 4:
            continue
        count = max(total_counts[key], 1)
        out_dir = os.path.join(output_dir, test_type, "all")
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"average_acc_results_{key.replace(' ', '_')}.csv")

        file_exists = os.path.isfile(file_path)
        with open(file_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["name","format","platforms","tool_names","tool_param_names","tool_param_values","overall"]
                writer.writerow(header)
            writer.writerow([
                model_name,
                round(total_format[key]    / count, 4),
                round(total_platforms[key] / count, 4),
                round(total_funcnames[key] / count, 4),
                round(total_args[key]      / count, 4),
                round(total_funcs[key]     / count, 4),
                round(total_overalls[key]  / count, 4)
            ])

    
person_name_list = ["u1","u2","u3","u4","u5","u6","u7","u8","u9","u10"]
query_type = "gpt"   

query_name_list = ["all_generated_queries_"+person_name for person_name in person_name_list]
for model_name in ["ToolACE","hammer","llama3","Mistral","Qwen2","watt","xlam","arch"]:
    print(cal_acc_all_avg("results_acc",query_name_list,"predict",model_name=model_name,query_type=query_type,answertype="oss"))
for model_name in ["4o","r1","v3"]:
    print(cal_acc_all_avg("results_acc",query_name_list,"predict",model_name=model_name,query_type=query_type,answertype="api"))