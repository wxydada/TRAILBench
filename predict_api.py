from openai import OpenAI
import os
import httpx
import json
from tqdm import tqdm
import argparse
import yaml
import ast
import re
from json import JSONDecodeError


prompt = """

You are a personal and proactive assistant that deeply integrates personal interaction data for personal assistance. You are able to utilize external tools as needed to answer user's questions or help user accomplish tasks based on their preferences. 

The user will give you a query. Based on the user's interaction history, try to solve the query by using the tools and APIs. The tool you choose should fit the user's interaction history or the needs of the user’s query. 

The interaction history is given in the format of
[{{"time": time1,"apitype": apitype1,"toolname": toolname1,"apiname": apiname1,"fulltoolname": apitype1.toolname1.apiname1,"parameters":{{param1_name = param1_value, param2...}}}}...]

You are given a user's interaction history:
{user_interaction_history} 

When answering question or utilizing tools, ensure they reflect the user's habits, priorities, and values. DO NOT ask the user for further information. Use the function call to answer. Please generate complete parameters according to the tool definition. 
"""

def _sanitize_tool_arguments(arg_str: str) -> dict:
    if arg_str is None:
        raise ValueError("Empty tool arguments")

    t = str(arg_str).strip()

    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()

    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and l < r:
        t = t[l:r+1]

    t = re.sub(r",\s*([}\]])", r"\1", t)

    try:
        return json.loads(t)
    except Exception:
        pass

    try:
        obj = ast.literal_eval(t)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    # 第三轮：尝试给未加引号的 key 补引号再 JSON 解析
    t2 = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)",
                r'\1"\2"\3', t)
    try:
        return json.loads(t2)
    except Exception as e:
        raise e


def predict_api(**kwargs):

    output_dir = kwargs["output_dir"]
    query_name = kwargs["query_name"]
    os.environ["OPENAI_API_KEY"] = kwargs["OPENAI_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = kwargs["OPENAI_BASE_URL"]

    test_dir = kwargs["test_dir"]

    if not os.path.exists(output_dir+'/'+test_dir+'/'+query_name+ '/' + kwargs["query_type"]):
        os.makedirs(output_dir+'/'+test_dir+'/'+query_name + '/' + kwargs["query_type"])
        
    httpx_client = httpx.Client(verify=False)
    client = OpenAI(http_client=httpx_client)

    with open("./query/history_query/"+ kwargs["person_name"] +"/all_generated_queries_" + kwargs["person_name"] + "_" + kwargs["query_type"] + ".json", "r",encoding='utf-8') as f:
        dataset = json.load(f)
    s_index = 0
    for data in tqdm(dataset):
        apitype = data["answer"]["apitype"]
        with open("./apis/apis_openai/" + apitype + "_openai.json", "r") as f:
            tool_description = json.load(f)
        tools = []
        # if kwargs['name'] == "gemini2.5":
        #     function_declare = {"function_declarations":[]}
        #     for i in tool_description:
        #         function_declare["function_declarations"].append(i["function"])
        #     tools.append(function_declare)
        # else:
        for i in tool_description:
            format_description = {"type":"function","function":i["function"]}
            tools.append(format_description)
        for querys in data["query"].values():
            for query in querys:
                # if s_index < 20:
                #     s_index = s_index + 1
                #     continue
                # s_index = s_index + 1
                messages = [{"role": "system", "content":prompt.format(user_interaction_history=data["user_history"])},
                        {"role": "user", "content": data["time"] + " " + query}
                        ]
                completion = client.chat.completions.create(
                    model = kwargs["api_model_name"],
                    messages = messages,
                    temperature = 0.01,
                    tools=tools,
                    tool_choice="required",
                    max_tokens = 256
                )
                try:
                    output_list = []
                    tool_calls = completion.choices[0].message.tool_calls
                    arguments_str = tool_calls[0].function.arguments
                    arguments = json.loads(arguments_str)  # 先按正常路径解析
                    full_name = tool_calls[0].function.name  # 如 Taobao_search_goods
                    split_name = full_name.split("_", 1)
                    if len(split_name) != 2:
                        print(f"Unexpected function name format: {full_name}")
                        with open(output_dir +'/'+test_dir+ "/" + query_name + '/' + kwargs["query_type"] + "/generation_" + kwargs["name"] + "_new.jsonl", "a", encoding='utf-8') as f:
                            json.dump({"query": query,
                                    "label": data["answer"],
                                    "predict": "wrong format"}, f, ensure_ascii=False)
                            f.write('\n')
                        continue
                    toolname, functionname = split_name
                    output_list.append((toolname, [{functionname: arguments}]))

                    with open(output_dir +'/'+test_dir+ "/" + query_name + '/' + kwargs["query_type"] +"/generation_" + kwargs["name"] + "_new.jsonl", "a", encoding='utf-8') as f:
                        json.dump({"query": query,
                                "label": data["answer"],
                                "predict": (toolname, [{functionname: arguments}])}, f, ensure_ascii=False)
                        f.write('\n')

                except Exception as e:
                    msg = completion.choices[0].message if 'completion' in locals() and completion and completion.choices else None
                    tool_calls = getattr(msg, "tool_calls", None) if msg else None
                    attempted_recover = False
            
                    if tool_calls and len(tool_calls) > 0:
                        raw_args = getattr(tool_calls[0].function, "arguments", None)
                        full_name = getattr(tool_calls[0].function, "name", None)

                        if raw_args is not None and full_name:
                            try:
                                fixed_args = _sanitize_tool_arguments(raw_args)
                                split_name = full_name.split("_", 1)
                                if len(split_name) == 2:
                                    print(e)
                                    print(msg)
                                    toolname, functionname = split_name
                                    with open(output_dir +'/'+test_dir+ "/" + query_name + '/' + kwargs["query_type"] + "/generation_" + kwargs["name"] + "_new.jsonl", "a", encoding='utf-8') as f:
                                        json.dump({"query": query,
                                                "label": data["answer"],
                                                "predict": (toolname, [{functionname: fixed_args}])}, f, ensure_ascii=False)
                                        f.write('\n')
                                    attempted_recover = True
                                else:
                                    # 名称格式仍异常，交给原逻辑
                                    attempted_recover = False
                            except Exception:
                                attempted_recover = False

                    if not attempted_recover:
                        # 原始降级处理：把 message.content（或错误文本）写入 predict
                        print(e)
                        print(msg)
                        with open(output_dir +'/'+test_dir+ "/" + query_name + '/' + kwargs["query_type"] + "/generation_" + kwargs["name"] + "_new.jsonl", "a", encoding='utf-8') as f:
                            json.dump({"query": query,
                                    "label": data["answer"],
                                    "predict": (msg.content if (msg and msg.content) else str(e))},
                                    f, ensure_ascii=False)
                            f.write('\n')


def upsert_jsonl_by_key(file_path: str, record: dict, key: str):
    """按 record[key] 在 JSONL 中去重更新：命中则替换，否则追加。"""
    val = record.get(key)

    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    replaced = False
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            new_lines.append(line)
            continue

        if obj.get(key) == val and not replaced:
            new_lines.append(json.dumps(record, ensure_ascii=False) + "\n")
            replaced = True
        else:
            new_lines.append(line)
    
    if not replaced:
        new_lines.append(json.dumps(record, ensure_ascii=False) + "\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

                    
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default = 'api')
args = parser.parse_args()
type = args.type

with open('config.yaml', 'r') as file:
    content = file.read()

config = yaml.safe_load(content)
person_name_list = ["u1","u2","u3","u4","u5","u6","u7","u8","u9","u10"]
for api_model_name in ["deepseek-r1"]:
    for person_name in person_name_list:
        query_type = "gpt"
        config["api_model_name"] = api_model_name
        for dir in ["generation_result"]:
            config["test_dir"] = dir
            if config["api_model_name"] == "gpt-4o":
                config["name"] = "4o"
            elif config["api_model_name"] == "deepseek-r1":
                config["name"] = "r1"
            elif config["api_model_name"] == "deepseek-v3":
                config["name"] = "v3"
            config["person_name"] = person_name
            config["query_type"] = query_type
            config["query_name"] = "all_generated_queries_" + person_name  #到24
            predict_api(**config)
