from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import os
from template import get_template
import argparse
import yaml
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

def predict_oss(**kwargs):

    output_dir = kwargs["output_dir"]
    query_name = kwargs["query_name"]
    template_name = kwargs["template"]
    adapter_name_or_path = kwargs["adapter_name_or_path"]
    model_name_or_path = kwargs["model_name_or_path"]
    gpu_memory_utilization = kwargs["gpu_memory_utilization"]

    test_dir = kwargs["test_dir"]

    if not os.path.exists(output_dir+'/'+test_dir+'/'+query_name+ '/' + kwargs["query_type"]):
        os.makedirs(output_dir+'/'+test_dir+'/'+query_name + '/' + kwargs["query_type"])
        
    with open("./query/history_query/"+ kwargs["person_name"] +"/all_generated_queries_" + kwargs["person_name"] + "_" + kwargs["query_type"] + ".json", "r",encoding='utf-8') as f:
        dataset = json.load(f)

    template = get_template(template_name)
    prompts = []
    for data in tqdm(dataset):
        apitype = data["answer"]["apitype"]
        with open("./apis/apis_openai/" + apitype + "_openai.json", "r") as f:
            tool_description = json.load(f)
        tools = []
        for i in tool_description:
            format_description = {"type":"function","function":i["function"]}
            tools.append(format_description)
            
        for querys in data["query"].values():
            for query in querys:
                prompts.append(template.return_prompt(
                    user_interaction_history = data["user_history"], tool_description = tool_description, query = data["time"] + " " + query))

    if adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, adapter_name_or_path)
    else:
        lora_request = None

    llm = LLM(model=model_name_or_path, gpu_memory_utilization=gpu_memory_utilization,tensor_parallel_size=4,enable_lora=(adapter_name_or_path is not None))
    
    stop_words = template.return_stop_words()

    sampling_params = SamplingParams(temperature=0.01,stop=stop_words,max_tokens=128)
    # We turn on tqdm progress bar to verify it's indeed running batch inference
    outputs = llm.generate(prompts,
                        sampling_params=sampling_params,
                        lora_request=lora_request)

    if not os.path.exists(output_dir+'/'+test_dir):
        os.makedirs(output_dir+'/'+test_dir)
    queries_with_answers = [
        (query, data["answer"]) 
        for data in dataset 
        for queries in data["query"].values() 
        for query in queries
    ]

    with open(output_dir +'/'+test_dir+ "/" + query_name + '/' + kwargs["query_type"] +"/generation_" + kwargs["name"] + "_new.jsonl", "a",encoding='utf-8') as f:
        for output, (query, answer) in zip(outputs, queries_with_answers):
            json.dump({"query":query, "label":answer, "predict":output.outputs[0].text}, f,ensure_ascii=False)
            f.write('\n')
            
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default = 'oss')
args = parser.parse_args()
type = args.type

with open('config.yaml', 'r') as file:
    content = file.read()

config = yaml.safe_load(content)

query_type = "gpt"


model_template_pair = {"./pretrainedllm/Arch-Agent-7B":"qwen","./pretrainedllm/Hammer2.1-7b":"hammer","./pretrainedllm/Llama-3.1-8B-Instruct":"llama3","./pretrainedllm/Mistral-7B-Instruct-v0.3":"mistral","./pretrainedllm/Qwen2.5-7B-Instruct":"qwen","./pretrainedllm/ToolACE-2-Llama-3.1-8B":"llama3","./pretrainedllm/watt-tool-8B":"llama3","./pretrainedllm/xLAM-7b-r":"xlam"}

person_name_list = ["u1","u2","u3","u4","u5","u6","u7","u8","u9","u10"]

for person_name in person_name_list:
    for key,value in model_template_pair.items():
        config["model_name_or_path"] = key
        config["template"] = value
        for dir in ["generation_result"]:
            config["test_dir"] = dir
            if "Llama-3.1-8B-Instruct" in config["model_name_or_path"]:
                config["name"] = "llama3"
            elif "Arch-Agent-7B" in config["model_name_or_path"]:
                config["name"] = "arch"
            elif "Qwen2.5" in config["model_name_or_path"]:
                config["name"] = "Qwen2"
            elif "ToolACE" in config["model_name_or_path"]:
                config["name"] = "ToolACE"
            elif "Mistral" in config["model_name_or_path"]:
                config["name"] = "Mistral"
            elif "DeepSeek-R1-Distill-Llama-8B" in config["model_name_or_path"]:
                config["name"] = "deepseek3-Llama"
            elif "DeepSeek-R1-Distill-Qwen-7B" in config["model_name_or_path"]:
                config["name"] = "deepseek3-Qwen"
            elif "Hammer" in config["model_name_or_path"]:
                config["name"] = "hammer"
            elif "watt" in config["model_name_or_path"]:
                config["name"] = "watt"
            elif "xLAM" in config["model_name_or_path"]:
                config["name"] = "xlam"
            config["person_name"] = person_name
            config["query_type"] = query_type
            config["query_name"] = "all_generated_queries_" + person_name 
            predict_oss(**config)
