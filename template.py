from dataclasses import dataclass
from typing import List, Dict
import types

prompt = """
You are a personal and proactive assistant that deeply integrates personal interaction data for personal assistance. You are able to utilize external tools as needed to answer user's questions or help user accomplish tasks based on their preferences. 

The user will give you a query. Based on the user's interaction history, try to solve the query by using the tools and APIs. The tool you choose should fit the user's interaction history or the needs of the user’s query. 

The interaction history is given in the format of
[{{"time": time1,"apitype": apitype1,"toolname": toolname1,"apiname": apiname1,"fulltoolname": apitype1.toolname1.apiname1,"parameters":{{param1_name = param1_value, param2...}}}}...]

You are given a user's interaction history:
{user_interaction_history} 
Here is some tools under the scenario:
{tool_description}

When answering question or utilizing tools, ensure they reflect the user's habits, priorities, and values. DO NOT ask the user for further information. You should respond in the format of {format}. Here is a example: JDWaimai_recommend_restaurant(location = \"home\", is_delivery = true, food = [\"fried chicken\"]) . No other text MUST be included."""

format1 = "function_name(param1_name = param1_value, param2...)"

TASK_INSTRUCTION = """

You are a personal and proactive assistant that deeply integrates personal interaction data for personal assistance. You are able to utilize external tools as needed to answer user's questions or help user accomplish tasks based on their preferences. 

The user will give you a query. Based on the user's interaction history, try to solve the query by using the tools and APIs. The tool you choose should fit the user's interaction history or the needs of the user’s query. 

The interaction history is given in the format of
[{{"time": time1,"apitype": apitype1,"toolname": toolname1,"apiname": apiname1,"fulltoolname": apitype1.toolname1.apiname1,"parameters":{{param1_name = param1_value, param2...}}}}...]

You are given a user's interaction history:
{user_interaction_history} 

When answering question or utilizing tools, ensure they reflect the user's habits, priorities, and values. DO NOT ask the user for further information.
"""
FORMAT_INSTRUCTION = """You should respond in the format of {format}.  Here is a example: JDWaimai_recommend_restaurant(location = \"home\", is_delivery = true, food = [\"fried chicken\"]) . No other text MUST be included."""


@dataclass
class Template:
    system_format: str ="{content}"
    user_format: str ="{content}"
    assistant_format: str ="{content}"
    formatting_format: str = None
    stop_words: List[str] = None

    def return_stop_words(self):
        return self.stop_words
    
    def return_prompt(self, user_interaction_history,tool_description, query):
        return  self.system_format.format(content = prompt.format(user_interaction_history=user_interaction_history,tool_description=tool_description,format=format1))+ self.user_format.format(content = query) + self.assistant_format

TEMPLATES: Dict[str, "Template"] = {}

def register_template(name, system_format, user_format, assistant_format, stop_words, return_prompt=None):
    template_instance = Template(system_format, user_format, assistant_format, stop_words)
    if return_prompt is not None:
        # Bind the function to the instance to make it a method
        template_instance.return_prompt = types.MethodType(return_prompt, template_instance)
    TEMPLATES[name] = template_instance
    
def get_template(name):
    return TEMPLATES[name]


register_template(
    name = "qwen",
    system_format = "<|im_start|>system\n{content}<|im_end|>\n",
    user_format = "<|im_start|>user\n{content}<|im_end|>\n",
    assistant_format = "<|im_start|>assistant",
    stop_words = ["<|im_end|>"]
)


register_template(
    name = "llama3",
    system_format = "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user_format = "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
    assistant_format = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    stop_words = ["<|eot_id|>"]
)

def deepseek3_return_prompt(self, user_interaction_history, tool_description, query):
    return self.system_format + self.user_format.format(content = prompt.format(user_interaction_history=user_interaction_history,tool_description=tool_description,format=format1)) + query + self.assistant_format

register_template(
    name="deepseek3",
    system_format="<｜begin▁of▁sentence｜>",
    user_format="<｜User｜>{content}",
    assistant_format= "<｜Assistant｜><think>\n",
    stop_words=["<｜end▁of▁sentence｜>"],
    return_prompt=deepseek3_return_prompt
)

register_template(
    name="mistral",
    system_format = "{content}\n\n",
    user_format = "[INST] {content}[/INST]",
    assistant_format = " ",
    stop_words = ["eos_token"]
)


def hammer_return_prompt(self, user_interaction_history, tool_description, query):
    content = f"[BEGIN OF TASK INSTRUCTION]\n{TASK_INSTRUCTION.format(user_interaction_history=user_interaction_history)}\n[END OF TASK INSTRUCTION]\n\n"
    content += f"[BEGIN OF AVAILABLE TOOLS]\n{tool_description}\n[END OF AVAILABLE TOOLS]\n\n"
    content += f"[BEGIN OF FORMAT INSTRUCTION]\n{FORMAT_INSTRUCTION.format(format=format1)}\n[END OF FORMAT INSTRUCTION]\n\n"
    user_query = f"<|im_start|>user\n{query}<im_end>\n"
    return self.system_format + self.user_format.format(content=content) + user_query + self.assistant_format

register_template(
    name="hammer",
    system_format="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>",
    assistant_format= "<|im_start|>assistant\n",
    stop_words=["<im_end>"],
    return_prompt=hammer_return_prompt
)

def xlam_return_prompt(self, user_interaction_history, tool_description, query):
    content = f"[BEGIN OF AVAILABLE TOOLS]\n{tool_description}\n[END OF AVAILABLE TOOLS]\n\n"
    content += f"[BEGIN OF FORMAT INSTRUCTION]\n{FORMAT_INSTRUCTION.format(format=format1)}\n[END OF FORMAT INSTRUCTION]\n\n"
    return self.system_format.format(content = TASK_INSTRUCTION.format(user_interaction_history=user_interaction_history)) + content + self.user_format.format(content=query) + self.assistant_format

register_template(
    name="xlam",
    system_format="[BEGIN OF TASK INSTRUCTION]\n{content}\n[END OF TASK INSTRUCTION]\n\n",
    user_format="[BEGIN OF QUERY]\n{content}\n[END OF QUERY]\n\n",
    assistant_format= "[BEGIN OF SOLUTION]\n",
    stop_words=["[END OF SOLUTION]"],
    return_prompt=xlam_return_prompt
)

