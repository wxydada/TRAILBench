import os
import json
with open("./apis/apis_openai/sport_and_health_openai.json", "r") as f:
    tool_description = json.load(f)
    num_of_tool = len(tool_description)
    all_param = 0
    all_require = 0
    for func in tool_description:
        num_of_param = len(func["function"]["parameters"]["properties"])
        num_of_require = len(func["function"]["parameters"]["required"])
        all_param += num_of_param
        all_require += num_of_require
    print(round(all_param / num_of_tool,4))
    print(round(all_require / num_of_tool,4))
    