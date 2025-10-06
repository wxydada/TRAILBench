import json

def find_null_predict_indices(file_path):
    indices = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            if data.get("predict") == None or data.get("predict") == "" or data.get("predict") == "'NoneType' object is not subscriptable" :  # 判断 predict 是否为 null
                indices.append(idx)  # 记录下标
            elif isinstance(data.get("predict"),str):
                if "Expecting value" in data.get("predict") or "Okay," in data.get("predict") or "Based on your interaction history" in data.get("predict") or "好的" in data.get("predict") or "Extra data" in data.get("predict") or "Alright" in data.get("predict"):
                    indices.append(idx)
    return indices

# 使用示例
file_path = "predict\generation_result\\all_generated_queries_u1\gpt\generation_r1_new.jsonl"
result = find_null_predict_indices(file_path)
print(result)
