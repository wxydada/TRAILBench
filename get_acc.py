import csv
import json
from parse import ast_parse
import os

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

def cal_acc(test_type, query_name,output_dir, model_name, query_type ,answertype="api"):
    with open(output_dir+"/generation_result/" + query_name + "/" + query_type + '/generation_'+model_name+'_new.jsonl', "r") as f:
        predict_all = f.readlines()
    predict = [json.loads(i)["predict"] for i in predict_all]
    format_acc = 0
    overalls_acc = 0
    platforms_acc = 0
    funcnames_acc = 0
    args_acc = 0
    funcs_acc = 0

    
    for i in range(len(predict)):
        real_answer = json.loads(predict_all[i])["label"]
        real_answer = [real_answer["toolname"],[{real_answer["apiname"]:{key:[value] for key,value in real_answer["parameters"].items()}}]]
        try:
            if answertype == "api":
                predict_answer = predict[i]
            else:
                predict_answer = ast_parse(predict[i])
            # print(predict_answer)
            # exit()
            overall_acc, platform_acc, funcname_acc, arg_acc, func_acc = evaluate(real_answer, predict_answer)
            format_acc += 1
            overalls_acc += overall_acc
            platforms_acc += platform_acc
            funcnames_acc += funcname_acc
            args_acc += arg_acc
            funcs_acc += func_acc
            # if overall_acc == 1:
            #     print(i)
        except Exception as e:
            print(i)
            continue

    file_path = output_dir + '/'+ test_type + "/" + query_name + "/" + query_type +"/acc_results_all.csv"
    if not os.path.exists(output_dir + '/'+ test_type + "/" + query_name + "/" + query_type):
        os.makedirs(output_dir + '/'+ test_type + "/" + query_name + "/" + query_type)
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
            name = ["name","format", "platforms", "tool_names", "tool_param_names", "tool_param_values", "overall"]
            writer.writerow(name)

        value = [model_name,
                round((format_acc / len(predict_all)),4),
                round((platforms_acc / len(predict_all)),4),
                round((funcnames_acc / len(predict_all)),4),
                round((args_acc / len(predict_all)),4),
                round((funcs_acc / len(predict_all)),4),
                round((overalls_acc / len(predict_all)),4)
                ]
        writer.writerow(value)

def cal_acc_all(test_type, query_name_list,output_dir, model_name, query_type,answertype="api"):
    all_len = 0
    format_acc = 0
    overalls_acc = 0
    platforms_acc = 0
    funcnames_acc = 0
    args_acc = 0
    funcs_acc = 0
    for query_name in query_name_list:
        with open(output_dir+"/generation_result/" + query_name + "/" + query_type + '/generation_'+model_name+'_new.jsonl', "r") as f:
            predict_all = f.readlines()
        all_len += len(predict_all)
        predict = [json.loads(i)["predict"] for i in predict_all]

        
        for i in range(len(predict)):
            real_answer = json.loads(predict_all[i])["label"]
            real_answer = [real_answer["toolname"],[{real_answer["apiname"]:{key:[value] for key,value in real_answer["parameters"].items()}}]]
            try:
                if answertype == "api":
                    predict_answer = predict[i]
                else:
                    predict_answer = ast_parse(predict[i])
                overall_acc, platform_acc, funcname_acc, arg_acc, func_acc = evaluate(real_answer, predict_answer)
                format_acc += 1
                overalls_acc += overall_acc
                platforms_acc += platform_acc
                funcnames_acc += funcname_acc
                args_acc += arg_acc
                funcs_acc += func_acc
                # if overall_acc == 1:
                #     print(i)
            except Exception as e:
                #print(i)
                continue

    file_path = output_dir + '/'+ test_type + "/all/acc_results_all.csv"
    if not os.path.exists(output_dir + '/'+ test_type + "/all"):
        os.makedirs(output_dir + '/'+ test_type + "/all")
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
            name = ["name","format", "platforms", "tool_names", "tool_param_names", "tool_param_values", "overall"]
            writer.writerow(name)

        value = [model_name,
                round((format_acc / all_len),4),
                round((platforms_acc / all_len),4),
                round((funcnames_acc / all_len),4),
                round((args_acc / all_len),4),
                round((funcs_acc / all_len),4),
                round((overalls_acc / all_len),4)
                ]
        writer.writerow(value)
person_name_list = ["u1","u2","u3","u4","u5","u6","u7","u8","u9","u10"]
query_type = "gpt"   
cal_all = True
cal_api = True
if cal_all == False:
    if cal_api == False:
        for person_name in person_name_list:
            query_name = "all_generated_queries_" + person_name
            for model_name in ["ToolACE","hammer","llama3","Mistral","Qwen2","watt","xlam","arch"]:
                print(cal_acc("results_acc",query_name,"predict",model_name=model_name,query_type=query_type,answertype="oss"))
    else:
        for person_name in person_name_list:
            query_name = "all_generated_queries_" + person_name
            for model_name in ["4o","r1","v3"]:
                print(cal_acc("results_acc",query_name,"predict",model_name=model_name,query_type=query_type,answertype="api"))
else:
    if cal_api == False:
        query_name_list = ["all_generated_queries_"+person_name for person_name in person_name_list]
        for model_name in ["ToolACE","hammer","llama3","Mistral","Qwen2","watt","xlam","arch"]:
            print(cal_acc_all("results_acc",query_name_list,"predict",model_name=model_name,query_type=query_type,answertype="oss"))
    else:
        query_name_list = ["all_generated_queries_"+person_name for person_name in person_name_list]
        for model_name in ["4o","r1","v3"]:
            print(cal_acc_all("results_acc",query_name_list,"predict",model_name=model_name,query_type=query_type,answertype="api"))