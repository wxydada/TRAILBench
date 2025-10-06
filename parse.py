import ast

def ast_parse(input_str): 
    return ast_parse_1(input_str)

def _func_name_from_node(func_node: ast.AST) -> str:
    """提取调用表达式的函数名，支持 a.b.c() 形式。"""
    parts = []
    node = func_node
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))

def ast_parse_1(input_str):
    input_str = '{'+input_str+'}'
    parsed = ast.parse(input_str, mode="eval")
    node = parsed.body
    extracted = []

    # 1) 旧格式：{toolname: [func1(...), func2(...)]} 或 {toolname: func1(...)}
    if isinstance(node, ast.Dict):
        # 取第一个key作为toolname
        key_node = node.keys[0]
        if isinstance(key_node, ast.Name):
            key = key_node.id
        elif isinstance(key_node, ast.Constant):
            key = key_node.value
        else:
            raise Exception(f"Unsupported key AST type: {type(key_node)}")

        body = node.values[0]
        if isinstance(body, ast.Call):
            extracted.append(resolve_ast_call(body))
        else:
            # 假定为 [Call, Call, ...] / (Call, ...)
            for elem in body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return (key, extracted)

    # 2) 新格式：{toolname(...)} 或 {toolA(...), toolB(...)}
    # 在 Python AST 中这是 Set：ast.Set(elts=[Call, Call, ...])
    if isinstance(node, ast.Set):
        if not node.elts:
            raise Exception("Empty set literal.")
        # 第一个调用的函数名作为 key
        first_call = node.elts[0]
        if not isinstance(first_call, ast.Call):
            raise Exception("Set must contain function calls.")
        key = _func_name_from_node(first_call.func).split("_", 1)[0]
        for elem in node.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(resolve_ast_call(elem))
        return (key, extracted)

    raise Exception(f"Unsupported top-level AST type: {type(node)}")

def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts)).split("_",1)[1]
    # func_name = ".".join(reversed(func_parts)).split("_",1)[1]  #注意
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}

def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.NameConstant):
        output = value.value
    elif isinstance(value, ast.BinOp):
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


# with open("./predict/test_untrained_user/generation_v3.jsonl") as f:
#     for data in f:
#         txt = json.loads(data)
#         print(txt["predict"])
#         key,extracted = ast_parse(txt["predict"])
#         print(extracted)
#         exit()


# key,extracted = ast_parse("GaodeMap_get_path(start_name = The River Mall, end_name = Shanghai Jiao Tong University, cartype = public_transit)")
# print(key)
# print(extracted)
