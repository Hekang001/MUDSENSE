import configparser
import requests
import os
import json
import zipfile

config = configparser.ConfigParser()


def extract_and_rename_zip(file_path):
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # 创建一个文件夹来解压文件
            folder_name = os.path.splitext(file_path)[0] + "_zip"
            os.makedirs(folder_name, exist_ok=True)
            zip_ref.extractall(folder_name)

        # 删除原始压缩文件
        os.remove(file_path)
        print(f"压缩文件 {file_path} 已解压并原文件已删除，文件夹名称为 {folder_name}")


def predict(root_directory, result_path, prompt_path):
    # 创建ConfigParser对象
    config = configparser.ConfigParser()

    # 读取.config文件
    read = config.read('LLM_code/config_file.config')

    # 获取配置项的值
    url = config.get('url', 'url')
    headers = eval(config.get('url', 'headers'))
    name = config.get('model', 'name')

    # root_directory = config.get('path', 'root')
    # result_path = config.get('path', 'result')
    # prompt_path = config.get('path', 'prompt')

    for root, directories, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(root, file)
            extract_and_rename_zip(file_path)

    # 使用os.walk遍历目录并获取所有文件路径
    if not os._exists(result_path):

        file_paths = []
        for root, directories, files in os.walk(root_directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        # 读入prompt文件
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        res = {}
        error_path = []  # 输出可能存在问题

        file_name = [".py", ".rs", ".sh", ".cpp", ".java"]

        for file_path in file_paths:
            # 是否需要过滤文件类型
            if file_path.endswith(tuple(file_name)):
                print("Extracting from %s" % file_path)
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    data = {
                        "model": name,
                        "messages": [{"role": "user",
                                      "content": prompt + "\ncode as follow:\n" + f"{file_content}" + "\noutput:\n"}]
                    }
                    response = requests.post(url, headers=headers, json=data)
                    try:
                        data = response.json()
                    except json.JSONDecodeError as e:
                        continue
                    result = data['choices'][0]['message']['content'].replace("\n", "")
                    res = [result]
                    key = file_path
                    # key = file_path.replace("/home/wanghaining/workspace/LLM_code_REG/LLM_code/", "")
                    with open(result_path, 'a', encoding='utf-8') as wf:
                        json_str = json.dumps({key: res})
                        wf.write(json_str)
                        wf.write("\n")


if __name__ == "__main__":
    root_directory = 'LLM_code/code_data'
    result_path = 'LLM_code/llm_result.jsonl'
    prompt_path = 'LLM_code/prompt.txt'

    predict(root_directory, result_path, prompt_path)