import json
import re
import copy
import argparse

def contains_non_chinese(text, l):
    # 使用正则表达式匹配非中文字符
    pattern = re.compile(r'[^\u4e00-\u9fa5 ]')
    non_chinese_chars = pattern.findall(text)

    return len(non_chinese_chars) >= l  # 如果非中文字符的数量大于等于l，则返回True，否则返回False



def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='./config/sense.json')
    # parser.add_argument('--save_path', type=str, default='sense_model.pt')
    parser.add_argument('--config', type=str, default='./MUDSENSE/config/sense.json')
    parser.add_argument('--save_path', type=str, default='./MUDSENSE/sense_model.pt')

    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    return args


def selector(text):
    sentence_list = []
    sub_list = ['\t', 'Android', 'Windows', 'Windws', 'Windws 7 ', 'Windows 7 ', 'VPN', 'Wi-Fi', 'DNS', 'L2TP', 'excel',
                'Java', 'Chrome', '.al', 'bit.EXE', '32位', '64位', r'\b([01]?\d|2[0-3]):([0-5]?\d)\b',
                r'\d{4}年\d{1,2}月\d{1,2}日', r'\b\d{4}-\d{4}\b', r'\d{4} 年 \d{2} 月', ' ']
    retain_list = []

    for i in sub_list:
        text = re.sub(i, '', text)
    sentences = re.split('\n', text)

    for sentence in sentences:
        sentence_splits = re.split(r'[，。：！？;；“”"《》\[\]【】（）、—－]', sentence)
        for sentence_split in sentence_splits:
            selector = 0
            # print(sentence_split)
            if contains_non_chinese(text=sentence_split, l=3):
                selector = 1
            # 关键字句保留
            else:
                for char in sentence_split:
                    if char in retain_list:
                        selector = 1
            if selector == 1:
                sentence_list.append(sentence_split)
    new_text = ' '.join(sentence_list)
    return new_text



def select(input_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        j = 0
        amount = 100
        for js in f_in:
            line = json.loads(js)
            for key, value in line.items():
                # print(line)
                text = value
                new_text = selector(text)
                print(f'before：{len(text)} \n{line} ')
                # if len(new_text) > 500:
                print(f'after：{len(new_text)} \n{new_text}')
                print('\n')
            j += 1
            if j >= amount:
                break

# 加载sense文件及包含的数据，输入模型进行预测
def load_data_process_file(inference_path):
    sense_relative_path = None
    selector_text = None
    result = []
    with open(inference_path, "r",encoding="utf-8") as file:
        for f in file:
            line = json.loads(f)
            for key,value in line.items():
                sense_relative_path = key
                num_value = len(list(value))
                # if num_value >500:
                #     selector_text = selector(value)
                # else:
                #     selector_text = value
                # words = list(selector_text)
                words = list(value)
                # 按照510进行截取
                max_length = 510
                segments = [words[i:i + max_length] for i in range(0, len(words), max_length)]
        
                for segment in segments:
                  result.append({
                      "relative_path":sense_relative_path,
                      "sentence":segment,
                      'ner':[]
                  })
      
    return result


def concatenate_file(result):
    # 创建一个字典，以"relative_path"作为键，将具有相同路径的字典内容拼接在一起
    merged_data = {}
    for entry in result:
        path = entry["relative_path"]
        if path in merged_data:
            # 如果已经存在相同路径的字典，则拼接sentence和predict_entity
            merged_data[path]["sentence"] += " " + entry["sentence"]
            # 也可以拼接其他的预测实体
            for key in entry:
                if key.startswith("predict_entity"):
                    if key not in merged_data[path]:
                        merged_data[path][key] = {}
                    merged_data[path][key].update(entry[key])
        else:
            # 如果路径不存在，直接添加到merged_data
            merged_data[path] = entry

    # 将合并后的字典转换为列表
    merged_data_list = list(merged_data.values())

    # 打印合并后的结果
    for entry in merged_data_list:
        print(entry)



# 修改正则部分的输出，将pos位置变更
def format_json(data):
    new_data = {}
    data = json.loads(data)
    
    # 获取文件路径和数据列表
    file_path = list(data.keys())[0]
    data_list = data[file_path]
    print("data_list:{}".format(data_list))
    post_list = data["pos"]
    
    for item in data_list:
        new_item = [{}]
        for i, (key, value) in enumerate(item.items()):
            pos = post_list[i]  # 获取位置信息
            if isinstance(pos, list):
                new_item[key] = [{"value": val, "pos": p} for val, p in zip(value, pos)]
            else:
                new_item[key] = {"value": value, "pos": pos}
        new_data[file_path] = new_item

    return new_data


def find_closest_value(value,dict_value:dict):
    items = list(dict_value.items())
    previous_key = None
    previous_value = None
    pos = None
    for i in range (len(items)):
        if items[i][0] == value:
            if i > 0:
                previous_key, previous_value = items[i - 1]
                pos = i
    
    return previous_key, previous_value, pos

# # 
# def sense_format_cluster(data):
#     result = []
#     for item in data:

#         new_item = copy.deepcopy(item)  # 复制原始数据
#         relative_path = new_item['relative_path']
#         cluster = []

#         # 判断是否存在 "predict_entity_1"，如果存在则创建 "cluster"
#         if "predict_entity_1" in new_item:
#             new_item["cluster"] = [
#                 new_item["predict_entity"], new_item["predict_entity_1"]
#             ]
#             # 移除 "predict_entity" 和 "predict_entity_1" 来减小数据体积
#             new_item.pop("predict_entity")
#             new_item.pop("predict_entity_1")
#             cluster.append(new_item['cluster'])
#         else:
#             new_item["cluster"] = [
#                 new_item["predict_entity"]
#             ]
#             new_item.pop("predict_entity")
#             cluster.append(new_item['cluster'])
        
#         result.append(new_item)

    
        
#     return result

# 转为cluster
def sense_format_cluster(data):
    result = {}
    index = 0
    for item in data:
        new_item = copy.deepcopy(item)  # 复制原始数据
        relative_path = new_item['relative_path']
        cluster = []
        pos= []

        # 判断是否存在 "predict_entity_1"，如果存在则创建 "cluster"
        if "predict_entity_1" in new_item:
            new_item["cluster"] = [
                new_item["predict_entity"], new_item["predict_entity_1"]
            ]
            # 移除 "predict_entity" 和 "predict_entity_1" 来减小数据体积
            new_item.pop("predict_entity")
            new_item.pop("predict_entity_1")
            for da in new_item['cluster']:
                cluster.append(da)
        else:
            new_item["cluster"] = [
                new_item["predict_entity"]
            ]
            new_item.pop("predict_entity")
            for da in new_item['cluster']:
                cluster.append(da)
   
        # 检查relative_path是否已经存在于result中
 
        if relative_path in result:
            index +=1
            for item in new_item['cluster']:
                for key, value in item.items():
                      if 'pos' in value:
                          format_pos = value['pos'] + index * 510
                          value['pos'] = format_pos
            # 如果存在，直接将新的cluster添加到已有的列表中
            if new_item['cluster']:
                result[relative_path].extend(new_item['cluster'])
        else:
            # 如果不存在，创建一个新的键值对
            if new_item['cluster']:
                result[relative_path] = new_item['cluster']

    return result


if __name__ =="__main__":
    inference_path = "/home/hekang/MUDSENSE/pptx_docx_text.jsonl"
    result = load_data_process_file(inference_path)
    select(inference_path)
    with open("inference_input.json",'w',encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4) 