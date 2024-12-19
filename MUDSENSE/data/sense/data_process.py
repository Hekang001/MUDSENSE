import json
import re


def jsonl_to_json(input_path,output_path):
    line_number = 0

    try:
        with open( input_path , 'r') as jsonl_file:
            for line in jsonl_file:
                line_number += 1
                json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Error on line {line_number}: {e}")

    # 如果没有发生错误，表示文件是正确的
    else:
        print("File parsed successfully.")
        with open(input_path, 'r') as jsonl_file:
            # 读取所有行并解析为JSON对象
            data = [json.loads(line) for line in jsonl_file.readlines()]

    # 将所有对象写入一个数组并保存为.json文件
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def remove_invalid_entities(data):
    """Remove invalid entities from the data."""
    for instance in data:
        sentence_length = len(instance['sentence'])

        entities_to_remove = []  # list to store invalid entities

        for entity in instance["ner"]:
            index = entity["index"]
            for i in index:
                if i >= sentence_length:
                    print(f"Error in data: Entity index {i} out of range for sentence of length {sentence_length}.")
                    print("Sentence:", instance['sentence'])
                    print("Entity:", entity)
                    print("\n")
                    entities_to_remove.append(entity)
                    break

        # Remove invalid entities from instance["ner"]
        for invalid_entity in entities_to_remove:
            instance["ner"].remove(invalid_entity)


'''
selector sentence
'''
def extract_relevant_subclauses(label, text):
    # Split the text into subclauses using comma and period as delimiters
    subclauses = re.split(r'[，。；！？]', text)
    
    relevant_subclauses = []

    for subclause in subclauses:
        for entity in label:
            for key, value in entity.items():
                # Check if either the key or the value is present in each subclause
                if key in subclause or value in subclause:
                    relevant_subclauses.append(subclause.strip())
                    break  # Once a match is found, break out of the inner loop
            if len(relevant_subclauses) > 0 and relevant_subclauses[-1] == subclause.strip():
                break  # Once a match is found, break out of the middle loop

    # Combine the relevant subclauses into a single string
    new_text = '，'.join(relevant_subclauses)  # Use Chinese comma as delimiter
    return new_text


'''
转为训练数据的格式
'''
result = []
def get_indexes_from_selector_text(file, selector_text, label):
    # Split the selector_text into individual words/characters
    selector_text = selector_text.replace("\n\n", "").replace("\n", "")
    words = list(selector_text)
    length = len(words)
    if length >= 510 :
        words = words[:510]
    instance = {"sentence": words, "ner":[]}
    
    if length >= 1000: 
        file.write (f"len: {length} text: {selector_text}\n")
    
    # For each entity in label, find its start and end index
    for entity in label:
        for key, value in entity.items():
            start_index = selector_text.find(value)
            if start_index != -1:
                end_index = start_index + len(value) - 1

                # Check if end_index is out-of-bounds and adjust if necessary
                # end_index = end_index + len(value) - 1
                if start_index > len(words) or end_index > len(words):
                    print(f"Warning: Entity indices ({start_index}, {end_index}) are out of range for sentence of length {len(words)}. Skipping this entity.")
                    continue

                if start_index < end_index and start_index != end_index:
                    instance["ner"].append({
                        # "index": [start_index, end_index],
                        "index": list(range(start_index, end_index + 1)),
                        "type": key
                    })
                else:
                    instance["ner"].append({
                        "index": [start_index],
                        "type": key
                    })


    result.append(instance)



input_path = '/home/hekang/MUDSENSE/data/sense/data_generate.jsonl'
output_path = '/home/hekang/MUDSENSE/data/sense/data.json'

jsonl_to_json(input_path,output_path)

with open(output_path ,'r') as f:
    datas= json.load(f)

# Update each item in the dataset
for item in datas:
    new_text = extract_relevant_subclauses(item['label'], item['text'])
    item['selector_text'] = new_text


with open('/home/hekang/MUDSENSE/data/sense/selector_data.json', 'w', encoding='utf-8') as file:
    json.dump(datas, file, ensure_ascii=False, indent=4)


for data in datas:
    with open("/home/hekang/MUDSENSE/data/sense/error.txt",'a') as file:
        get_indexes_from_selector_text(file, data['selector_text'], data['label'])

'''
划分数据集
'''
total_numbers = len(result)
train_data = result[:int(total_numbers*0.8)]
dev_data = result[int(total_numbers*0.8):int(total_numbers*0.9)]
test_data = result[int(total_numbers*0.9):]


remove_invalid_entities(train_data)
remove_invalid_entities(dev_data)
remove_invalid_entities(test_data)


with open("/home/hekang/MUDSENSE/data/sense/train.json", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)
with open('/home/hekang/MUDSENSE/data/sense/dev.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, ensure_ascii=False, indent=4)
with open("/home/hekang/MUDSENSE/data/sense/test.json", 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)


