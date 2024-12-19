import json
path="./result.jsonl"
all_data=[]
with open(path,'r') as f:
    for line in f:
        all_data.append(json.loads(line))
print("hh")