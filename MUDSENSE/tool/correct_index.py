import json

def find_invalid_entries(data):
    """Find and return indices of invalid entries in the data."""
    invalid_indices = []

    for idx, instance in enumerate(data):
        sentence_length = len(instance['sentence'])
        for entity in instance["ner"]:
            index = entity["index"]
            for i in index:
                if i >= sentence_length:
                    print(f"Error in data: Entity index {i} out of range for sentence of length {sentence_length}.")
                    print("Sentence:", instance['sentence'])
                    print("Entity:", entity)
                    print("\n")
                    invalid_indices.append(idx)
                    break

    return set(invalid_indices)


if __name__ == "__main__":
    with open("/home/hekang/W2NER/VER_data/data/1_ver_train_1.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('/home/hekang/W2NER/VER_data/data/1_ver_dev_1.json', 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open("/home/hekang/W2NER/VER_data/data/1_ver_test_1.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    invalid_train_indices = find_invalid_entries(train_data)
    train_data = [entry for i, entry in enumerate(train_data) if i not in invalid_train_indices]

    invalid_dev_indices = find_invalid_entries(dev_data)
    dev_data = [entry for i, entry in enumerate(dev_data) if i not in invalid_dev_indices]

    invalid_test_indices = find_invalid_entries(test_data)
    test_data = [entry for i, entry in enumerate(test_data) if i not in invalid_test_indices]

    # Save corrected data back if required
    with open("/home/hekang/W2NER/VER_data/data/corrected_ver_train_1.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open('/home/hekang/W2NER/VER_data/data/corrected_ver_dev_1.json', 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
    with open("/home/hekang/W2NER/VER_data/data/corrected_ver_test_1.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

