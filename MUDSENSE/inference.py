import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config 
import data_loader
from tool.data_process import find_closest_value, sense_format_cluster,get_args
import utils
from model import Model


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]
        updates_total=216200
        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-1]]

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        # logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        '''
        将模型设置为评估模式。dropout和batch normalization层将固定，不会在前向传播中进行任何更改。
        '''
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                '''
                后处理预测结果:将模型的输出转化为实体标签，并与真实的实体标签进行比较以计算评估指标。
                '''
                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        # logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
        #                                                     pred_result.numpy(),
        #                                                     average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        # logger.info("\n{}".format(table))
        #return e_f1
        return e_p

    def predict(self, epoch, data_loader, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i+config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
                
                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence_ori =sentence["sentence"]
                    sentence_1 = sentence["sentence"]
                    full_sentence = ' '.join(sentence_1)
                    # instance = {"sentence": full_sentence, "predict_entity": [], "true_entity":[]}
                    output = {"sentence":full_sentence, "result":[]}
                    # "entity":[], "预测结果":[],"真实结果":[]
                    for ent in ent_list:
                        for ner_entry in sentence["ner"]:
                            start, end = ner_entry["index"][0],ner_entry['index'][-1]
                            entity_text_1 = sentence_ori[start:end+1]
                            entity_type = ner_entry["type"]
                            text =  [sentence_1[x] for x in ent[0]]
                            full_text_predict= ' '.join(text)
                            full_text_out = ' '.join(entity_text_1)

                            if full_text_predict == full_text_out:
                                output['result'].append({"实体": full_text_predict,
                                                        "预测结果": config.vocab.id_to_label(ent[1]),
                                                        "真实结果":entity_type})
                        
                        
                    result.append(output)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        # logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
        #                                                     pred_result.numpy(),
        #                                                     average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        # logger.info("\n{}".format(table))
        # print(result)

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1
    

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

 
    '''
    sense预测
    '''
    def inference(self, data_loader, data, inference_save_path):
        self.model.eval()

        result = []

        i = 0

        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i+config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, _ , grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length
                # print(length)

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                _, _, _, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence_1 = sentence["sentence"]
                    relative_path = sentence['relative_path']
                    full_sentence = ''.join(sentence_1)

                    # 创建一个新的字典来存储预测实体
                    instance = {"sentence": full_sentence,"relative_path":relative_path, "predict_entity": {}}
                    out = {}
                    
                    instance_1 = None  # 初始化一个新的字典

                    sorted_ent_list = sorted(ent_list, key=lambda ent: min(ent[0]))
                    
                    # dict中的键为value，而值为value对应的最小下标{'12051178029': 9, 'lisa65': 28, 'Joe Tate': 38, '05004822': 52, '008180303066': 64, '.U]}y:/_psxN#m': 80}
                    dict1 = { }
                    for ent in sorted_ent_list:
                        # print(ent[0])
                        value = ''.join([sentence_1[x] for x in ent[0]])
                        key = config.vocab.id_to_label(ent[1])

                        if value not in dict1 or ent[0][0] < dict1[value]:
                            dict1[value] = ent[0][0]    

                    for ent in sorted_ent_list:
                        # print(ent[0])
                        pos =min(x for x in ent[0])
                        value = ''.join([sentence_1[x] for x in ent[0]])
                        key = config.vocab.id_to_label(ent[1])
                        prev_key = None
                        prev_value = None
                       
                        # 检查是否已经存在相同的键
                        if key in instance["predict_entity"]:
                            # 如果键已经存在，检查是否已经创建instance_1
                            if instance_1 is None:
                                # instance_1['predict_entity_1'][key]=value

                                instance_1 = {"predict_entity_1": {key: {"value":value,"pos":pos}}}
                                instance['predict_entity_1'] = instance_1['predict_entity_1']
                            else:
                                # 如果instance_1已经存在，将新的键-值对添加到instance_1中
                                instance["predict_entity_1"][key] = {"value":value,"pos":pos}

                        else:
                            if "predict_entity_1" in instance:
                                prev_key, prev_value, pos1 = find_closest_value(value, dict1)
                                # 查看哪个字典中包含value的前一个value 
                                if prev_key in instance['predict_entity'].values():
                                    instance["predict_entity"][key] = {"value":value,"pos":pos1}
                                else:
                                    # 如果键不存在，将键-值对添加到instance中
                                    instance["predict_entity_1"][key] = {"value":value,"pos":pos1}
                            else:
                                instance["predict_entity"][key] = {"value":value,"pos":pos}

                    result.append(instance)
            
                i += config.batch_size

        result = sense_format_cluster(result)

        with open(inference_save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)


args = get_args()
config = config.Config(args)
# logger = utils.get_logger(config.dataset)
# logger.info(config)
# config.logger = logger
# logger = utils.get_logger(config.dataset)
# logger.info(config)
# config.logger = logger
config.label_num = 7

if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
model = Model(config)

model = model.cuda()

datasets, ori_data = data_loader.load_data_bert(config)

train_loader, dev_loader, test_loader = (
    DataLoader(dataset=dataset,
                batch_size=config.batch_size,
                collate_fn=data_loader.collate_fn,
                shuffle=i == 0,
                num_workers=4,
                drop_last=i == 0)
    for i, dataset in enumerate(datasets)
)



def predict(input_path, output_path):
    '''
    输入数据地址，进行NER预测
    '''
    predict_datasets, predict_data = data_loader.inference_load_data_bert(inference_path=input_path, config=config)

    predict_loader = DataLoader(dataset=predict_datasets,
                                batch_size=config.batch_size,
                                collate_fn=data_loader.collate_fn,
                                shuffle=False,
                                num_workers=4,
                                drop_last=False)

    # logger.info("Building Model")
    print("Building Model")

    trainer = Trainer(model)
    # 训练模型
    trainer.load(config.save_path)

    trainer.inference(predict_loader, predict_data, output_path)


if __name__ == '__main__':
    input_path = "inference/inference.jsonl"
    output_path = 'inference/inference_222222.json'
    predict(input_path,output_path)
    print("预测成功")