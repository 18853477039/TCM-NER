# !/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from toolz.tests.test_dicttoolz import defaultdict
from torch.optim import AdamW
import lightning as L
from transformers import GenerationConfig
from rouge_chinese import Rouge
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import collections
import re
from rich import print

jieba.initialize()


class Model(L.LightningModule):
    def __init__(self, model, tokenizer, input_max_length, output_max_length, model_save_dir,lr, output_entity_scores=False, save_model=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.input_ids = []
        self.output_ids = []
        self.labels = []
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.score_dict = {
            'rouge-1':[],
            'rouge-2':[],
            'rouge-l':[],
            'bleu-4':[],
            'F1':[]
        }
        self.best = 0.
        self.model_save_dir = model_save_dir
        self.lr = lr
        self.output_entity_scores = output_entity_scores
        self.save_model = save_model
        if output_entity_scores:
            self.entity_counter = collections.defaultdict(dict)

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        input_ids = batch['input_ids']
        labels = torch.where(batch['labels']==-100, self.tokenizer.pad_token_id, batch['labels'])
        attention_mask = batch['attention_mask']
        generation_config = GenerationConfig(
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=False,
            output_scores=False,
            max_new_tokens=self.output_max_length ,
        )
        output_ids = self.model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         generation_config=generation_config,
                                         )
        output_ids = output_ids[...,input_ids.size(1):]

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        label_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(output_text,label_text):
            if not pred:
                pred = '-'
            if self.output_entity_scores:
                self.count_entity(pred, label)
            pred = ' '.join(jieba.cut(pred))
            label = ' '.join(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(pred, label)
            result = scores[0]
            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            self.score_dict['F1'].append(round(result['rouge-1']['f']*100, 4)) #F1 score就是rouge-1的f值

    def on_validation_epoch_end(self):
        metrics = collections.defaultdict(float)
        for k,v in self.score_dict.items():
            metrics[k] = float(np.mean(v))
        metrics.update({'main':metrics['rouge-l']})
        for k in self.score_dict.keys():
            self.score_dict[k].clear()
        self.log_dict(metrics)
        self.print(metrics)
        if self.save_model and metrics['main'] > self.best:
            self.model.save_pretrained(self.model_save_dir)
        self.log_best(metrics)

        if self.output_entity_scores:
            print(self.entity_counter)
            metrics_entity = collections.defaultdict(dict)
            for k,v in self.entity_counter.items():
                precision = float(v['tp']/(v['pred_entity_num'])) if v['pred_entity_num'] > 0 else 0.
                recall = float(v['tp']/(v['label_entity_num'])) if v['label_entity_num'] > 0 else 0.
                f1 = float(2*precision*recall/(precision+recall)) if precision+recall > 0 else 0.
                metrics_entity[k] = {'precision':precision,
                                     'recall':recall,
                                     'f1':f1
                                     }
            print(metrics_entity)
            self.entity_counter.clear()

    def count_entity(self, pred, label):
        '''计算每个实体类别的tp,pred_entity_num,label_entity_num'''
        entities_pred = self.parse_entity(pred)
        entities_label = self.parse_entity(label)
        all_names = [*entities_pred.keys(), *entities_label.keys()]
        for name in all_names:
            counter = self.entity_counter.get(name, {'tp':0, 'pred_entity_num':0 ,'label_entity_num':0})
            counter['tp'] += len([ent for ent in entities_pred.get(name,[]) if ent in entities_label.get(name,[])]) #预测正确的实体数
            counter['pred_entity_num'] += len(entities_pred.get(name,[])) #预测的实体总数
            counter['label_entity_num'] += len(entities_label.get(name,[])) #实际的实体总数
            self.entity_counter[name].update(counter)

    # def parse_entity(self, text):
    #     '''解析实体'''
    #     #假设文本中的实体都是 XXX:YYY 的形式
    #     p = r'[^,]+:[^,]+'
    #     res = defaultdict(list)
    #     entities = re.findall(p, text)
    #     for entity in entities:
    #         ent, name = entity.split(':')
    #         res[name.strip()].append(ent.strip())
    #     return res

    def parse_entity(self, text):
        entities = defaultdict(list)
        if not text:
            return entities  # 如果文本为空，返回空列表

        for line in text.split("\n"):
            if line.startswith("上述句子中的实体包含："):
                continue
            if "实体：" in line:
                entity_type, entity_values = line.split("实体：")
                for value in entity_values.split("，"):
                    if value.strip():
                        entities[entity_type.strip()].append(value.strip())
        return entities

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.model.train()

    def log_best(self, metrics) -> None:
        self.best = metrics['main'] if self.best < metrics['main'] else self.best

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, eps=1e-8)
        return optimizer

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_start(self):
        self.model.eval()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()










if __name__ == "__main__":
    run_code = 0
