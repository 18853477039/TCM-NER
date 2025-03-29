# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 实现卡号匹配.
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning import seed_everything
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType,PromptEncoderConfig, PeftModel
import yaml
from data import MyDataset, CausalCollatorForEvaluate, CausalCollatorForTrain
from model import Model

logger = CSVLogger("./output/logs", name="")

def get_model(model_path:str, peft_config:dict, do_train:bool=True, peft_model_path:str=None):
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True, torch_dtype='auto')
    if do_train:
        if peft_config['peft_type'] == 'ptuning':
            config = peft_config['ptuning_config']
            config.update({'task_type':TaskType.CAUSAL_LM})
            config = PromptEncoderConfig(**config)
        elif peft_config['peft_type'] == 'lora':
            config = peft_config['lora_config']
            config.update({'task_type': TaskType.CAUSAL_LM})
            config = LoraConfig(**config)
        else:
            raise ValueError
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        assert peft_model_path is not None
        model = PeftModel.from_pretrained(model, peft_model_path)
    return model

def get_tokenizer(model_path:str):
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    return tokenizer

def main(config):
    train_config = config['train']
    peft_config = config['peft']
    seed_everything(train_config['seed'])
    model = get_model(train_config['model_path'], peft_config, train_config['do_train'], train_config['peft_model_path'])
    tokenizer = get_tokenizer(train_config['model_path'])
    save_path = train_config['save_path']
    os.makedirs(save_path, exist_ok=True)
    # train_dataset = MyDataset(train_config['train_data_path'])
    eval_dataset = MyDataset(train_config['eval_data_path'])
    train_collator = CausalCollatorForTrain(tokenizer=tokenizer,
                                            input_max_length=train_config['input_max_length'],
                                            output_max_length=train_config['output_max_length'],
                                            model_type=train_config['model_type'],
                                            )
    eval_collator = CausalCollatorForEvaluate(tokenizer=tokenizer,
                                              input_max_length=train_config['input_max_length'],
                                              output_max_length=train_config['output_max_length'],
                                              model_type=train_config['model_type'],
                                              )
    # dataloader_train = DataLoader(dataset=train_dataset,
    #                               batch_size=train_config['train_batch_size'],
    #                               collate_fn=train_collator,
    #                               drop_last=False,
    #                               shuffle=True
    #                               )
    dataloader_eval = DataLoader(dataset=eval_dataset,
                                  batch_size=train_config['eval_batch_size'],
                                  collate_fn=eval_collator,
                                  drop_last=False,
                                  )
    trainer = Trainer(max_epochs=train_config['epochs'],
                      val_check_interval=train_config['val_check_interval'],
                      log_every_n_steps=10,
                      accelerator='auto',
                      logger=logger,
                      accumulate_grad_batches=4,
                      )
    tuner = Model(model,
                  tokenizer,
                  input_max_length=train_config['input_max_length'],
                  output_max_length=train_config['output_max_length'],
                  model_save_dir=train_config['save_path'],
                  lr = train_config['lr'],
                  output_entity_scores = train_config['output_entity_scores'],
                  save_model = train_config['do_train'],
                  )
    if train_config['do_train']:
        trainer.fit(
            model=tuner,
            train_dataloaders=dataloader_train,
            val_dataloaders=dataloader_eval,
        )
    else:
        trainer.test(model=tuner, dataloaders=dataloader_eval)


if __name__ == "__main__":
    from rich import print
    # config = yaml.safe_load(open('config.yml', 'r'))
    config = yaml.safe_load(open('config_local.yml', 'r'))
    main(config)
