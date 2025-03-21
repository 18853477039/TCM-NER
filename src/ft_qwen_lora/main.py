import json
import logging
import os
import sys

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

import transformers
from sklearn.metrics import f1_score
from transformers import HfArgumentParser, AutoTokenizer, set_seed, Seq2SeqTrainingArguments, AutoModelForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainer

import sys
sys.path.append("./")

from src.ft_qwen_lora.arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )
    print("raw_datasets: ", raw_datasets)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    if model_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, model_args.peft_path)
    else:
        logger.info("Init new peft model")
        # target_modules = model_args.trainable.split(',')
        modules_to_save = model_args.modules_to_save.split(',') if model_args.modules_to_save != "null" else None
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        # print(target_modules)
        print(lora_rank)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # target_modules=target_modules if target_modules != "null" else None,
            inference_mode=False if training_args.do_train else True,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad, p.numel())

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function_train(examples):
        max_seq_length = min(data_args.max_source_length + data_args.max_target_length, 2048)

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]
                prompt = prefix + query

                # 编码输入和输出
                inputs = tokenizer(
                    prompt,
                    max_length=data_args.max_source_length,
                    truncation=True,
                    padding="max_length",
                )
                labels = tokenizer(
                    answer,
                    max_length=data_args.max_target_length,
                    truncation=True,
                    padding="max_length",
                )["input_ids"]

                # 忽略填充部分的损失
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                # 添加到 model_inputs
                model_inputs["input_ids"].append(inputs["input_ids"])
                model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if not examples[response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[response_column][i])

            if examples[prompt_column][i]:
                query = examples[prompt_column][i]
                inputs.append(query)

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs,
                                 max_length=data_args.max_source_length,
                                 truncation=True,
                                 padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def print_dataset_example(example):
        print("input_ids: ", example["input_ids"])
        print("inputs: ", tokenizer.decode(example["input_ids"]))
        print("label_ids: ", example["labels"])
        print("labels: ", tokenizer.decode([token_id for token_id in example["labels"] if token_id != -100]))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0])
        print_dataset_example(train_dataset[1])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])
        print_dataset_example(eval_dataset[1])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])
        print_dataset_example(predict_dataset[1])

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False if training_args.do_train else True
    )

    # Metric
    def compute_metrics(eval_preds):
        """
        计算基于词的 F1 分数。
        :param eval_preds: 包含 predictions 和 label_ids 的元组。
        :return: 包含 F1 分数的字典。
        """
        predictions, label_ids = eval_preds

        # 将 predictions 和 label_ids 解码为文本
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 计算基于词的 F1 分数
        def compute_word_f1(preds, labels):
            # 将生成文本转换为实体列表
            def extract_entities(text):
                entities = []
                for line in text.split("\n"):
                    if line.startswith("上述句子中的实体包含："):
                        continue
                    if "实体：" in line:
                        entity_type, entity_values = line.split("实体：")
                        for value in entity_values.split("，"):
                            if value.strip():
                                entities.append({"type": entity_type.strip(), "value": value.strip()})
                return entities

            # 提取真实标签和预测标签
            true_entities = extract_entities(labels)
            pred_entities = extract_entities(preds)

            # 计算 Precision、Recall 和 F1 score
            true_labels = [e["type"] for e in true_entities]
            pred_labels = [e["type"] for e in pred_entities]
            f1 = f1_score(true_labels, pred_labels, average="weighted")
            return f1

        # 计算每个样本的 F1 分数
        f1_scores = [compute_word_f1(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
        avg_f1 = float(np.mean(f1_scores))  # 计算平均 F1 分数

        return {"word_f1": avg_f1}

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=512, temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 读取原test file
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            # max_tokens=512,
            max_new_tokens=data_args.max_target_length,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            # top_p=0.7,
            # temperature=0.95,
            # repetition_penalty=1.1
        )
        metrics = predict_results.metrics
        print(metrics)
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    return results


if __name__ == "__main__":
    main()
