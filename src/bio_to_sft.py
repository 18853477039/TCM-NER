import json
import argparse


def load_bio_data(file_path):
    """加载BIO格式的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tokens = []
    labels = []
    current_tokens = []
    current_labels = []
    line_num = 0

    for line in lines:
        try:
            line_num += 1
            line = line.rstrip()
            if line:
                # 处理行首有空格的情况
                if line.startswith(' '):  # 如果行首有空格
                    token = ' '  # 将空格作为 token
                    label = line[1:].strip()  # 剩余部分作为 label
                else:
                    # 正常情况：按第一个空格分割
                    parts = line.split(maxsplit=1)
                    if len(parts) == 1:
                        # 如果只有一位（例如 "O"），则默认标签为 "O"
                        token = parts[0]
                        label = "O"
                    else:
                        # 正常情况：token 和 label 分开
                        token, label = parts
                current_tokens.append(token)
                current_labels.append(label)
            else:
                # 如果遇到空行，表示一个句子的结束
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
        except Exception as e:
            print(f"Error in line {line_num}: {line}")
            print(f"Error message: {e}")
            continue

    # 添加最后一个句子（如果存在）
    if current_tokens:
        tokens.append(current_tokens)
        labels.append(current_labels)

    return tokens, labels


def extract_entity_types(labels):
    """从labels中提取实体类型（去掉B-/I-前缀并去重）"""
    entity_types = set()
    for label_list in labels:
        for label in label_list:
            if label != "O":
                entity_type = label.split("-")[1]  # 去掉B-/I-前缀
                entity_types.add(entity_type)
    return list(entity_types)


def convert_to_sft_format(tokens, labels, dataset_name):
    """将BIO格式的数据转换为SFT格式"""
    sft_data = []
    entity_types = extract_entity_types(labels)  # 提取实体类型

    for i, (token_list, label_list) in enumerate(zip(tokens, labels)):
        # 构造 input
        input_text = "中医药命名实体识别: \n" + "".join(token_list) + "\n答："

        # 构造 target
        entity_dict = {et: [] for et in entity_types}  # 按实体类型分组
        current_entity = None
        current_tokens = []

        for token, label in zip(token_list, label_list):
            if label.startswith("B-"):
                # 如果是新的实体，保存上一个实体
                if current_entity:
                    entity_dict[current_entity].append("".join(current_tokens))
                current_entity = label.split("-")[1]  # 提取实体类型
                current_tokens = [token]  # 开始新的实体
            elif label.startswith("I-"):
                # 如果是实体的中间部分，继续添加token
                current_tokens.append(token)
            else:
                # 如果是O，保存上一个实体
                if current_entity:
                    entity_dict[current_entity].append("".join(current_tokens))
                current_entity = None
                current_tokens = []

        # 保存最后一个实体
        if current_entity:
            entity_dict[current_entity].append("".join(current_tokens))

        # 构造target文本
        target_text = "上述句子中的实体包含：\n"
        no_entity = True
        for et in entity_types:
            if entity_dict[et]:
                no_entity = False
                target_text += f"{et}实体：{'，'.join(entity_dict[et])}\n"
        if no_entity:
            target_text = "上述句子没有指定类型实体"

        # 构造样本
        sample = {
            "input": input_text,
            "target": target_text.strip(),  # 去掉末尾的换行符
            "task_type": "ner",
            "task_dataset": dataset_name,
            "sample_id": f"train-{i}"  # 生成唯一的样本ID
        }
        sft_data.append(sample)

    return sft_data


def save_to_jsonl(data, file_path):
    """将数据保存为 .jsonl 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def bio_to_sft(input_file, output_file, dataset_name):
    """将BIO格式的文件转换为SFT格式并保存为 .jsonl 文件"""
    # 加载BIO数据
    tokens, labels = load_bio_data(input_file)

    # 转换为SFT格式
    sft_data = convert_to_sft_format(tokens, labels, dataset_name)

    # 保存为 .jsonl 文件
    save_to_jsonl(sft_data, output_file)
    print(f"数据已保存到 {output_file}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将BIO格式的文件转换为SFT格式")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入的BIO文件路径")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出的SFT文件路径")
    parser.add_argument("--dataset_name", type=str, default="MEDICAL", help="数据集名称")
    args = parser.parse_args()

    # 调用 bio_to_sft 函数
    bio_to_sft(args.input, args.output, args.dataset_name)
