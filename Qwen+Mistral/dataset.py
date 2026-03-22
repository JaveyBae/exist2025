"""
EXISTS 2024 Memes Dataset - Qwen3.5-4B 微调数据加载器
支持 Task4 (二分类) 和 Task6 (多标签分类)

数据集划分：8:1:1 (训练：验证：测试)
"""

import json
import os
import random
import csv
import ast
from PIL import Image
from torch.utils.data import Dataset, Subset
from typing import Optional, Dict, List
from sklearn.model_selection import train_test_split

# Task4 标签映射
TASK4_LABEL2ID = {"NO": 0, "YES": 1}
TASK4_ID2LABEL = {0: "NO", 1: "YES"}

# Task6 标签映射（多标签）
TASK6_LABELS = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE",
]
TASK6_LABEL2ID = {label: i for i, label in enumerate(TASK6_LABELS)}
TASK6_ID2LABEL = {i: label for label, i in TASK6_LABEL2ID.items()}


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    force_split: bool = False,
):
    """
    创建数据集划分 (8:1:1)

    Args:
        data_dir: 数据集根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        force_split: 是否强制重新划分
    """
    split_file = os.path.join(data_dir, "data_splits.json")

    # 如果已有划分文件，直接加载
    if os.path.exists(split_file) and not force_split:
        print(f"Loading existing data splits from {split_file}")
        with open(split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        return splits

    # 加载原始训练数据
    train_json = os.path.join(data_dir, "training", "EXIST2024_training.json")
    with open(train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_ids = list(data.keys())
    random.seed(seed)
    random.shuffle(all_ids)

    # 计算划分点
    n = len(all_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # 划分数据集
    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "info": {
            "total": n,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "ratio": f"{train_ratio}:{val_ratio}:{test_ratio}",
            "seed": seed
        }
    }

    # 保存划分结果
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"\n=== 数据集划分 (8:1:1) ===")
    print(f"总样本数：{n}")
    print(f"训练集：{len(train_ids)} ({len(train_ids)/n*100:.1f}%)")
    print(f"验证集：{len(val_ids)} ({len(val_ids)/n*100:.1f}%)")
    print(f"测试集：{len(test_ids)} ({len(test_ids)/n*100:.1f}%)")
    print(f"划分文件已保存至：{split_file}\n")

    return splits


def get_subset_dataset(dataset, split_ids, split_name):
    """根据 ID 列表创建子数据集"""
    # 建立 id 到索引的映射
    id_to_idx = {dataset.data[id_]["id_EXIST"]: idx for idx, id_ in enumerate(dataset.ids)}
    indices = [id_to_idx[sid] for sid in split_ids if sid in id_to_idx]
    return Subset(dataset, indices)


class EXIST2024Dataset(Dataset):
    """EXISTS 2024 Memes 数据集"""

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        processor=None,
        max_length: int = 512,
        task: str = "task4",
    ):
        """
        Args:
            data_dir: 数据集根目录
            split: "training" 或 "test"
            processor: Qwen3.5 的 processor (用于图像和文本处理)
            max_length: 最大序列长度
            task: "task4" 或 "task6" 或 "both"
        """
        self.data_dir = data_dir
        self.split = split
        self.processor = processor
        self.max_length = max_length
        self.task = task

        # 加载数据
        if split == "training":
            self.data_path = os.path.join(data_dir, "training", "EXIST2024_training.json")
            resized_dir = os.path.join(data_dir, "training", "memes_resized_256")
            self.memes_dir = resized_dir if os.path.exists(resized_dir) else os.path.join(data_dir, "training", "memes")
        else:
            self.data_path = os.path.join(data_dir, "test", "EXIST2024_test_clean.json")
            self.memes_dir = os.path.join(data_dir, "test", "memes")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.ids = list(self.data.keys())
        print(f"Loaded {len(self.ids)} samples from {split} split")

    def __len__(self):
        return len(self.ids)

    def _get_vote_label(self, labels: List[str], task: str = "task4") -> int:
        """根据标注者的投票计算最终标签"""
        if task == "task4":
            # 多数投票
            yes_count = sum(1 for l in labels if l == "YES")
            no_count = len(labels) - yes_count
            return 1 if yes_count > no_count else 0

        return 0

    def _get_task6_labels(self, labels_task6: List[List[str]]) -> List[int]:
        """处理 Task6 多标签标注"""
        positive_labels = set()
        for annotator_labels in labels_task6:
            for label in annotator_labels:
                if label != "-" and label in TASK6_LABEL2ID:
                    positive_labels.add(TASK6_LABEL2ID[label])
        return list(positive_labels)

    def _create_prompt(self, sample: Dict) -> str:
        """创建推理 prompt"""
        text = sample.get("text", "")
        lang = sample.get("lang", "es")

        if self.task == "task4":
            prompt = f"""你是一个性别歧视内容检测助手。请分析这个 meme 是否包含性别歧视内容。

Meme 文字内容（{lang}）: {text}

问题：这个 meme 是否包含性别歧视？
请只回答 YES 或 NO。"""

        elif self.task == "task6":
            prompt = f"""你是一个性别歧视内容分类助手。请分析这个 meme 包含哪些类型的性别歧视。

Meme 文字内容（{lang}）: {text}

可选的歧视类型:
- IDEOLOGICAL-INEQUALITY (意识形态不平等)
- STEREOTYPING-DOMINANCE (刻板印象与支配)
- OBJECTIFICATION (物化)
- SEXUAL-VIOLENCE (性暴力)
- MISOGYNY-NON-SEXUAL-VIOLENCE (厌女/非性暴力)

问题：这个 meme 包含哪些类型的性别歧视？（可以多选，用逗号分隔）
如果没有歧视，回答 NONE。"""

        else:  # both
            prompt = f"""你是一个性别歧视内容检测助手。请分析这个 meme。

Meme 文字内容（{lang}）: {text}

请回答两个问题：
1. 这个 meme 是否包含性别歧视？（YES/NO）
2. 如果包含，是哪些类型？（IDEOLOGICAL-INEQUALITY, STEREOTYPING-DOMINANCE, OBJECTIFICATION, SEXUAL-VIOLENCE, MISOGYNY-NON-SEXUAL-VIOLENCE）"""

        return prompt

    def __getitem__(self, idx: int):
        sample_id = self.ids[idx]
        sample = self.data[sample_id]

        # 获取图像路径
        image_path = os.path.join(self.memes_dir, sample["meme"])

        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回空白图像
            image = Image.new("RGB", (224, 224), color="white")

        # 创建 prompt
        prompt = self._create_prompt(sample)

        # 准备标签
        labels = {}

        if self.task == "task4":
            labels_task4 = sample.get("labels_task4", [])
            label = self._get_vote_label(labels_task4, "task4")
            labels["task4"] = label

        elif self.task == "task6":
            labels_task6 = sample.get("labels_task6", [[]])
            label_ids = self._get_task6_labels(labels_task6)
            # 创建多标签向量
            label_vector = [0] * len(TASK6_LABELS)
            for lid in label_ids:
                label_vector[lid] = 1
            labels["task6"] = label_vector

        else:  # both
            labels_task4 = sample.get("labels_task4", [])
            label4 = self._get_vote_label(labels_task4, "task4")
            labels["task4"] = label4

            labels_task6 = sample.get("labels_task6", [[]])
            label_ids = self._get_task6_labels(labels_task6)
            label_vector = [0] * len(TASK6_LABELS)
            for lid in label_ids:
                label_vector[lid] = 1
            labels["task6"] = label_vector

        # 使用 processor 处理图像和文本
        if self.processor is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 构建训练用的文本（包含答案）
            if self.task == "task4":
                answer = "YES" if labels["task4"] == 1 else "NO"
                messages[0]["content"].append({"type": "text", "text": f"\n答案：{answer}"})

            text = self.processor.apply_chat_template(messages, tokenize=False)

            # 处理图像和文本
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )

            # 压缩 inputs
            inputs = {k: v.squeeze(0) if isinstance(v, dict) else v for k, v in inputs.items()}
            inputs["labels"] = labels
            inputs["sample_id"] = sample_id

            return inputs

        # 返回原始数据（用于 debug）
        return {
            "image": image,
            "prompt": prompt,
            "labels": labels,
            "sample_id": sample_id
        }


class EXIST2024DatasetV2(Dataset):
    """
    简化版本 - 直接输出格式化的训练数据
    适用于 Qwen3.5-4B 的官方训练格式
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train", "val", "test"
        processor=None,
        max_length: int = 512,
        task: str = "task4",
        splits: Optional[Dict] = None,
    ):
        self.data_dir = data_dir
        self.split_name = split
        self.processor = processor
        self.max_length = max_length
        self.task = task

        # 读取训练 CSV（包含 task 标签）
        train_csv = os.path.join(data_dir, "training", "memes_training.csv")
        resized_dir = os.path.join(data_dir, "training", "memes_resized_256")
        self.memes_dir = resized_dir if os.path.exists(resized_dir) else os.path.join(data_dir, "training", "memes")
        self.full_data = {}
        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = str(row.get("id", "")).strip()
                if sample_id:
                    self.full_data[sample_id] = row

        # 获取划分
        if splits is None:
            splits = create_data_splits(data_dir)

        # 根据划分获取对应的 ID
        split_ids = splits.get(split, splits.get("train"))
        self.ids = split_ids
        self.data = {k: self.full_data[k] for k in self.ids if k in self.full_data}
        self.ids = list(self.data.keys())

        print(f"Loaded {len(self.ids)} samples from {split} split")

    def __len__(self):
        return len(self.ids)

    def _get_vote_label(self, labels: List[str]) -> int:
        """多数投票"""
        yes_count = sum(1 for l in labels if l == "YES")
        return 1 if yes_count > len(labels) / 2 else 0

    def _parse_list(self, value, default=None):
        if default is None:
            default = []
        if value is None:
            return default
        if isinstance(value, list):
            return value
        value = str(value).strip()
        if not value:
            return default
        try:
            return ast.literal_eval(value)
        except Exception:
            return default

    def _get_task4_label_from_row(self, row: Dict) -> int:
        # 训练集常用列：task1_label
        task1_label = row.get("task1_label", "")
        if str(task1_label).strip() != "":
            return int(float(task1_label))

        # 测试集常见列：task4_hard / task4_soft，格式如 [0.0, 1.0]
        for key in ("task4_hard", "task4_soft"):
            vec = self._parse_list(row.get(key), default=[])
            if isinstance(vec, list) and len(vec) >= 2:
                return 1 if float(vec[1]) >= float(vec[0]) else 0

        # 兜底：由原始标注投票
        labels_task1 = self._parse_list(row.get("labels_task1"), default=[])
        if labels_task1:
            return self._get_vote_label(labels_task1)

        return 0

    def _get_task6_label_from_row(self, row: Dict) -> List[int]:
        # 训练集常见列：task2_hard/task2_soft，长度通常为6（第0位为NONE）
        for key in ("task2_hard", "task2_soft", "task6_hard", "task6_soft"):
            vec = self._parse_list(row.get(key), default=[])
            if not isinstance(vec, list):
                continue
            if len(vec) == 6:
                # [NONE, IDEOLOGICAL-INEQUALITY, STEREOTYPING-DOMINANCE, OBJECTIFICATION, SEXUAL-VIOLENCE, MISOGYNY-NON-SEXUAL-VIOLENCE]
                return [1 if float(v) > 0 else 0 for v in vec[1:6]]
            if len(vec) == 5:
                return [1 if float(v) > 0 else 0 for v in vec]

        # 兜底：由 labels_task2 聚合
        nested_labels = self._parse_list(row.get("labels_task2"), default=[])
        result = [0] * len(TASK6_LABELS)
        if isinstance(nested_labels, list):
            for annotator_labels in nested_labels:
                if not isinstance(annotator_labels, list):
                    continue
                for label in annotator_labels:
                    if label in TASK6_LABEL2ID:
                        result[TASK6_LABEL2ID[label]] = 1
        return result

    def _resolve_image_path(self, sample_id: str) -> str:
        for ext in (".jpeg", ".jpg", ".png", ".webp"):
            p = os.path.join(self.memes_dir, f"{sample_id}{ext}")
            if os.path.exists(p):
                return p
        return os.path.join(self.memes_dir, f"{sample_id}.jpeg")

    def _get_task6_multi_label(self, labels_task6: List[List[str]]) -> List[int]:
        """获取多标签"""
        result = [0] * len(TASK6_LABELS)
        for annotator_labels in labels_task6:
            for label in annotator_labels:
                if label in TASK6_LABEL2ID:
                    result[TASK6_LABEL2ID[label]] = 1
        return result

    def __getitem__(self, idx: int):
        sample_id = self.ids[idx]
        sample = self.data[sample_id]

        # 图像路径
        image_path = self._resolve_image_path(sample_id)

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), color="gray")

        text = sample.get("text", "")

        task4_label = self._get_task4_label_from_row(sample)
        task6_multi_hot = self._get_task6_label_from_row(sample)

        # 构建对话格式
        if self.task == "task4":
            user_content = f"""分析这个 meme 图片是否包含性别歧视内容。

图片中的文字：{text}

请只回答 YES 或 NO。"""

            # 获取标签
            answer = "YES" if task4_label == 1 else "NO"
            assistant_content = answer

        elif self.task == "task6":
            user_content = f"""分析这个 meme 图片包含哪些类型的性别歧视。

可选类型:
- IDEOLOGICAL-INEQUALITY
- STEREOTYPING-DOMINANCE
- OBJECTIFICATION
- SEXUAL-VIOLENCE
- MISOGYNY-NON-SEXUAL-VIOLENCE

图片中的文字：{text}

请列出所有适用的类型，用逗号分隔。如果没有，回答 NONE。"""

            if sum(task6_multi_hot) == 0:
                assistant_content = "NONE"
            else:
                assistant_content = ", ".join([TASK6_LABELS[i] for i, v in enumerate(task6_multi_hot) if v == 1])

        else:  # both
            user_content = f"""分析这个 meme 图片：
1. 是否包含性别歧视？（YES/NO）
2. 如果包含，是哪些类型？

图片中的文字：{text}"""

            answer4 = "YES" if task4_label == 1 else "NO"

            if sum(task6_multi_hot) == 0:
                answer6 = "无"
            else:
                answer6 = ", ".join([TASK6_LABELS[i] for i, v in enumerate(task6_multi_hot) if v == 1])

            assistant_content = f"1. {answer4}\n2. {answer6}"

        # 构建 messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_content}]
            }
        ]

        return {
            "messages": messages,
            "image": image,
            "sample_id": sample_id
        }


def collate_fn(batch, processor):
    """自定义 collate function"""
    messages_list = [item["messages"] for item in batch]
    images = [item["image"] for item in batch]

    # 使用 processor 处理
    texts = [processor.apply_chat_template(msg, tokenize=False) for msg in messages_list]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # 语言模型训练标签：默认对所有 token 计算 loss，pad 位置忽略
    if "input_ids" in inputs:
        labels = inputs["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        inputs["labels"] = labels

    return inputs


if __name__ == "__main__":
    # 测试数据加载和划分
    data_dir = r"D:\exist2025\Finetune-VLM\EXIST 2024 Memes Dataset"

    # 创建数据划分
    splits = create_data_splits(data_dir, force_split=False)

    # 测试加载训练集
    print("\n=== 加载训练集 ===")
    train_dataset = EXIST2024DatasetV2(
        data_dir=data_dir,
        split="train",
        task="task4"
    )

    # 测试加载验证集
    print("\n=== 加载验证集 ===")
    val_dataset = EXIST2024DatasetV2(
        data_dir=data_dir,
        split="val",
        task="task4"
    )

    # 测试加载测试集
    print("\n=== 加载测试集 ===")
    test_dataset = EXIST2024DatasetV2(
        data_dir=data_dir,
        split="test",
        task="task4"
    )

    print(f"\n总计：{len(train_dataset) + len(val_dataset) + len(test_dataset)}")
