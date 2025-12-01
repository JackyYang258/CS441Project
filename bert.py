from pathlib import Path

import pandas as pd
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments


def load_iclr_dataset(years=(2022, 2023)):
    """Load and merge ICLR datasets for the specified years."""
    data_dir = Path("ICLR_Dataset")
    frames = []
    for year in years:
        path = data_dir / f"ICLR{year}" / "dataset.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        yearly_df = pd.read_csv(path, sep="\t")
        yearly_df["year"] = year
        frames.append(yearly_df)
    merged = pd.concat(frames, ignore_index=True)
    merged["text"] = merged["title"].astype(str) + " [SEP] " + merged["abstract"].astype(str)
    merged["label"] = merged["decision"].apply(lambda x: 1 if "Accept" in str(x) else 0)
    return merged


# 1. Load Dataset (ICLR 2022 + 2023)
df = load_iclr_dataset()

# 2. Train / Validation / Test Split (80/10/10 with stratification)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["label"],
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1111,  # ≈10% of original after prior split
    random_state=42,
    stratify=train_val_df["label"],
)

# Convert pandas → HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# allenai/scibert_scivocab_uncased
# tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set PyTorch format
train_dataset = train_dataset.with_format("torch", columns=['input_ids','attention_mask','label'])
val_dataset = val_dataset.with_format("torch", columns=['input_ids','attention_mask','label'])
test_dataset = test_dataset.with_format("torch", columns=['input_ids','attention_mask','label'])

# 4. Load BERT Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 5. Define Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 6. Training Settings
training_args = TrainingArguments(
    output_dir="./iclr_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",            
    learning_rate=2e-5,
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,
    num_train_epochs=5,               
    weight_decay=0.01,
    fp16=True,                        # 优化：开启混合精度加速
    load_best_model_at_end=True,
    metric_for_best_model="f1",       # 优化：根据 F1 分数选最好的模型，而不是 Loss
    save_total_limit=2                # 只保存最近的2个模型，节省硬盘
)

# 7. Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 8. Train BERT
trainer.train()

# 9. Evaluate BERT
print("Final Evaluation:")
print("Validation set:")
trainer.evaluate(eval_dataset=val_dataset)
print("Test set:")
trainer.evaluate(eval_dataset=test_dataset)
