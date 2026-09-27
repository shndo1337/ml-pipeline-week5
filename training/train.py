import json
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import set_seed
from transformers import get_linear_schedule_with_warmup

set_seed(42)
NUM_EPOCHS = 3
ACCUM_STEPS = 8   

class ReviewsDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        label = self.rows[idx].get('label', None)
        input = self.tokenizer(
            text=self.rows[idx].get('text', ''),      
            truncation=True,       
            max_length=self.max_length,        
            return_tensors="pt"    
        )
        return {
            "input_ids": input["input_ids"].squeeze(0),
            "attention_mask": input["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }


tokenizer = AutoTokenizer.from_pretrained("deepvk/RuModernBERT-base")

def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    

rows = load_data("train.jsonl")
val_rows = load_data("validation.jsonl")
# labels = Counter([row.get('label') for row in rows])
# print(labels)

# lens = [len(tokenizer.encode(row.get('text', ''))) for row in rows]
# print(np.percentile(lens, [50, 90, 95, 99]))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    "deepvk/RuModernBERT-base",
    num_labels=3,
    id2label = {0: "Bad", 1: "Neutral", 2: "Good"},
    label2id = {"Bad": 0, "Neutral": 1, "Good": 2},
    classifier_pooling="mean",
    classifier_dropout=0.1
)
model = model.to(device)

collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(
    ReviewsDataset(rows, tokenizer, max_length=1024),
    batch_size = 4,
    shuffle= True,
    collate_fn=collator
)
val_loader = DataLoader(
    ReviewsDataset(val_rows, tokenizer, max_length=1024),
    batch_size = 4,
    shuffle= False,
    collate_fn=collator
)
decay_params = [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
no_decay_params = [p for n, p in model.named_parameters() if "bias" in n or "norm" in n]
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0}
], lr=0.00005)

total_steps = len(train_loader) * NUM_EPOCHS // ACCUM_STEPS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.06 * total_steps),
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

best_f1 = -1.0

scaler = torch.amp.GradScaler("cuda")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss_sum = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [train]")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**batch)
        loss = loss_fn(outputs.logits.float(), batch["labels"])
        train_loss_sum += loss.item()

        scaler.scale(loss / ACCUM_STEPS).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    avg_train_loss = train_loss_sum / len(train_loader)

    model.eval() 
    val_loss_sum = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): 
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [val]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch)
            val_loss_sum += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    avg_val_loss = val_loss_sum / len(val_loader)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")
    report = classification_report(
        all_labels, all_preds,
        target_names=["Bad", "Neutral", "Good"],
        zero_division=0,
        output_dict=True
    )
    print(
        classification_report(
            all_labels, all_preds,
            target_names=["Bad", "Neutral", "Good"],
            zero_division=0,
        )
    )

    macro_f1 = report["macro avg"]["f1-score"]
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        print(f"  -> новый лучший macro_f1={best_f1:.4f}, сохраняю модель в best_model")
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_model")

model = AutoModelForSequenceClassification.from_pretrained("best_model")
model = model.to(device)
model.eval()

test_rows = load_data("test.jsonl")
test_loader = DataLoader(
    ReviewsDataset(test_rows, tokenizer, max_length=1024),
    batch_size = 4,
    shuffle= False,
    collate_fn=collator
)
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

print("Test")
print(
    classification_report(
        all_labels, all_preds,
        target_names=["Bad", "Neutral", "Good"],
        zero_division=0,
    )
)
