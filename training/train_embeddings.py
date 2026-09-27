import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("deepvk/RuModernBERT-base")

def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


rows = load_data("train.jsonl")
val_rows = load_data("validation.jsonl")
test_rows = load_data("test.jsonl")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("deepvk/RuModernBERT-base")
model = model.to(device)
model.eval()

collator = DataCollatorWithPadding(tokenizer=tokenizer)

def get_embeddings(rows):
    encoded = [tokenizer(row.get('text', ''), truncation=True, max_length=1024) for row in rows]
    loader = DataLoader(encoded, batch_size=8, shuffle=False, collate_fn=collator)

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch)
            mask = batch["attention_mask"].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state.float() * mask).sum(dim=1)
            embeddings.append((summed / mask.sum(dim=1)).cpu().numpy())

    return np.concatenate(embeddings)


X_train = get_embeddings(rows)
X_val = get_embeddings(val_rows)
X_test = get_embeddings(test_rows)

y_train = [row['label'] for row in rows]
y_val = [row['label'] for row in val_rows]
y_test = [row['label'] for row in test_rows]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

target_names = ["Bad", "Neutral", "Good"]

for C in [0.0001, 0.001, 0.01, 0.1, 1]:
    model_logreg = LogisticRegression(C=C, class_weight='balanced', random_state=42, max_iter=2000)
    model_logreg.fit(X_train, y_train)
    print(f"Логрег C={C}: val macro_f1={f1_score(y_val, model_logreg.predict(X_val), average='macro'):.4f}")

model_logreg = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=2000)
model_logreg.fit(X_train, y_train)

print('Логрег — validation')
print(classification_report(y_val, model_logreg.predict(X_val), target_names=target_names))

print('Логрег — test')
print(classification_report(y_test, model_logreg.predict(X_test), target_names=target_names))

for C in [0.0001, 0.001, 0.01, 0.1, 1]:
    model_svm = LinearSVC(C=C, class_weight='balanced', random_state=42, max_iter=5000)
    model_svm.fit(X_train, y_train)
    print(f"СВМ C={C}: val macro_f1={f1_score(y_val, model_svm.predict(X_val), average='macro'):.4f}")

model_svm = LinearSVC(C=0.001, class_weight='balanced', random_state=42, max_iter=5000)
model_svm.fit(X_train, y_train)

print('СВМ — validation')
print(classification_report(y_val, model_svm.predict(X_val), target_names=target_names))

print('СВМ — test')
print(classification_report(y_test, model_svm.predict(X_test), target_names=target_names))
