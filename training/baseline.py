import json
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForMaskedLM, AutoTokenizer

label_words = {
    0: ["плохо", "ужасно", "отвратительно", "скучно", "разочарованием", "провалом", "кошмарно"],
    1: ["нормально", "неплохо", "средне", "терпимо", "приемлемо", "неоднозначно"],
    2: ["хорошо", "отлично", "прекрасно", "замечательно", "великолепно", "потрясающе", "восхитительно"],
}

def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("deepvk/RuModernBERT-base")
model = AutoModelForMaskedLM.from_pretrained("deepvk/RuModernBERT-base")
model = model.to(device)
model.eval()

label_word_ids = {0: [], 1: [], 2: []}
for label, words in label_words.items():
    for word in words:
        for variant in [word, " " + word]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                label_word_ids[label].append(ids[0])

rows = load_data("test.jsonl")

y_true = []
y_pred = []

for row in rows:
    prompt = row["text"] + " Это было [MASK]."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(device)

    input_ids = inputs["input_ids"][0].tolist()
    if tokenizer.mask_token_id not in input_ids:
        continue
    masked_index = input_ids.index(tokenizer.mask_token_id)

    with torch.no_grad():
        logits = model(**inputs).logits[0, masked_index]
    probs = torch.softmax(logits, dim=-1)

    scores = {}
    for label, word_ids in label_word_ids.items():
        scores[label] = sum(probs[wid].item() for wid in word_ids)

    if all(score == 0.0 for score in scores.values()):
        continue

    y_true.append(row["label"])
    y_pred.append(max(scores, key=scores.get))

print(f"Предсказано {len(y_pred)} из {len(rows)}")
print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=["Bad", "Neutral", "Good"], zero_division=0))
print("Confusion matrix [Bad, Neutral, Good]:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
