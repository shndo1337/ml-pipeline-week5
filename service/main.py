from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

app = FastAPI(title="Анализ тональности отзывов")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("best_model_lora")
model = AutoModelForSequenceClassification.from_pretrained(
    "deepvk/RuModernBERT-base",
    num_labels=3,
    classifier_pooling="mean"
)
model = PeftModel.from_pretrained(model, "best_model_lora")
model = model.merge_and_unload()
model = model.to(device)
model.eval()
LABEL_NAMES = {0: "Плохой", 1: "Нейтральный", 2: "Хороший"}


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: int
    label_name: str
    probability: float
    probabilities: dict


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    inputs = tokenizer(request.text, truncation=True, max_length=1024, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    proba = torch.softmax(logits, dim=-1).cpu().tolist()
    pred_label = int(torch.argmax(logits))

    return PredictionResponse(
        label=pred_label,
        label_name=LABEL_NAMES[pred_label],
        probability=float(proba[pred_label]),
        probabilities={LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)}
    )


@app.get("/health")
def health():
    return {"status": "ok"}
