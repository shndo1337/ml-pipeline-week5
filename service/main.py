from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocessing import preprocess_text

app = FastAPI(title="Анализ тональности отзывов")

model = joblib.load("model_logreg.joblib")
vectorizer = joblib.load("vectorizer_logreg.joblib")
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
    processed = preprocess_text(request.text)
    X = vectorizer.transform([processed])

    pred_label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return PredictionResponse(
        label=pred_label,
        label_name=LABEL_NAMES[pred_label],
        probability=float(proba[pred_label]),
        probabilities={LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)}
    )


@app.get("/health")
def health():
    return {"status": "ok"}