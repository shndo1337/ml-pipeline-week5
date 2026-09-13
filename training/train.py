import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

df_train = pd.read_json("train_processed.jsonl", lines=True)
df_test = pd.read_json("test_processed.jsonl", lines=True)
df_val = pd.read_json("validation_processed.jsonl", lines=True)

df_train["text_joined"] = df_train["processed_tokens"].apply(lambda tokens: " ".join(tokens))
df_test["text_joined"] = df_test["processed_tokens"].apply(lambda tokens: " ".join(tokens))
df_val["text_joined"] = df_val["processed_tokens"].apply(lambda tokens: " ".join(tokens))

vectorizer_logreg = TfidfVectorizer(min_df=5, max_df=0.85, ngram_range=(1, 2))
X_train_logreg = vectorizer_logreg.fit_transform(df_train["text_joined"])
X_val_logreg = vectorizer_logreg.transform(df_val["text_joined"])
X_test_logreg = vectorizer_logreg.transform(df_test["text_joined"])

vectorizer_svm = TfidfVectorizer(min_df=2, max_df=0.85, ngram_range=(1, 2))
X_train_svm = vectorizer_svm.fit_transform(df_train["text_joined"])
X_val_svm = vectorizer_svm.transform(df_val["text_joined"])
X_test_svm = vectorizer_svm.transform(df_test["text_joined"])

y_train = df_train["label"]
y_val = df_val["label"]
y_test = df_test["label"]

model_logreg = LogisticRegression(C=100, class_weight='balanced', random_state=42, max_iter=1000)
model_logreg.fit(X_train_logreg, y_train)

print('Логрег — validation')
print(classification_report(y_val, model_logreg.predict(X_val_logreg), target_names=['Плохой (0)', 'Нейтральный (1)', 'Хороший (2)']))

print('Логрег — test')
print(classification_report(y_test, model_logreg.predict(X_test_logreg), target_names=['Плохой (0)', 'Нейтральный (1)', 'Хороший (2)']))

model_svm = LinearSVC(C=1, class_weight='balanced', loss='squared_hinge', random_state=42, max_iter=5000)
model_svm.fit(X_train_svm, y_train)

print('СВМ - validation')
print(classification_report(y_val, model_svm.predict(X_val_svm), target_names=['Плохой (0)', 'Нейтральный (1)', 'Хороший (2)']))

print('СВМ — test')
print(classification_report(y_test, model_svm.predict(X_test_svm), target_names=['Плохой (0)', 'Нейтральный (1)', 'Хороший (2)']))

joblib.dump(model_logreg, "model_logreg.joblib")
joblib.dump(vectorizer_logreg, "vectorizer_logreg.joblib")