import pandas as pd
from nltk.corpus import stopwords
from pymystem3 import Mystem
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\.\.\.", " МНОГОТОЧИЕ ", text)
    text = re.sub(r"!", " ВОСКЛИЦАНИЕ ", text)
    text = re.sub(r"\?", " ВОПРОС ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply_negation(tokens, negation_words={"не", "ни"}):
    result = []
    skip_next = False
    for i, word in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if word in negation_words and i + 1 < len(tokens):
            merged = word + "_" + tokens[i + 1]
            result.append(merged)
            skip_next = True
        else:
            result.append(word)
    return result

def process_file(input_path, output_path, mystem, stop_words, chunk_size=500):
    df = pd.read_json(input_path, lines=True)
    cleaned = df["text"].apply(clean_text).tolist()

    SEP = " ЖЕСТКИЙРАЗДЕЛИТЕЛЬ1337 "
    processed_texts = []

    for start in range(0, len(cleaned), chunk_size):
        chunk = cleaned[start:start + chunk_size]
        big_text = SEP.join(chunk)
        lemmas = mystem.lemmatize(big_text)
        joined = "".join(lemmas)
        docs = joined.split(SEP.strip())

        for doc in docs:
            tokens = doc.split()
            filtered_tokens = [w for w in tokens if w.strip() and w not in stop_words]
            filtered_tokens = apply_negation(filtered_tokens)
            processed_texts.append(filtered_tokens)

        print(f"[{input_path}] Обработано строк: {min(start + chunk_size, len(cleaned))}/{len(cleaned)}")

    df["processed_tokens"] = processed_texts
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Сохранено: {output_path}\n")


mystem = Mystem()
stop_words = set(stopwords.words("russian")) - {"не"}

process_file("train.jsonl", "train_processed.jsonl", mystem, stop_words)
process_file("validation.jsonl", "validation_processed.jsonl", mystem, stop_words)
process_file("test.jsonl", "test_processed.jsonl", mystem, stop_words)