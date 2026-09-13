import re
from nltk.corpus import stopwords
from pymystem3 import Mystem

mystem = Mystem()
stop_words = set(stopwords.words("russian")) - {"не"}


def clean_text(text: str) -> str:
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
            result.append(word + "_" + tokens[i + 1])
            skip_next = True
        else:
            result.append(word)
    return result


def preprocess_text(raw_text: str) -> str:
    cleaned = clean_text(raw_text)
    lemmas = mystem.lemmatize(cleaned)
    joined = "".join(lemmas)
    tokens = joined.split()
    filtered = [w for w in tokens if w.strip() and w not in stop_words]
    filtered = apply_negation(filtered)
    return " ".join(filtered)