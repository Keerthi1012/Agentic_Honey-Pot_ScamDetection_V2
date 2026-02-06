import os
import re
import joblib
from fastapi import APIRouter, Request

# from detection.vector_store import search_similar, add_messages

# -----------------------------
# Load trained ML artifacts
# -----------------------------
"""BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)"""


# -----------------------------
# Default keywords
# -----------------------------
DEFAULT_SPAM_KEYWORDS = [
    "upi",
    "otp",
    "kyc",
    "sim",
    "blocked",
    "loan",
    "account",
    "call now",
    "winner",
    "congratulations",
    "credit",
    "paytm",
    "bank",
]


# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text.strip()


# -----------------------------
# Keyword score
# -----------------------------
def keyword_score(message, keywords=None):
    if keywords is None:
        keywords = DEFAULT_SPAM_KEYWORDS

    message_lower = message.lower()
    hits = sum(1 for kw in keywords if kw in message_lower)
    score = hits / len(keywords) if keywords else 0
    return score


# -----------------------------
# Extra features
# -----------------------------
def extra_features(message):
    return {
        "length": len(message),
        "uppercase_count": sum(1 for c in message if c.isupper()),
        "digit_count": sum(1 for c in message if c.isdigit()),
        "exclamation_count": message.count("!"),
        "url_count": len(re.findall(r"http\S+", message)),
    }


# -----------------------------
# Combined spam score
# -----------------------------
def combined_spam_score(
    spam_model, vectorizer, message, ml_weight=0.7, keyword_weight=0.3, keywords=None
):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    spam_prob = spam_model.predict_proba(vec)[0][1]  # ML probability
    kw_score = keyword_score(message, keywords)

    # Combine into single score
    score = ml_weight * spam_prob + keyword_weight * kw_score

    # Optional: incorporate extra features heuristically (e.g., high exclamation -> +0.05)
    features = extra_features(message)
    if features["exclamation_count"] >= 3:
        score += 0.05
    if features["url_count"] > 0:
        score += 0.05
    # Cap score at 1.0
    score = min(score, 1.0)

    return {
        "message": message,
        "spam_probability": round(spam_prob, 3),
        "keyword_score": round(kw_score, 3),
        "combined_score": round(score, 3),
        "features": features,
    }


def should_use_vector_search(score):
    return 0.4 <= score <= 0.85


def final_risk_score_cal(spam_score, vector_results):
    if not vector_results:
        return spam_score

    # max_similarity = max(r["similarity"] for r in vector_results)
    max_similarity = sum(sorted(r["similarity"] for r in vector_results)[-3:]) / 3

    # Weighted fusion
    return (0.7 * spam_score) + (0.3 * max_similarity)


def spam_ml_probability(msg: str, spam_model, vectorizer):
    spam_score_result = combined_spam_score(spam_model, vectorizer, msg)
    spam_score = spam_score_result["combined_score"]
    """
    if should_use_vector_search(spam_score):
        similar_scams = search_similar(msg)
    else:
        similar_scams = []
    """
    # final_ML_risk_score = final_risk_score_cal(spam_score,similar_scams)
    final_ML_risk_score = spam_score
    print(f"final_risk_score: {final_ML_risk_score}")
    return final_ML_risk_score

    """
    if spam_score > 0.6:
        add_messages([msg])
    """
