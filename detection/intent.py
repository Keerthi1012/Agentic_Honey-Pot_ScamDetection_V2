import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from detection.message_scoring import spam_ml_probability

# -----------------------------
# Keyword sets (easy to extend)
# -----------------------------
confidence = 0.0
URGENCY_KEYWORDS = [
    "immediately",
    "urgent",
    "today",
    "now",
    "within",
    "24 hours",
    "limited time",
    "asap",
    "right away",
    "limited time",
    "today only",
]

THREAT_KEYWORDS = [
    "blocked",
    "suspended",
    "terminated",
    "legal action",
    "penalty",
    "frozen",
]

ACTION_KEYWORDS = ["verify", "click", "login", "pay", "transfer", "update", "confirm"]

AUTHORITY_KEYWORDS = [
    "bank",
    "government",
    "support",
    "customer care",
    "admin",
    "official",
]

SENSITIVE_INFO_KEYWORDS = ["otp", "pin", "password", "cvv", "account number", "upi"]

SUSPICIOUS_TLDS = [".xyz", ".top", ".info", ".click", ".link"]

URL_SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co"]

# -----------------------------
# Helper functions
# -----------------------------


def extract_urls(text: str) -> List[str]:
    return re.findall(r"(https?://[^\s]+)", text.lower())


def has_grammar_anomaly(text: str) -> bool:
    """
    Lightweight heuristic:
    - ALL CAPS
    - Excessive punctuation
    - Very short but urgent messages
    """
    if text.isupper():
        return True
    if "!!!" in text or "???" in text:
        return True
    if len(text.split()) < 5 and any(w in text.lower() for w in URGENCY_KEYWORDS):
        return True
    return False


# -----------------------------
# Main intent detection logic
# -----------------------------
# stage 1: Heuristic Scoring
def detect_intent(text: str, spam_model, vectorizer, session) -> Dict:
    text_lower = text.lower()
    signals = session.get("signals", [])
    suspiciousKeywords = session.get("suspiciousKeywords", [])

    confidence = session.get("confidence", 0.0)
    # 1. Urgency
    matched_keywords = [word for word in URGENCY_KEYWORDS if word in text_lower]

    if matched_keywords:
        confidence += 0.15
        signals.append("urgency")

        for kw in matched_keywords:
            if kw not in suspiciousKeywords:
                suspiciousKeywords.append(kw)

    # 2. Threat / Fear
    matched_keywords = [word for word in THREAT_KEYWORDS if word in text_lower]
    if matched_keywords:
        confidence += 0.15
        signals.append("threat")

        for kw in matched_keywords:
            if kw not in suspiciousKeywords:
                suspiciousKeywords.append(kw)

    # 3. Action demand
    matched_keywords = [word for word in ACTION_KEYWORDS if word in text_lower]
    if matched_keywords:
        confidence += 0.15
        signals.append("action_request")

        for kw in matched_keywords:
            if kw not in suspiciousKeywords:
                suspiciousKeywords.append(kw)

    # 4. Authority impersonation
    matched_keywords = [word for word in AUTHORITY_KEYWORDS if word in text_lower]
    if matched_keywords:
        confidence += 0.10
        signals.append("authority_impersonation")

        for kw in matched_keywords:
            if kw not in suspiciousKeywords:
                suspiciousKeywords.append(kw)

    # 5. Sensitive information request
    matched_keywords = [word for word in SENSITIVE_INFO_KEYWORDS if word in text_lower]
    if matched_keywords:
        confidence += 0.15
        signals.append("sensitive_info_request")

        for kw in matched_keywords:
            if kw not in suspiciousKeywords:
                suspiciousKeywords.append(kw)

    # 6. Suspicious URLs
    urls = extract_urls(text)
    for url in urls:
        if any(tld in url for tld in SUSPICIOUS_TLDS) or any(
            s in url for s in URL_SHORTENERS
        ):
            confidence += 0.15
            signals.append("suspicious_url")
            break

    # 7. Grammar / style anomaly
    if has_grammar_anomaly(text):
        confidence += 0.10
        signals.append("grammar_anomaly")

    # Cap confidence
    heuristic_score = min(confidence, 1.0)
    print(f"Heuristic Score: {heuristic_score}")

    # stage 2: Spam ML Probability
    spam_ml_risk_score = spam_ml_probability(text, spam_model, vectorizer)

    # Stage 3: Length + urgency patterns
    urgency_patterns = len(
        re.findall(r"\b(urgent|immediately|now|today)\b", text.lower())
    )
    urgency_score = min(urgency_patterns, 2) * 0.05
    msg_length_score = 1 if 20 < len(text) < 200 else 0  # Typical scam length

    # Stage 4: Calculate Risk Score
    final_risk_score = heuristic_score * 0.6 + spam_ml_risk_score * 0.4 + urgency_score

    print(f"Final_Risk_Score: {final_risk_score}")

    # Decision bucket
    if final_risk_score < 0.3:
        decision = "probe"
    elif final_risk_score < 0.85:
        decision = "extract"
    else:
        decision = "terminate"

    return {
        "decision": decision,
        "confidence": final_risk_score,
        "signals": signals,
        "suspiciousKeywords": suspiciousKeywords,
    }
