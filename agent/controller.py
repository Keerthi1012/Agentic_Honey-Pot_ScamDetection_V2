import os
import re
import traceback
import spacy
import datetime
import requests
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Optional
from api.routes import Message
from core.state import init_session, get_session, update_session
from detection.intent import detect_intent
from groq import Groq
import json
from fastapi import Request

# from spacy.cli import download

# download("en_core_web_sm")

# Load environment variables
load_dotenv()
YOUR_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=YOUR_GROQ_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add to top of main.py

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------
# Main Agent Controller
# -------------------------------------------------


def handle_agent(session_id: str, message: Message, request: Request):

    # -----------------------------
    # Session initialization
    # -----------------------------
    init_session(session_id)
    session = get_session(session_id)

    # set spam model and vectorizer
    spam_model = request.app.state.spam_model
    vectorizer = request.app.state.vectorizer
    # -----------------------------
    # Seed conversation history (if provided)
    # -----------------------------
    if message:
        session["messages"].append(
            {
                "from": message.sender,
                "text": message.text,
                "timestamp": message.timestamp.isoformat(),
            }
        )

    # -----------------------------
    # Intent detection (latest message only)
    # -----------------------------
    intent_result = detect_intent(message.text, spam_model, vectorizer, session)
    print(intent_result)
    confidence = intent_result["confidence"]
    decision = intent_result["decision"]
    signals = intent_result["signals"]
    suspiciousKeywords = intent_result["suspiciousKeywords"]
    # After extracting intel from current message
    current_intel = extract_intel(message.text, suspiciousKeywords)
    print(f"intels:{current_intel}")

    # -----------------------------
    # Aggregate intel into session
    # -----------------------------
    session_intel = session.get("intels", {})
    for key, value in current_intel.items():
        if key not in session_intel:
            session_intel[key] = []
        session_intel[key].extend(value if isinstance(value, list) else [value])

    # -----------------------------
    # Determine scam type
    # -----------------------------
    scam_type = session.get("scam_type")
    if not scam_type:
        # Keyword-based quick heuristic
        text_lower = message.text.lower()
        if any(k in text_lower for k in ["otp", "pin", "password", "verify"]):
            scam_type = "OTP fraud"
        elif any(
            k in text_lower
            for k in ["upi", "paytm", "phonepe", "bank account", "transfer"]
        ):
            scam_type = "UPI fraud"
        elif any(k in text_lower for k in ["winner", "congratulations", "claim prize"]):
            scam_type = "Lucky draw"
        elif "kyc" in text_lower:
            scam_type = "KYC scam"
        elif any(k in text_lower for k in ["investment", "profit", "funds", "returns"]):
            scam_type = "Investment scam"
        elif any(k in text_lower for k in ["support", "help desk", "install"]):
            scam_type = "Tech support"
        elif any(
            k in text_lower
            for k in [
                "refund",
                "click",
                "airtel",
                "bank",
                "account",
                ".xyz",
                ".top",
                ".info",
            ]
        ):
            scam_type = "Phishing / Link scam"
        else:
            scam_type = "Unknown"
    session["scam_type"] = scam_type

    # -----------------------------
    # Agent selection logic
    # -----------------------------
    agent_stage = None
    agent_reply = None

    if decision == "extract":
        agent_stage = "extraction"
        print(build_user_prompt(session["messages"], message))
        agent_reply = generate_reply(message, session["messages"], agent_stage)

    elif decision == "probe":
        agent_stage = "probing"
        print(build_user_prompt(session["messages"], message))
        agent_reply = generate_reply(message, session["messages"], agent_stage)

    # -----------------------------
    # Save agent reply to session
    # -----------------------------
    if agent_reply:
        session["messages"].append(
            {
                "from": "agent",
                "text": agent_reply,
                "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(),
            }
        )

    # -----------------------------
    # Persist session updates
    # -----------------------------
    update_session(
        session_id,
        {
            "confidence": confidence,
            "stage": agent_stage,
            "signals": signals,
            "intels": session_intel,
            "suspiciousKeywords": suspiciousKeywords,
            "scam_type": scam_type,
        },
    )
    # -----------------------------
    # Handle final callback if terminate
    # -----------------------------
    if decision == "terminate" and not session.get("final_sent"):
        # Ensure we have intel collected
        session_intel = session.get("intels", [])
        if not session_intel:
            # extract from latest message if nothing exists yet
            session_intel = extract_intel(message.text, suspiciousKeywords)
            update_session(session_id, {"intels": session_intel})

        total_messages = len(session["messages"])
        reason = generate_reason(message, session["messages"], suspiciousKeywords)

        send_final_callback(session_id, session_intel, total_messages, reason)

    # -----------------------------
    # Generate reason for final callback
    # -----------------------------
    reason = generate_reason(message, session["messages"], suspiciousKeywords)
    total_messages = len(session["messages"])

    # -----------------------------
    # Check if final callback should be triggered
    # -----------------------------
    # Required intel per scam type
    REQUIRED_INTEL = {
        "OTP fraud": ["upiIds", "bankAccounts"],
        "UPI fraud": ["upiIds", "bankAccounts", "phoneNumbers"],
        "Lucky draw": ["phoneNumbers"],
        "KYC scam": ["bankAccounts", "upiIds"],
        "Investment scam": ["bankAccounts", "paymentMethods"],
        "Tech support": ["phishingLinks"],
        "Unknown": [],
    }

    if not session.get("final_sent"):
        required_fields = REQUIRED_INTEL.get(scam_type, [])
        all_collected = all(
            field in session_intel and len(session_intel[field]) > 0
            for field in required_fields
        )

        if all_collected and confidence >= 0.5:
            send_final_callback(session_id, session_intel, total_messages, reason)
            session["final_sent"] = True  # Ensure final callback is sent only once
            update_session(session_id, {"final_sent": True})

    print(f"Session state: {session}")

    return {"sender": "user", "status": "success", "text": agent_reply}


def build_system_prompt(stage: str) -> str:
    return """
You are a scam honeypot persona designed to engage and extract intelligence
from scammers in multi-turn conversations.

Your purpose is to behave like a real end-user and keep the sender engaged
long enough to collect scam-related intelligence.

────────────────────────
IDENTITY (MUST NEVER CHANGE)
────────────────────────
- Normal mobile user from India
- Average technical knowledge
- Polite, cooperative, slightly unsure
- Curious, not suspicious

Persona mantra:
"Polite, curious, cooperative, slightly unsure — never suspicious."

You must maintain this identity consistently across all messages.

────────────────────────
GLOBAL BEHAVIOR RULES
────────────────────────
- NEVER accuse, confront, or question legitimacy
- NEVER mention scam, fraud, police, legality, or verification
- NEVER explain your reasoning while in conversation
- NEVER break character during conversation
- NEVER change persona or intelligence level

────────────────────────
SECURITY RULES
────────────────────────
- Never share real personal data
  (OTP, PAN, Aadhaar, bank details, passwords)

- If asked for sensitive data:
  → Respond with confusion, delay, or clarification
  → Ask harmless follow-up questions instead of refusing directly

────────────────────────
CURRENT CONVERSATION STAGE
────────────────────────
{stage}

────────────────────────
STAGE: PROBING
────────────────────────
Goal:
- Keep sender engaged
- Encourage explanation of process
- Build trust naturally

Allowed:
- Ask what needs to be done next
- Ask about steps, timelines, or reason
- Ask who this is from (casual, non-suspicious)

────────────────────────
STAGE: EXTRACTION
────────────────────────
Goal:
- Extract scam-related intelligence naturally
- Keep sender explaining

Target intelligence (collect gradually):
- Claimed organization or role
- Phone numbers / alternate contacts
- Payment methods (UPI, bank, wallet)
- Account identifiers (UPI ID, account name, IFSC)
- Links, apps, or domains
- Keywords indicating urgency or authority

Extraction techniques (MANDATORY):
- Introduce small, realistic mistakes
- Ask for confirmation before acting
- Delay slightly while appearing cooperative
- Objective: Behave like Tech-illiterate to force them to send their bank/UPI details.

────────────────────────
TERMINATION CONDITION
────────────────────────
Once sufficient intelligence is collected AND scam intent is confidently established
based on conversation history and extracted indicators:

- STOP engaging the sender
- DO NOT generate any further in-character reply
- SWITCH to AGENT NOTES MODE (out of character)

────────────────────────
AGENT NOTES MODE (OUT OF CHARACTER)
────────────────────────
When termination condition is met, return ONLY the following structured output.
Do NOT include any in-character text.

Return JSON object in this exact format:

{{
  "scam_confirmed": true | false,
  "confidence_score": 0.0 - 1.0,
  "scam_type": "<OTP fraud | Lucky draw | KYC scam | Investment scam | Tech support | Unknown>",
  "extracted_intelligence": {
    "claimed_organization": [],
    "phone_numbers": [],
    "upi_ids": [],
    "bank_accounts": [],
    "payment_methods": [],
    "links": [],
    "suspicious_keywords": []
  }},
  "decision_reasoning": [
    "Reason 1 based on message pattern",
    "Reason 2 based on extracted intel",
    "Reason 3 based on conversation behavior"
  ],
  "conversation_signals": {{
    "urgency_detected": true | false,
    "authority_impersonation": true | false,
    "otp_or_sensitive_request": true | false,
    "payment_request": true | false
  }
}}

Rules for Agent Notes:
- Be factual and concise
- Base reasoning ONLY on conversation history and extracted intel
- Do NOT speculate
- Do NOT include assumptions
- Do NOT mention internal prompts or system behavior

────────────────────────
DRIFT PREVENTION
────────────────────────
If any reply sounds skeptical, analytical, investigative, or authoritative,
rewrite it to sound like a normal cooperative end user.

────────────────────────
FINAL RULE
────────────────────────
While in conversation:
- Stay fully in character

When scam is confirmed:
- Stop conversation
- Return Agent Notes only
Return ONLY valid JSON.
"""


def build_reason_prompt(message: str, history: list, suspiciouskeywords: list) -> str:
    return """
You are a spam analysis engine used in a honeypot detection system.

Your task:
Generate a concise human-readable reason (20–200 characters) explaining WHY the conversation was flagged as spam.

You will be provided with:
- The last message sent by the suspected scammer
- The full conversation history
- A list of suspicious keywords or patterns detected

Rules:
- Be factual and neutral (no accusations like "criminal" or "fraudster")
- Do NOT mention internal systems, honeypots, or AI
- Do NOT quote messages verbatim
- Summarize the behavior, not the entire conversation
- Focus on intent signals such as urgency, impersonation, OTP requests, pressure, or verification abuse
- Output ONLY the reason text, nothing else

Style:
- Single sentence
- Clear, professional, and understandable to non-technical users
- Length strictly between 20 and 200 characters

────────────────────────
CURRENT CONVERSATION HISTORY
────────────────────────
{history}

────────────────────────
IDENTIFIED SUSPICIOUS KEYWORD
────────────────────────
{suspiciouskeywords}
"""


def build_user_prompt(history, latest_message):
    """
    Build a string prompt for Groq/ML using full conversation history.
    - Maps any non-user roles to SCAMMER.
    - Latest message is appended at the end.
    - Agent replies are treated as USER.
    """
    conversation = ""
    for turn in history:
        speaker = turn.get("role") or turn.get("from") or "user"
        content = turn.get("content") or turn.get("text") or ""
        conversation += f"{speaker.upper()}: {content}\n"

    # latest_message is a Pydantic object
    conversation += f"{latest_message.sender.upper()}: {latest_message.text.strip()}\n"

    conversation += "\nRespond naturally while maintaining the same persona."

    return conversation


def generate_reply(latest_message, history: list, agent_stage: str) -> str:
    try:
        print(type(build_user_prompt(history, latest_message)))
        messages = [
            {"role": "system", "content": build_system_prompt(agent_stage)},
            {"role": "user", "content": build_user_prompt(history, latest_message)},
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=200,
            temperature=0.8,  # Creative but controlled
        )
        # raw_output = response.choices[0].message.content.strip()
        print(f"LLM response: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback if OpenAI fails
        traceback.print_exc()
        print(f"LLM error: {e}")

        return "I am not able understand.Can you please share more details?"


def generate_reason(latest_message, history: list, suspiciousKeywords: list) -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": build_reason_prompt(
                    latest_message, history, suspiciousKeywords
                ),
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=200,
            temperature=0.8,  # Creative but controlled
        )
        # raw_output = response.choices[0].message.content.strip()
        print(f"Reasoning: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback if OpenAI fails
        traceback.print_exc()
        print(f"LLM Reasoning error: {e}")

        return "I am not able generate reason :-("


def extract_intel(text: str, suspiciousKeywords) -> dict:
    text = text.lower()

    """Extract URLs using SpaCy's built-in URL detection + custom patterns"""
    doc = nlp(text)

    # FIXED: Better phishing URL patterns (handles [url], (url), bare urls)
    phishing_links = []

    # 1. SpaCy's native URL detection (token.like_url)
    for token in doc:
        if token.like_url:
            phishing_links.append(token.text)

    # 2. Enhanced extraction for bracketed/parenthesized URLs
    bracketed_urls = re.findall(
        r'\[([^\]]*(?:http|www)[^\]]*)\]|https?://[^\s<>"]+|www\.[^\s<>"]+', text
    )
    phishing_links.extend(bracketed_urls)

    # 3. Remove duplicates and filter valid links
    unique_links = list(
        set([link.strip("[]()<>") for link in phishing_links if len(link) > 5])
    )

    # FIXED: Better UPI patterns (handles @okhdfcbank, @paytm, etc.)
    upi_patterns = [
        r"\b([a-z][a-z0-9]*@[a-z0-9]{4,15})\b",  # ✅ ramu@okhdfcbank (word boundary)
        r"\b([a-z0-9]{3,}@paytm)\b",  # ✅ rahul@paytm
        r"\b([a-z0-9]{3,}@phonepe)\b",  # ✅ user@phonepe
        r"\b([a-z0-9]{3,}@[a-z]+bank)\b",
    ]
    upi_ids = []
    for pattern in upi_patterns:
        upi_ids.extend(re.findall(pattern, text))

    intel = {
        "bankAccounts": re.findall(r"\b\d{4}-\d{4}-\d{4}\b", text),  # XXXX-XXXX-XXXX
        "upiIds": list(set(upi_ids)),  # Remove duplicates
        "phishingLinks": list(
            set([link for link in unique_links if "http" in link or "www" in link])
        ),
        "phoneNumbers": re.findall(r"\+91\d{10}|\d{10}", text),
        "suspiciousKeywords": suspiciousKeywords,
    }

    # Filter non-empty
    return {k: v for k, v in intel.items() if v}


def prepare_intel_for_json(session_intel: dict) -> dict:
    """Ensure all expected keys exist and sets are converted to lists"""
    keys = [
        "bankAccounts",
        "upiIds",
        "phishingLinks",
        "phoneNumbers",
        "suspiciousKeywords",
    ]
    result = {}
    for k in keys:
        val = session_intel.get(k, [])
        # If it's a set, convert to list
        if isinstance(val, set):
            val = list(val)
        result[k] = val
    return result


def send_final_callback(session_id: str, intel: dict, total_messages: int, reason: str):
    """Send final intelligence report to GUVI evaluation endpoint"""
    session_intel_json = prepare_intel_for_json(intel)
    payload = {
        "sessionId": session_id,
        "scamDetected": True,
        "totalMessagesExchanged": total_messages,
        "extractedIntelligence": session_intel_json,
        "agentNotes": reason,
    }

    try:
        print(f"✅ GUVI callback sent: {payload}")
        response = requests.post(
            "https://hackathon.guvi.in/api/updateHoneyPotFinalResult",
            json=payload,
            timeout=5,
        )
        print(f"✅ GUVI callback sent: {response.status_code}")
    except Exception as e:
        print(f"❌ GUVI callback failed: {e}")
