"""
Session state management for Agentic Honeypot

Currently:
- In-memory store (per process)
- Session scoped by session_id

Later:
- Replace with Redis / DB without changing controller logic
"""

from typing import Dict, Any

# -----------------------------
# In-memory session store
# -----------------------------

_sessions: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Session APIs
# -----------------------------

_sessions: Dict[str, Dict[str, Any]] = {}


def init_session(session_id: str):
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session_id": session_id,
            "messages": [],  # conversation history
            "confidence": 0.0,  # last computed confidence
            "stage": "unknown",  # probing / extraction / benign
            "intels": {},  # dict of intel: upiIds, bankAccounts, links etc.
            "signals": [],
            "suspiciousKeywords": [],
            "scam_type": None,  # OTP fraud, Lucky draw, KYC scam etc.
            "final_sent": False,
        }


def get_session(session_id: str) -> Dict[str, Any]:
    """
    Fetch session state
    """
    return _sessions.get(session_id)


def update_session(session_id: str, updates: dict):
    """
    Safely update session state.
    Handles:
    - Lists → extend or append
    - Dicts of lists → merge subkeys
    - Scalars → overwrite
    """
    if session_id not in _sessions:
        init_session(session_id)

    for key, value in updates.items():
        # Key not present → initialize appropriately
        if key not in _sessions[session_id]:
            if isinstance(value, list):
                _sessions[session_id][key] = []
            elif isinstance(value, dict):
                _sessions[session_id][key] = {k: [] for k in value.keys()}
            else:
                _sessions[session_id][key] = value

        # Merge dict of lists
        if isinstance(value, dict) and isinstance(_sessions[session_id][key], dict):
            for subkey, subval in value.items():
                if subkey not in _sessions[session_id][key]:
                    _sessions[session_id][key][subkey] = []
                if isinstance(subval, list):
                    _sessions[session_id][key][subkey].extend(subval)
                else:
                    _sessions[session_id][key][subkey].append(subval)

        # Merge lists
        elif isinstance(value, list) and isinstance(_sessions[session_id][key], list):
            _sessions[session_id][key].extend(value)

        # Append scalar to list if existing is list
        elif isinstance(_sessions[session_id][key], list):
            _sessions[session_id][key].append(value)

        # Otherwise, overwrite
        else:
            _sessions[session_id][key] = value
