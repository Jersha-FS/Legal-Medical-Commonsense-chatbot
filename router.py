"""Lightweight keyword router adapted from the uploaded notebook's route_query()."""

MEDICAL_TERMS = [
    "fever", "pain", "symptom", "illness", "disease", "medicine",
    "injury", "anxiety", "depression", "panic", "mental health",
    "weakness", "treatment", "doctor", "medical", "health", "stress",
]
LEGAL_TERMS = [
    "law", "legal", "rights", "police", "court", "lawyer", "complaint",
    "harassment", "crime", "arrest", "fraud", "cybercrime", "cyber crime",
    "employer", "legal action", "victim", "report", "threat", "threatening",
    "private photos", "intimate images", "share my photos", "blackmail",
    "sextortion", "without consent", "leaked photos", "online abuse",
    "cyber cell", "personal images", "revenge porn", "extortion",
]
COMMONSENSE_TERMS = [
    "feel", "feeling", "afraid", "scared", "worried", "support", "friend",
    "family", "help", "relationship", "emotion", "anxious", "cope", "lonely",
]
STRONG_LEGAL_TERMS = [
    "threatening to share", "private photos", "intimate images", "blackmail",
    "sextortion", "revenge porn", "cybercrime", "cyber crime",
]


def route_query(query: str):
    q = query.lower()
    counts = {
        "medical": sum(term in q for term in MEDICAL_TERMS),
        "legal": sum(term in q for term in LEGAL_TERMS),
        "commonsense": sum(term in q for term in COMMONSENSE_TERMS),
    }
    if any(term in q for term in STRONG_LEGAL_TERMS):
        counts["legal"] += 2
    total = sum(counts.values())
    if total == 0:
        scores = {"medical": 0.0, "legal": 0.0, "commonsense": 1.0}
        selected = ["commonsense"]
    else:
        scores = {domain: count / total for domain, count in counts.items()}
        selected = [domain for domain, count in counts.items() if count > 0]
    return scores, selected
