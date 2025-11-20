import re

def is_safe(text: str) -> bool:
    bad_patterns = [
        r"(?i)password",
        r"(?i)api[_-]?key",
        r"(?i)credit card",
        r"(?i)ssn",
        r"(?i)violence",
        r"(?i)hate",
        r"(?i)terror",
        r"(?i)delete db"
    ]
    return not any(re.search(p, text) for p in bad_patterns)
