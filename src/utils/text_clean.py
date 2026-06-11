import re

_whitespace = re.compile(r"\s+")

def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    s = _whitespace.sub(" ", s)
    return s