"""Known wrong: defines solution() rather than the entry point the task names."""


def solution(text):
    return ' '.join(w[:1].upper() + w[1:].lower() for w in text.split())
