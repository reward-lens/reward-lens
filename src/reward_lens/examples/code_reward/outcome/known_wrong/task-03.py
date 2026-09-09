"""Known wrong: defines solution() rather than the entry point the task names."""


def solution(text):
    return ' '.join(reversed(text.split()))
