"""Write count_vowels(text) returning how many of a, e, i, o, u appear, ignoring case."""


def count_vowels(text):
    return sum(1 for c in text.lower() if c in 'aeiou')
