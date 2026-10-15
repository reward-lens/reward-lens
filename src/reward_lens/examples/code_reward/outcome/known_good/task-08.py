"""Write title_case(text) returning the text with each word's first letter capitalised."""


def title_case(text):
    return ' '.join(w[:1].upper() + w[1:].lower() for w in text.split())
