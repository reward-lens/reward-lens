"""Write is_palindrome(text) returning True when text reads the same backwards, ignoring case and spaces."""


def is_palindrome(text):
    s = ''.join(c.lower() for c in text if c.isalnum())
    return s == s[::-1]
