"""Write factorial(n) returning n! for n >= 0."""


def factorial(n):
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out
