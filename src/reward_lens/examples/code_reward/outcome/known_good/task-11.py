"""Write gcd(a, b) returning the greatest common divisor of two positive integers."""


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
