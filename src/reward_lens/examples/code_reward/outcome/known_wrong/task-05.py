"""Known wrong: divides by zero at import time, so nothing is defined."""


SCALE = 1 / 0

def fizzbuzz(n):
    if n % 15 == 0:
        return 'FizzBuzz'
    if n % 3 == 0:
        return 'Fizz'
    if n % 5 == 0:
        return 'Buzz'
    return str(n)
