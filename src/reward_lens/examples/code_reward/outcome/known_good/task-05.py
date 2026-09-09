"""Write fizzbuzz(n) returning 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else str(n)."""


def fizzbuzz(n):
    if n % 15 == 0:
        return 'FizzBuzz'
    if n % 3 == 0:
        return 'Fizz'
    if n % 5 == 0:
        return 'Buzz'
    return str(n)
