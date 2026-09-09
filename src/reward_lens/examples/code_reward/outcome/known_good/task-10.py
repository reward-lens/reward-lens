"""Write fib(n) returning the nth Fibonacci number with fib(0) == 0 and fib(1) == 1."""


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
