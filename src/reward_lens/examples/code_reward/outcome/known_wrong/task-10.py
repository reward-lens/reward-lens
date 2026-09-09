"""Known wrong: raises ValueError instead of returning a value."""


def fib(*args, **kwargs):
    raise ValueError('not implemented yet')
