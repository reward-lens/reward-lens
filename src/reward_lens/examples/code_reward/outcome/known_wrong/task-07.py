"""Known wrong: raises ValueError instead of returning a value."""


def unique_sorted(*args, **kwargs):
    raise ValueError('not implemented yet')
