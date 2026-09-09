"""Write max_diff(nums) returning the largest value minus the smallest, or 0 for an empty list."""


def max_diff(nums):
    if not nums:
        return 0
    return max(nums) - min(nums)
