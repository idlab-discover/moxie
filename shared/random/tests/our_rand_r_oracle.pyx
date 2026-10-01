"""Test bridge to the unmodified pinned sklearn inline implementation."""
from sklearn.utils._random cimport our_rand_r
from sklearn.utils._typedefs cimport uint32_t


def sequence(uint32_t seed, int count):
    result = []
    cdef uint32_t value
    for _ in range(count):
        value = our_rand_r(&seed)
        result.append((value, seed))
    return result
