# NumPy Developers, BSD-3-Clause. See families/ensemble/LICENSES/NumPy.txt.
# PCG64 uses the MIT notice in families/ensemble/LICENSES/PCG64.txt.
"""NumPy's seeded PCG64 and boolean shuffle protocol for HGB split searches."""

from std.math import ceil


def _entropy_hash(value: UInt32, mut multiplier: UInt32) -> UInt32:
    var mixed = value ^ multiplier
    multiplier *= 0x931e8875
    mixed *= multiplier
    return mixed ^ (mixed >> 16)


def _seed_sequence(seed: UInt32) -> InlineArray[UInt64, 4]:
    # HGB draws its feature seed from a UInt32 range. SeedSequence pads that
    # single entropy word with three zeros before mixing the four-word pool.
    var pool = InlineArray[UInt32, 4](fill=0)
    var multiplier: UInt32 = 0x43b0d7e5
    pool[0] = _entropy_hash(seed, multiplier)
    for i in range(1, len(pool)):
        pool[i] = _entropy_hash(0, multiplier)
    for source in range(len(pool)):
        for dest in range(len(pool)):
            if source != dest:
                var hashed = _entropy_hash(pool[source], multiplier)
                var mixed = UInt32(0xca01f9dd) * pool[dest] - UInt32(0x4973f715) * hashed
                pool[dest] = mixed ^ (mixed >> 16)

    multiplier = 0x8b51f9dd
    var words = InlineArray[UInt64, 4](fill=0)
    for i in range(2 * len(words)):
        var word = pool[i % len(pool)] ^ multiplier
        multiplier *= 0x58f38ded
        word *= multiplier
        word ^= word >> 16
        # NumPy reads the generated UInt32 pairs in little-endian order.
        words[i // 2] |= UInt64(word) << UInt64(32 * (i % 2))
    return words^


struct _FeatureSampler(Copyable, Movable):
    var _state: UInt128
    var _increment: UInt128
    var _cached_high: Optional[UInt32]

    def __init__(out self, seed: UInt32):
        var words = _seed_sequence(seed)
        var init_state = (UInt128(words[0]) << 64) | UInt128(words[1])
        var stream = (UInt128(words[2]) << 64) | UInt128(words[3])
        self._increment = (stream << 1) | 1
        self._state = self._increment + init_state
        self._cached_high = None
        self._advance()

    def _advance(mut self):
        comptime multiplier = (UInt128(2549297995355413924) << 64) | UInt128(
            4865540595714422341
        )
        self._state = self._state * multiplier + self._increment

    def _next_uint64(mut self) -> UInt64:
        self._advance()
        var high = UInt64(self._state >> 64)
        var mixed = high ^ UInt64(self._state)
        var rotation = high >> 58
        return (mixed >> rotation) | (mixed << ((-rotation) & 63))

    def _next_uint32(mut self) -> UInt32:
        if self._cached_high:
            return self._cached_high.take()
        var bits = self._next_uint64()
        self._cached_high = UInt32(bits >> 32)
        return UInt32(bits)

    def _shuffle_index(mut self, upper: UInt64) -> Int:
        var mask = upper
        comptime for i in range(6):
            mask |= mask >> (UInt64(1) << UInt64(i))
        while True:
            var bits = (
                UInt64(self._next_uint32())
                if upper <= UInt64(0xffffffff)
                else self._next_uint64()
            )
            var index = bits & mask
            if index <= upper:
                return Int(index)

    def sample_mask(mut self, n_allowed: Int, fraction: Float64) -> List[Bool]:
        """Select features at a node for a validated fraction below one."""
        var count = max(1, Int(ceil(fraction * Float64(n_allowed))))
        var mask = List[Bool](length=n_allowed, fill=False)
        for i in range(count):
            mask[i] = True
        # Even an all-true mask consumes draws if rounding selected every feature.
        for i in range(len(mask) - 1, 0, -1):
            var j = self._shuffle_index(UInt64(i))
            var selected = mask[i]
            mask[i] = mask[j]
            mask[j] = selected
        return mask^
