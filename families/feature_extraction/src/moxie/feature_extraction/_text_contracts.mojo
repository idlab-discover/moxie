"""Extension signatures only. Text and hashing engines are deferred."""


trait TextAnalyzer(Copyable, Movable):
    def analyze(self, document: String) raises -> List[String]:
        ...


trait DeterministicHasher(Copyable, Movable):
    def hash_utf8(self, key: String, seed: UInt32 = 0) raises -> Int32:
        ...


def check_ngram_range(min_n: Int, max_n: Int) raises:
    if min_n < 1 or max_n < min_n:
        raise Error("ValueError: ngram bounds must satisfy 1 <= min_n <= max_n")
