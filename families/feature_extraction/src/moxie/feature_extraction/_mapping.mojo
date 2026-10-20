"""Owned column metadata; no vectorization or hashing engine."""


def category_key(name: String, value: String, separator: String = "=") -> String:
    """Match sklearn concatenation, including intentional separator collisions."""
    return name + separator + value


struct OrderedVocabulary[Key: Copyable & Deinitable & Hashable & Equatable](Copyable, Movable):
    """Snapshot of unique keys in caller-selected column order."""
    var _keys: List[Self.Key]
    var _columns: Dict[Self.Key, Int]

    def __init__(out self, keys: List[Self.Key]) raises:
        self._keys = keys.copy()
        self._columns = Dict[Self.Key, Int]()
        for i in range(len(keys)):
            if keys[i] in self._columns:
                raise Error("ValueError: duplicate vocabulary key")
            self._columns[keys[i].copy()] = i

    def size(self) -> Int:
        return len(self._keys)

    def keys(self) -> List[Self.Key]:
        return self._keys.copy()

    def column(self, key: Self.Key) raises -> Int:
        """Return -1 for unknown keys; never assign new columns at inference."""
        if key not in self._columns:
            return -1
        return self._columns[key]


trait FeatureNames(Copyable, Movable):
    """Optional capability; hashing and image extractors need not provide it."""
    def get_feature_names_out(self) raises -> List[String]:
        ...
