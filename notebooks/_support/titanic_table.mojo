"""Load typed Titanic columns exchanged by the mixed-types notebook."""
from moxie._categorical import categorical_column
from moxie.compose import ColumnTable
from moxie_nb_io import load, load_strings


def load_table(split: String, named: Bool) raises -> ColumnTable[]:
    var table = ColumnTable(load(split + "_numeric"), ["age", "fare"])
    var additional: List[String] = ["embarked", "sex", "pclass"]
    if split == "test":
        additional = ["pclass", "sex", "embarked"]
    for i in range(len(additional)):
        var name = additional[i]
        var position = i if split == "train" else i + 2
        var key = split + "_" + name
        if name == "pclass" and not named:
            table.add_numeric(name, load[DType.int64](key + "_integer"), position=position)
            continue
        var cells = List[Optional[String]]()
        var labels = load_strings(key)
        var missing = load[DType.bool](key + "_missing")
        for j in range(len(labels)):
            cells.append(None if missing.item(j) else Optional[String](labels[j]))
        table.add_categorical(name, categorical_column(cells), position=position)
    return table^
