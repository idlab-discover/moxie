"""File-driven native oracle bridge; Python is only the test harness."""
import numojo as nm
from std.sys import argv
from moxie.neighbors import KDTree

struct Reader(Movable):
    var tokens: List[String]
    var cursor: Int
    def __init__(out self, path: String) raises:
        self.tokens = List[String]()
        self.cursor = 0
        with open(path,"r") as file:
            var contents = file.read()
            for token in contents.split(): self.tokens.append(String(token))
    def number(mut self) raises -> Float64:
        var value = Float64(self.tokens[self.cursor])
        self.cursor += 1
        return value
    def matrix[dtype: DType](mut self) raises -> nm.NDArray[dtype]:
        var rows = Int(self.number())
        var cols = Int(self.number())
        var result = nm.zeros[dtype](nm.Shape(rows,cols))
        for i in range(rows*cols): result.unsafe_set(i,self.number().cast[dtype]())
        return result^

def run[dtype: DType](args: List[String]) raises:
    var input = Reader(args[1])
    var x = input.matrix[dtype]()
    var q = input.matrix[dtype]()
    var radii = nm.zeros[DType.float64](nm.Shape(Int(input.number())))
    for i in range(radii.size): radii.unsafe_set(i,input.number())
    var w = nm.zeros[dtype](nm.Shape(x.shape[0]))
    for i in range(w.size): w.unsafe_set(i,input.number().cast[dtype]())
    var weights = Optional[nm.NDArray[dtype]](None)
    if args[12]=="1": weights = w.view()
    var tree = KDTree(x,Int(args[3]),args[4],weights,p=Float64(args[5]))
    var k = Int(args[6])
    var dual = args[7]=="1"
    var breadth = args[8]=="1"
    var result = tree.query(q,k,dualtree=dual,breadth_first=breadth)
    for i in range(q.shape[0]*k): print("Q",result.indices[i],Float64(result.distances.value().unsafe_get(i)))
    var query_radii = radii.copy()
    if radii.size != 1 and radii.size != q.shape[0]: query_radii = nm.zeros[DType.float64](nm.Shape(1))
    var radius = tree.query_radius(q,query_radii,True,sort_results=True)
    ref neighbors = radius.neighbors.value()
    for offset in neighbors.offsets: print("O",offset)
    for i in range(len(neighbors.indices)): print("R",neighbors.indices[i],Float64(neighbors.distances.value().unsafe_get(i)))
    var counts = tree.query_radius(q,query_radii,count_only=True)
    for count in counts.counts.value(): print("C",count)
    var density = tree.kernel_density(q,Float64(args[10]),args[9],atol=Float64(args[13]),rtol=Float64(args[14]),breadth_first=breadth,return_log=args[11]=="1")
    for i in range(density.size): print("D",Float64(density.unsafe_get(i)))
    var correlation = tree.two_point_correlation(q,radii,dual)
    for count in correlation: print("T",count)
    var arrays = tree.get_arrays()
    for node in arrays.node_data: print("N",node.idx_start,node.idx_end,Int(node.is_leaf),node.radius)
    for i in arrays.idx_array: print("I",i)
    for i in range(arrays.node_bounds.size): print("B",Float64(arrays.node_bounds.unsafe_get(i)))

def main() raises:
    var args = List[String]()
    for arg in argv(): args.append(String(arg))
    if args[2]=="f32": run[DType.float32](args)
    else: run[DType.float64](args)
