from std.sys import argv
import numojo as nm
from moxie.svm import l1_min_c

def main() raises:
    var args=argv()
    var contents: String
    with open(args[1],"r") as file:
        contents=file.read()
    var t=contents.split()
    var n=Int(t[0])
    var p=Int(t[1])
    var loss=String(t[2])
    var fit=Int(t[3])!=0
    var scale=Float64(t[4])
    var x=nm.zeros[DType.float64](nm.Shape(n,p))
    var y=List[Float64]()
    for i in range(n*p):
        x.unsafe_set(i,Float64(t[5+i]))
    for i in range(n):
        y.append(Float64(t[5+n*p+i]))
    print(l1_min_c(x,y,loss=loss,fit_intercept=fit,intercept_scaling=scale))
