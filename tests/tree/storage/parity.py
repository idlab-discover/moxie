"""Differential fixtures evaluated by pinned Cython Tree and pure Mojo Tree."""
import json
import sys
from pathlib import Path
import numpy as np
from oracle import load, ROOT
sys.path.insert(0, str(ROOT/'tools'))
from tree_checks import compile_storage as compile, run

def number(x):
    x=float(x)
    if np.isnan(x): return '(Float64(0) / Float64(0))'
    if np.isinf(x): return ('-' if x < 0 else '')+'(Float64(1) / Float64(0))'
    return repr(x)

def main():
    oracle=load()
    rng=np.random.default_rng(173)
    functions=[]
    counts=[]
    def case(name, kinds, categories, classes, X, missing=0, seed=0, zero=False):
        n=len(kinds); nf=len(categories); no=len(classes); nc=max(classes)
        nodes=np.zeros(n,dtype=oracle.NODE_DTYPE)
        for i,k in enumerate(kinds):
            nodes[i]['left_child']=2*i+1 if k>=0 else -1
            nodes[i]['right_child']=2*i+2 if k>=0 else -1
            nodes[i]['feature']=i%nf if k>=0 else -2
            nodes[i]['threshold']=0.5 if k==0 else -2
            nodes[i]['split_kind']=k
            nodes[i]['missing_go_to_left']=missing if k>=0 else 0
            nodes[i]['n_node_samples']=n-i
            nodes[i]['weighted_n_node_samples']=0 if zero else (n-i)*0.75
            nodes[i]['impurity']=rng.random()
            if k==1: nodes[i]['left_cat_bitset']=rng.integers(0,2**32,8,dtype=np.uint32)
            if k==2: nodes[i]['left_cat_bitset'][0]=seed
        values=rng.random((n,no,nc))
        for o,c in enumerate(classes): values[:,o,c:]=0
        tree=oracle.Tree(nf,np.array(classes,dtype=np.intp),no,np.array(categories,dtype=np.intp))
        tree.__setstate__({'max_depth':3,'node_count':n,'nodes':nodes,'values':values})
        ids=tree.apply(X); pred=tree.predict(X).reshape(len(X),no,nc)
        with np.errstate(invalid='ignore',divide='ignore'):
            raw=tree.compute_feature_importances(False); norm=tree.compute_feature_importances(True)
        lines=[f'def case_{len(functions)}() raises:', '    var engine = Tree()',f'    var tree = new_tree({nf}, {classes}, {categories})']
        for i,node in enumerate(nodes):
            words=', '.join(str(int(v)) for v in node['left_cat_bitset'])
            lines += [f'    var bits{i} = CategoryBits({words})',f'    _ = engine.add_node(tree, {((i-1)//2) if i else -2}, {str(i%2==1)}, Node(-1, -1, {int(node["feature"])}, {number(node["threshold"])}, bits{i}, {number(node["impurity"])}, {int(node["n_node_samples"])}, {number(node["weighted_n_node_samples"])}, {int(node["missing_go_to_left"])}, {int(node["split_kind"])}))']
        lines += ['    var values: List[Float64] = ['+', '.join(number(v) for v in values.flat)+']','    for i in range(len(values)):', '        tree.value.unsafe_set(i, values[i])', f'    var X = zeros[DType.float32](Shape({len(X)}, {nf}))']
        if X.size:
            lines += ['    var inputs: List[Float64] = ['+', '.join(number(v) for v in X.flat)+']','    for i in range(len(inputs)):', '        X.unsafe_set(i, Float32(inputs[i]))']
        for label,expr,expected,exact in [('ids','engine.apply(tree, X)',ids,True),('prediction','engine.predict(tree, X)',pred,False),('depths','engine.compute_node_depths(tree)',tree.compute_node_depths(),True),('raw','engine.compute_feature_importances(tree, False)',raw,False),('norm','engine.compute_feature_importances(tree)',norm,False)]:
            lines += [f'    var {label} = {expr}',f'    assert_equal({label}.size, {expected.size})']
            if expected.size:
                vals=', '.join(str(int(v)) if exact else number(v) for v in expected.flat)
                lines += [f'    var expected_{label}: List[{"Int64" if exact else "Float64"}] = [{vals}]',f'    for i in range(len(expected_{label})):',f'        {"assert_equal" if exact else "close"}({label}.unsafe_get(i), expected_{label}[i])']
        functions.append('\n'.join(lines));counts.append({'name':name,'nodes':n,'rows':len(X)})
    edge=np.array([[-np.inf,0],[0.5,np.nan],[np.nextafter(np.float32(.5),np.float32(1)),1],[np.inf,-np.inf],[np.nan,np.inf],[0,np.nan]],np.float32)
    for missing in (0,1):
        case('numeric boundaries', [0,0,0,-1,-1,-1,-1],[-1,-1],[1,3],edge,missing)
        case('bitset all words',[1,-1,-1],[256],[2],np.array(list(range(256))+[np.nan],np.float32)[:,None],missing)
        for seed in (0,1,2**32-1,0xDEADBEEF):
            case('hash boundaries',[2,-1,-1],[2**24],[1],np.array(list(range(64))+[255,256,65535,2**24-1,np.nan],np.float32)[:,None],missing,seed)
    for i in range(6):
        case('random numeric',[0,0,0,-1,-1,-1,-1],[-1]*3,[1,1],rng.normal(size=(31,3)).astype(np.float32),i%2)
    case('leaf',[ -1],[-1],[1],edge[:,:1])
    case('zero root weight',[-1],[-1],[1],edge[:,:1],zero=True)
    case('empty batch',[-1],[-1],[1],np.zeros((0,1),np.float32))
    source='''from std.testing import assert_equal, assert_true
from std.math import abs
from numojo import Shape, zeros
from moxie.tree._tree import Tree, new_tree
from moxie.tree._records import Node, CategoryBits

def close(actual: Float64, expected: Float64) raises:
    if expected != expected:
        assert_true(actual != actual)
    else:
        assert_true(abs(actual - expected) <= 2e-14 * max(1.0, abs(expected)))

'''+ '\n\n'.join(functions)+'\n\ndef main() raises:\n'+''.join(f'    case_{i}()\n' for i in range(len(functions)))+'    print("PASS pinned sklearn differential fixtures")\n'
    path=ROOT/'builds/tree/storage/parity.mojo';path.write_text(source)
    compile(path,ROOT/'builds/tree/storage/parity');run(ROOT/'builds/tree/storage/parity')
    (ROOT/'docs/private/tree/storage/parity.json').write_text(json.dumps({'seed':173,'cases':counts,'tolerance':'2e-14 relative/absolute for float64; exact IDs/depths'},indent=2)+'\n')
if __name__=='__main__': main()
