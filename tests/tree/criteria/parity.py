"""Differential criterion checks against exact compiled local Cython methods."""
import json,subprocess
import numpy as np
from oracle import load,ROOT

def main():
    oracle=load()
    output=subprocess.check_output([ROOT/'builds/tree/criteria/parity_driver'],text=True)
    (ROOT/'docs/private/tree/criteria/native-parity.txt').write_text(output)
    actual={}
    for line in output.splitlines():
        tag,*fields=line.split()
        prefix=2 if tag=='NODE' else 3
        actual[(tag,*map(int,fields[:prefix]))]=np.array(list(map(float,fields[prefix:])))
    failures=[]; comparisons=0
    for kind in range(5):
        for case in range(12):
            n=4+2*case; outputs=1+case%3
            nc=np.array([2+k%2 if kind<2 else 1 for k in range(outputs)],dtype=np.intp)
            y=np.array([[(i*7+k+case)%nc[k] if kind<2 else ((i*7+k*3+case)%19)/4 for k in range(outputs)] for i in range(n)],dtype=np.float64)
            weights=np.array([0 if case%4==1 and i%3==0 else 1+i%4 for i in range(n)],dtype=np.float64)
            if case==8:weights[::7]=-0.25
            samples=np.arange(n-1,-1,-1,dtype=np.intp)
            start=1 if case%2 else 0; end=n-1 if case%3 else n
            positions=[p for p in range(start+1,end) if weights[samples[start:p]].sum()>0 and weights[samples[p:end]].sum()>0]
            node,value,splits=oracle.evaluate(kind,y,weights,samples,start,end,weights.sum(),positions,nc)
            expected={('NODE',kind,case):[node]}
            expected.update({('VALUE',kind,case,j):[v] for j,v in enumerate(value.flat)})
            expected.update({('SPLIT',kind,case,p):v for p,v in zip(positions,splits)})
            for key,want in expected.items():
                comparisons+=1
                try:np.testing.assert_allclose(actual[key],want,rtol=2e-12,atol=2e-12,equal_nan=True)
                except AssertionError:failures.append({'key':key,'native':actual[key].tolist(),'oracle':np.asarray(want).tolist()})
    report={'comparisons':comparisons,'failures':failures,'tolerance':'Float64 rtol=atol=2e-12; finite bounded rational inputs; permits summation/log order only'}
    (ROOT/'docs/private/tree/criteria/parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'comparisons':comparisons,'failures':len(failures),'first':failures[:4]},indent=2))
    assert not failures,'See docs/private/tree/criteria/parity.json'
if __name__=='__main__':main()
