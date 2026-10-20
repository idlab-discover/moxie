"""Check the minimized weighted-relocation regression.

Run after pixi run build. Expected centers and masses come from the local Cython oracle.
"""
from pathlib import Path
import subprocess, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from oracle import load
_,lloyd,_,_=load()
x=np.array([[1.],[1.],[0.],[0.]])
w=np.array([1.,2.,1.,1.]);c=np.zeros((3,1));new=np.zeros_like(c)
mass=np.zeros(3);labels=np.zeros(4,dtype=np.int32);shift=np.zeros(3)
lloyd.lloyd_iter_chunked_dense(x,w,c,new,mass,labels,shift,1)
raw=subprocess.check_output([str(ROOT/'builds/repro_relocation')],text=True)
actual=np.array([[float(v) for v in row.split()] for row in raw.splitlines()])
print('Input X=[1,1,0,0], centers=[0,0,0], weights=[1,2,1,1]')
print('Expected masses:',mass.tolist(),'actual:',actual[:,2].tolist())
np.testing.assert_array_equal(actual[:,1],new[:,0])
np.testing.assert_array_equal(actual[:,2],mass)
