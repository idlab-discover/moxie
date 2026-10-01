"""Context baseline: Python dispatch plus NumPy operations, single BLAS thread."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pyperf


def run(x,mode):
    x[0,0]=(float(x[0,0])+0.125)%0.875
    if mode=='norm':return np.einsum('ij,ij->i',x,x)
    if mode=='center':return x-x.mean(axis=0)
    if mode=='matmul':return x@x.T

if __name__=='__main__':
 runner=pyperf.Runner()
 runner.metadata.update({'numpy':np.__version__,'timing':'Python dispatch + NumPy call; not native-region protocol','BLAS_threads':1,'seed':'deterministic modular integers'})
 for dtype in (np.float32,np.float64):
  for rows,cols in ((2,4),(64,32),(256,256)):
   for layout in ('C','S'):
    owner=((np.arange(rows)[:,None]*17+np.arange(cols*2)[None,:]*3)%31-15).astype(dtype)/8
    x=owner[:,::2]
    if layout=='C':x=np.ascontiguousarray(x)
    for mode in ('norm','center','matmul'):
     runner.bench_func(f'{np.dtype(dtype).name}/{rows}x{cols}/{layout}/{mode}',run,x,mode)
