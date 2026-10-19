"""Reproduce stale non-root MAE impurity in the immutable sklearn oracle."""
import json
from oracle import load, ROOT
oracle=load()
root,child=oracle.mae_offset_probe()
report={'input':{'y':[0.,1.,2.,3.],'weights':[1.,1.,1.,1.],'samples':[0,1,2,3]},
        'calls':['initialize [0,4); node_impurity','initialize [1,3); node_impurity'],
        'expected':[1.,.5],'actual':[root,child],
        'cause':'MAE.node_impurity reads right_abs_errors[0]; reset fills only [start,end), leaving stale root entry 4; divides it by new node weight 2.'}
print(json.dumps(report,indent=2))
(ROOT/'docs/private/tree/criteria/mae-offset-reproducer.json').write_text(json.dumps(report,indent=2)+'\n')
assert root==1. and child==2., 'Reference changed: investigate and update consumer pins deliberately'
