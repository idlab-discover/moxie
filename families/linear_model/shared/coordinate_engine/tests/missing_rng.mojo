"""Proposed shared API; expected compile failure until its owner supplies it."""
from moxie._random import our_rand_r

def main():
    var state = UInt32(1)
    print(our_rand_r(state))
