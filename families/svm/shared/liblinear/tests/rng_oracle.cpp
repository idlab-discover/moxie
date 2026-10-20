// Validation only: execute the pinned header's actual bounded sampler.
#include "newrand.h"
#include <iostream>
int main() {
    for(unsigned seed: {0u,42u,4294967295u}) {
        set_seed(seed);
        for(unsigned range: {1u,2u,3u,2147483649u,4294967295u})
            for(int i=0;i<1000;++i) std::cout<<bounded_rand_int(range)<<'\n';
        std::cout<<mt_rand()<<'\n';
    }
}
