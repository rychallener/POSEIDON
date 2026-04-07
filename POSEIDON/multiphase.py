from __future__ import absolute_import, unicode_literals, print_function

import numpy as np

def apply_shared_params(allp, nphases, nppm,
                        shared_param_idx=None):
    '''
    Applies shared parameters to a cube by propagating shared
    parameters at supplied indices into further indices that correspond
    with other phases.

    allp: array
        pymultinest cube

    nphases: int
        number of phases in the fit

    nppm: int
        number of parameter per model (nphases * nppm = total
        number of parameters needed by the forward model(s))

    shared_param_idx: array
        indices of the parameters in the first phase that need
        to be propagated to other phases
    
    '''
    # If no parameters are shared (e.g., not a multiphase
    # retrieval), do nothing
    # (this is just a safeguard)
    if type(shared_param_idx) is type(None):
        return allp
    else:
        nshared = len(shared_param_idx)
        nnshared = nppm - nshared

        # New cube with shared parameters inserted
        cube = np.zeros(nppm * nphases)
        
        nonshared_param_idx = \
            np.delete(np.arange(nppm), shared_param_idx)

        for iphase in range(nphases):
            for idx in shared_param_idx:
                cube[nppm*iphase+idx] = allp[idx]

            for i, idx in enumerate(nonshared_param_idx):
                if iphase == 0:
                    cube[nppm*iphase+idx] = allp[idx]
                else:
                    cube[nppm*iphase+idx] = \
                        allp[(nppm+(nnshared*(iphase-1))+i)]

    return cube

def phase_cube(cube, iphase, nppm):
    '''
    Simple function to take a multiphase parameter vector
    (``cube'') and pare down to a single phase (e.g., for
    use in a forward model). The input cube must already be 
    processed by apply_shared_params().
    '''
    istart =  iphase      * nppm
    iend   = (iphase + 1) * nppm

    return cube[istart:iend]
