

import torch

def compute_kl(logp, ref_logp, method="k1"):
    logr=ref_logp - logp
    if method=="k1":
        kl=-logr
    elif method=="k2":
        kl=(logr ** 2) / 2
    else:
        kl=torch.exp(logr) - logr - 1
    return kl
     
