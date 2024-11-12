#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

with open(sys.argv[1]) as f:
    N, K = map(int, f.readline().split())
    c = np.zeros((N,K))
    t = np.zeros((K,))
    for n in range(N):
        for k in range(K):
            tau, cr, ci = map(float, f.readline().split())
            c[n,k] = cr
            t[k] = tau

plt.figure(figsize=(6,4), dpi=300)
plt.errorbar(t, np.mean(c,axis=0), yerr=np.std(c,axis=0)/np.sqrt(N))
plt.yscale('log')
plt.ylabel('Nucleon correlator')
plt.xlabel('(Imaginary) time separation')
plt.tight_layout()
plt.savefig('correlator.png')

