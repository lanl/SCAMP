#!/bin/bash

set -e

cdat=data/posac-aho-correlators

for s in 1 30; do
	for skip in 1 100; do
		{
			for T in `seq 0 10 200`; do
				echo -n "$T "
				./realtime -T $T -s $s --skip $skip $cdat
			done
		} > rt-bounds-$s-$skip
	done
done

