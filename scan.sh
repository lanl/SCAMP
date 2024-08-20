#!/bin/bash

set -e

cdat=data/posac-aho-correlators
colddat=data/posac-aho-cold-correlators
scolddat=data/posac-scalar-cold-correlators

if false; then
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
fi

if false; then
		for o in `seq 0 0.02 0.4`; do
				echo -n "$o "
				./spectral --omega $o --sigma 0.1 $cdat
		done > spectral-bounds
fi

if false; then
		{
				for T in `seq 0 10 500`; do
						echo -n "$T "
						./realtime -T $T -s 30 --skip 1  $colddat
				done
		} > rt-bounds-cold
fi

if true; then
		{
				for T in `seq 0 5 200`; do
						echo -n "$T "
						./realtime -T $T -s 10 --skip 1 $scolddat
				done
		} > rt-bounds-scalar-cold
fi
