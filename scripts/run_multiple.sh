#!/bin/bash

N=${1:-20}
PROBLEM=${2:-'sum4'}
export RUST_LOG=info

for i in $(seq 1 $N); do
	echo '--------------------------------'
	echo Beginning Run $i.
	cargo run -- --problem $PROBLEM --gens 300
	echo Completed Run $i.
	echo '--------------------------------'
done