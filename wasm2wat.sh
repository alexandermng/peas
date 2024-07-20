#!/usr/bin/bash

for dir in trial_*.log; do
	pushd $dir >/dev/null || continue
	[ "$(ls -A .)" ] || continue
	find . -name '*.wasm' \
		| xargs basename -s.wasm \
		| xargs -I_ wasm-tools print -v _.wasm -o _.wat
	popd >/dev/null
done