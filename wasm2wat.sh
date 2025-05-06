#!/usr/bin/bash

dir=${1:-"data/sum3"}

pushd $dir >/dev/null || continue
[ "$(ls -A .)" ] || continue
find . -name '*.wasm' \
	| xargs basename -s.wasm \
	| xargs -I_ wasm-tools print -v _.wasm -o _.wat
popd >/dev/null