# Program Evolution through Augmenting Synthesis

Research into Program Synthesis based on NEAT.

## Architecture

- Wasmtime to run a given Agent against a Problem
- `WasmGenome` implements `Genome` and emits a `Agent` for evaluation
- `Agent`s implement `Solution<Problem>` (other algorithms' solutions can implement it as well, to compare results)
- `Problem`s evaluate `Solution`s and return their fitness for the task

For each example problem, we define `OurProblem` and an implementation of `Agent` as `Solution<OurProblem>`.
We then define our WasmGA with parameters. (TODO you just define GAParams and mutations and selection functions,
and it does the rest)

## TODO

- rework Solutions for RL / simulation-based fitness w/ timesteps
- store graph representation??
- consider caching things
- rayon parallelize compilation. also, need to config wasmtime pooling allocator? high-instantiation definitely

... *(just saving this here)*

```bash
ls | xargs basename -s.wasm | xargs -I_ wasm-tools print _.wasm -o _.wat
```
