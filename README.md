# Program Evolution through Augmenting Synthesis

Research into Program Synthesis based on NEAT.

## Architecture

- Wasmtime to run a given Agent against a Problem
- `WasmGenome` implements `Genome` and emits a `Agent` for evaluation
- `Agent`s implement `Solution<Problem>` (other algorithms' solutions can implement it as well, to compare results)
- `Problem`s evaluate `Solution`s and return their fitness for the task

For each example problem, we define `OurProblem` and implement it as a `Problem`. The WasmGA does the rest.

## TODO

- fix mutation rates
- fix selection
- uhh... historical markers (!!)
- rework Solutions for RL / simulation-based fitness w/ timesteps
- store graph representation?? how to detect valid
- consider caching things
- rayon parallelize compilation. also, need to config wasmtime pooling allocator? high-instantiation definitely
- impl Serialize for WasmGenome, for logging
