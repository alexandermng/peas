# Program Evolution through Augmenting Synthesis

Research into Program Synthesis based on NEAT.

## Architecture

- Wasmtime to run a given Agent against a Problem
- Agents implement Genome and become a corresponding Wasm Module once built
- Agents implement Solution
- Problems evaluate Solutions and return their fitness for the task

## TODO

- Solutions
- store graph representation??
- rayon for parallelization of simulation
- consider caching things
