//!
//! Evolve code that sums 3 inputs
//!

use peas::{
	genetic::{Mutator, OnceMutator, Problem, Solution},
	selection::TournamentSelection,
	wasm::{
		mutations::{MutationLog, NeutralAddOp, SequenceMutator, SwapRoot},
		Context, WasmGenAlgBuilder, WasmGene, WasmGenome,
	},
};
use rand::{
	distributions::{Distribution, Uniform},
	thread_rng, Rng,
};
use wasm_encoder::Instruction;
// use rayon::prelude::*;

struct Sum3 {
	tests: Vec<((i32, i32, i32), i32)>, // input-output pairs
}

impl Problem for Sum3 {
	type In = (i32, i32, i32);
	type Out = i32;

	fn fitness(&self, soln: impl Solution<Sum3>) -> f64 {
		let passed: f64 = self
			.tests
			// .par_iter()
			.iter()
			.map(|t| if soln.exec(t.0) == t.1 { 1.0 } else { 0.0 }) // pass test
			.sum();
		passed / (self.tests.len() as f64)
	}
}

// impl Solution<Sum3> for Agent {
// 	fn exec(&self, input: (i32, i32, i32)) -> i32 {
// 		let linker = Linker::new(&self.engine);
// 		let mut store = Store::new(&self.engine, ());
// 		let instance = linker.instantiate(&mut store, &self.module).unwrap();
// 		let main = instance
// 			.get_typed_func::<<Sum3 as Problem>::In, <Sum3 as Problem>::Out>(&mut store, "main")
// 			.unwrap();

// 		main.call(&mut store, input).unwrap()
// 	}
// }

fn gen_tests(num: usize) -> Vec<((i32, i32, i32), i32)> {
	let rng = &mut thread_rng();
	let dist = Uniform::new(-256, 256);
	(0..num)
		.map(|_| (dist.sample(rng), dist.sample(rng), dist.sample(rng)))
		.map(|arg| (arg, (arg.0 + arg.1 + arg.2)))
		.collect()
}

fn main() {
	pretty_env_logger::init();

	let tests = gen_tests(1000);
	let prob = Sum3 { tests };
	let seed: u64 = thread_rng().gen();
	let muts: [&dyn Mutator<_, _>; 2] = [
		// for use in sequence
		&NeutralAddOp::from_rate(0.2), // local variable
		&SwapRoot::from_rate(0.3),     // consts, locals, push onto stack

		                               // NeutralAddLocal::with_rate(0.01),
		                               // SwapOp::with_rate(0.02),
		                               // AddTee::with_rate(0.02),
	];
	let init = |ctx: &mut Context, mut g: WasmGenome| -> WasmGenome {
		g.genes.push(WasmGene::new(
			Instruction::I32Const(0),
			ctx.innov(MutationLog::Unique("seed")),
		));
		g
	};
	let mut ga = WasmGenAlgBuilder::default()
		.problem(prob)
		.pop_size(250)
		.generations(100)
		.stop_condition(Box::new(|ctx: &mut Context, _: &[WasmGenome]| -> bool {
			ctx.max_fitness >= 1.0
		}))
		.selection(TournamentSelection::new(0.8, 2, 0.9, true)) // can do real tournament selection when selection is fixed
		.enable_elitism(true)
		.elitism_rate(0.02)
		.enable_crossover(true)
		.crossover_rate(0.95)
		// .crossover()
		.mutation_rate(1.0)
		.mutation(SequenceMutator::from(&muts[..]))
		.init_genome(OnceMutator::from(init))
		.seed(seed)
		.build();
	ga.run();
}
