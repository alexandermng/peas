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

impl Sum3 {
	pub fn new(num: usize) -> Self {
		const PARTIAL1_TESTS_RATE: f64 = 0.1;
		const PARTIAL2_TESTS_RATE: f64 = 0.2;
		let mut tests = Vec::with_capacity(num);
		let partial1_tests_num = (num as f64 * PARTIAL1_TESTS_RATE) as usize;
		let partial2_tests_num = (num as f64 * PARTIAL2_TESTS_RATE) as usize;
		let dist = Uniform::new(-256, 256);
		let rng = &mut thread_rng();

		// partial a
		tests.extend(
			(0..partial1_tests_num)
				.map(|_| (dist.sample(rng), 0, 0))
				.map(|i| (i, (i.0))),
		);

		// partial b
		tests.extend(
			(0..partial1_tests_num)
				.map(|_| (0, dist.sample(rng), 0))
				.map(|i| (i, (i.1))),
		);

		// partial c
		tests.extend(
			(0..partial1_tests_num)
				.map(|_| (0, 0, dist.sample(rng)))
				.map(|i| (i, (i.2))),
		);

		// partial a+b
		tests.extend(
			(0..partial2_tests_num)
				.map(|_| (dist.sample(rng), dist.sample(rng), 0))
				.map(|input| (input, (input.0 + input.1))),
		);

		// partial b+c
		tests.extend(
			(0..partial2_tests_num)
				.map(|_| (0, dist.sample(rng), dist.sample(rng)))
				.map(|input| (input, (input.1 + input.2))),
		);

		// partial a+c
		tests.extend(
			(0..partial2_tests_num)
				.map(|_| (dist.sample(rng), 0, dist.sample(rng)))
				.map(|input| (input, (input.0 + input.2))),
		);

		// full
		let num_left = num - tests.len();
		tests.extend(
			(0..num_left)
				.map(|_| (dist.sample(rng), dist.sample(rng), dist.sample(rng)))
				.map(|input| (input, (input.0 + input.1 + input.2))),
		);

		Self { tests }
	}
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

fn main() {
	pretty_env_logger::init();

	let prob = Sum3::new(1000);
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
		.pop_size(100)
		.generations(100)
		.stop_condition(Box::new(|ctx: &mut Context, _: &[WasmGenome]| -> bool {
			ctx.max_fitness >= 1.0
		}))
		.selection(TournamentSelection::new(0.6, 3, 0.9, false)) // can do real tournament selection when selection is fixed
		.enable_elitism(true)
		.elitism_rate(0.05)
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
