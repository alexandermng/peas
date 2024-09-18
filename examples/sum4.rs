//!
//! Evolve code that sums 4 inputs
//!

use peas::{
	genetic::{Mutator, Problem, Solution},
	selection::TournamentSelection,
	wasm::{
		mutations::{AddOperation, ChangeRoot, SequenceMutator},
		Context, InnovNum, StackValType, WasmGenAlgBuilder, WasmGene, WasmGenome,
	},
};
use rand::{
	distributions::{Distribution, Uniform},
	thread_rng, Rng,
};
use wasm_encoder::Instruction;
// use rayon::prelude::*;

struct Sum3 {
	tests: Vec<((i32, i32, i32, i32), i32)>, // input-output pairs
}

impl Sum3 {
	pub fn new(num: usize) -> Self {
		const PARTIAL1_TESTS_RATE: f64 = 0.02;
		const PARTIAL2_TESTS_RATE: f64 = 0.04;
		const PARTIAL3_TESTS_RATE: f64 = 0.08;
		const _: () = {
			const TOTAL_PARTIALS: f64 =
				PARTIAL1_TESTS_RATE * 4.0 + PARTIAL2_TESTS_RATE * 6.0 + PARTIAL3_TESTS_RATE * 4.0;
			assert!(TOTAL_PARTIALS < 1.0, "Test rates cannot exceed 1.0!")
		};
		let mut tests = Vec::with_capacity(num);
		let partial1_tests_num = (num as f64 * PARTIAL1_TESTS_RATE) as usize;
		let partial2_tests_num = (num as f64 * PARTIAL2_TESTS_RATE) as usize;
		let partial3_tests_num = (num as f64 * PARTIAL3_TESTS_RATE) as usize;
		let dist = Uniform::new(-256, 256);
		let rng = &mut thread_rng();

		tests.extend((0..partial1_tests_num).map(|_| (dist.sample(rng), 0, 0, 0))); // partial_a
		tests.extend((0..partial1_tests_num).map(|_| (0, dist.sample(rng), 0, 0))); // partial_b
		tests.extend((0..partial1_tests_num).map(|_| (0, 0, dist.sample(rng), 0))); // partial_c
		tests.extend((0..partial1_tests_num).map(|_| (0, 0, 0, dist.sample(rng)))); // partial_d

		tests.extend((0..partial2_tests_num).map(|_| (dist.sample(rng), dist.sample(rng), 0, 0))); // partial_ab
		tests.extend((0..partial2_tests_num).map(|_| (0, dist.sample(rng), dist.sample(rng), 0))); // partial_bc
		tests.extend((0..partial2_tests_num).map(|_| (0, 0, dist.sample(rng), dist.sample(rng)))); // partial_cd
		tests.extend((0..partial2_tests_num).map(|_| (dist.sample(rng), 0, dist.sample(rng), 0))); // partial_ac
		tests.extend((0..partial2_tests_num).map(|_| (0, dist.sample(rng), 0, dist.sample(rng)))); // partial_bd
		tests.extend((0..partial2_tests_num).map(|_| (dist.sample(rng), 0, 0, dist.sample(rng)))); // partial_ad

		tests.extend(
			(0..partial3_tests_num)
				.map(|_| (dist.sample(rng), dist.sample(rng), dist.sample(rng), 0)),
		); // partial_abc
		tests.extend(
			(0..partial3_tests_num)
				.map(|_| (0, dist.sample(rng), dist.sample(rng), dist.sample(rng))),
		); // partial_bcd
		tests.extend(
			(0..partial3_tests_num)
				.map(|_| (dist.sample(rng), 0, dist.sample(rng), dist.sample(rng))),
		); // partial_acd
		tests.extend(
			(0..partial3_tests_num)
				.map(|_| (dist.sample(rng), dist.sample(rng), 0, dist.sample(rng))),
		); // partial_abd

		// full
		let num_left = num - tests.len();
		tests.extend((0..num_left).map(|_| {
			(
				dist.sample(rng),
				dist.sample(rng),
				dist.sample(rng),
				dist.sample(rng),
			)
		})); // full

		let tests = tests
			.into_iter()
			.map(|input| (input, (input.0 + input.1 + input.2 + input.2)))
			.collect();

		Self { tests }
	}
}

impl Problem for Sum3 {
	type In = (i32, i32, i32, i32);
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
		&AddOperation::from_rate(0.2), // local variable
		&ChangeRoot::from_rate(0.3),   // consts, locals, push onto stack

		                               // NeutralAddLocal::with_rate(0.01),
		                               // SwapOp::with_rate(0.02),
		                               // AddTee::with_rate(0.02),
	];
	let init = {
		let params = &[
			StackValType::I32,
			StackValType::I32,
			StackValType::I32,
			StackValType::I32,
		];
		let result = &[StackValType::I32];
		let mut wg = WasmGenome::new(params, result);
		wg.genes
			.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
		wg
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
		.init_genome(init)
		.seed(seed)
		.build();
	ga.run();
}
