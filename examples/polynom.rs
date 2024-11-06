//!
//! Evolve code that sums 3 inputs
//!

use peas::{
	genetic::{Problem, Solution},
	params::GenAlgParams,
	selection::TournamentSelection,
	wasm::{
		mutations::{AddOperation, ChangeRoot, WasmMutation},
		InnovNum, StackValType, WasmGenAlg, WasmGene, WasmGenome,
	},
};
use rand::{
	distributions::{Distribution, Uniform},
	thread_rng, Rng,
};
use wasm_encoder::Instruction;
// use rayon::prelude::*;

struct Polynom<const N: usize = 2> {
	tests: Vec<([i32; N], i32)>, // input-output pairs
}

impl Polynom<2> {
	/// Desired expression / solution
	pub fn expr(x: i32, y: i32) -> i32 {
		(x * x) + (x * y) + (y * y)
	}

	pub fn new(num: usize) -> Self {
		const PARTIAL1_TESTS_RATE: f64 = 0.3;
		let mut tests = Vec::with_capacity(num);
		let partial1_tests_num = (num as f64 * PARTIAL1_TESTS_RATE) as usize;
		let dist = Uniform::new(-256, 256);
		let rng = &mut thread_rng();

		// partial x=0
		tests.extend(
			(0..partial1_tests_num)
				.map(|_| [0, dist.sample(rng)])
				.map(|i| (i, Self::expr(i[0], i[1]))),
		);

		// partial y=0
		tests.extend(
			(0..partial1_tests_num)
				.map(|_| [dist.sample(rng), 0])
				.map(|i| (i, Self::expr(i[0], i[1]))),
		);

		// full
		let num_left = num - tests.len();
		tests.extend(
			(0..num_left)
				.map(|_| [dist.sample(rng), dist.sample(rng)])
				.map(|i| (i, Self::expr(i[0], i[1]))),
		);

		Self { tests }
	}
}

impl Problem for Polynom<2> {
	type In = (i32, i32);
	type Out = i32;

	fn fitness(&self, soln: impl Solution<Polynom<2>>) -> f64 {
		let passed: f64 = self
			.tests
			// .par_iter()
			.iter()
			// .map(|&(args, exp)| if soln.exec(args) == exp { 1.0 } else { 0.0 }) // old
			.map(|&(args, exp)| match soln.try_exec(args.into()) {
				Ok(res) if res == exp => 1.0,
				Ok(_) => 0.0,
				Err(_) => -1.0,
			})
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

	let prob = Polynom::new(1000);
	let seed: u64 = thread_rng().gen();
	let muts: Vec<WasmMutation> = vec![
		AddOperation::from_rate(0.2).into(), // local variable
		ChangeRoot::from_rate(0.3).into(),   // consts, locals, push onto stack
	];
	let init = {
		let params = &[StackValType::I32, StackValType::I32];
		let result = &[StackValType::I32];
		let mut wg = WasmGenome::new(0, params, result);
		wg.genes
			.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
		wg
	};
	let params = GenAlgParams::builder()
		.seed(seed)
		.pop_size(100)
		.num_generations(100)
		.max_fitness(1.0)
		.mutators(muts)
		.mutation_rate(1.0)
		.selector(TournamentSelection::new(0.6, 3, 0.9, false)) // can do real tournament selection when selection is fixed
		.init_genome(init)
		.elitism_rate(0.05)
		.crossover_rate(0.95)
		.enable_speciation(false)
		.output_dir(format!("trial_{seed}.log"))
		.build();
	let mut ga = WasmGenAlg::new(params, prob);
	ga.run();
}
