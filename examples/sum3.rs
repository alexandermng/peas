//!
//! Evolve code that sums 3 inputs
//!

use std::{
	collections::HashMap,
	fs::File,
	io::{BufReader, Write},
};

use derive_more::derive::Display;
use eyre::Result;
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

#[derive(Display)]
#[display("sum3")]
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

fn main() -> Result<()> {
	pretty_env_logger::init();

	let prob = Sum3::new(1000);
	let seed: u64 = thread_rng().gen();
	let muts: Vec<WasmMutation> = vec![
		AddOperation::from_rate(0.2).into(), // local variable
		ChangeRoot::from_rate(0.3).into(),   // consts, locals, push onto stack
	];
	let init = {
		let params = &[StackValType::I32, StackValType::I32, StackValType::I32];
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
		.output_dir(format!("data/trial_{seed}.log"))
		.build();
	let mut ga = WasmGenAlg::new(params, prob);
	ga.run();

	// TODO move
	{
		// record to manifest
		let manifest_filename = "data/manifest.json";
		let mut manifest: HashMap<String, Vec<String>> = match File::open(manifest_filename) {
			Ok(f) => serde_json::from_reader(BufReader::new(f))?,
			Err(_) => HashMap::new(), // no such file
		};

		// ga.params.output_dir
		let files = manifest.entry(ga.problem.to_string()).or_default();
		files.push(ga.params.output_dir);

		let mut f = File::create(manifest_filename)?;
		let buf = serde_json::to_vec(&manifest)?;
		f.write_all(&buf)?;
	}

	Ok(())
}
