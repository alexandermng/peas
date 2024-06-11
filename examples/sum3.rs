//!
//! Evolve code that sums 3 inputs
//!

use peas::{
	genetic::{Mutator, Problem, Solution},
	mutations::{NeutralAddOp, Rated, SequenceMutator, SwapRoot},
	selection::TournamentSelection,
	Context, WasmGABuilder,
};
use rand::{
	distributions::{Distribution, Uniform},
	thread_rng, Rng,
};
use walrus::FunctionBuilder;

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
		// NeutralAddLocal::with_rate(0.01), // local variable
		&Rated::new(NeutralAddOp, 0.3),
		&Rated::new(SwapRoot, 0.1), // consts, locals, push onto stack

		                            // SwapOp::with_rate(0.02),
		                            // AddTee::with_rate(0.02),
	];
	let mut ga = WasmGABuilder::default()
		.problem(prob)
		.pop_size(100)
		.generations(10)
		.selection(TournamentSelection::new(0.8, 3, 0.9, false)) // can do real tournament selection when selection is fixed
		.enable_elitism(true)
		.elitism_rate(0.05)
		.enable_crossover(false)
		// .crossover_rate(0.8)
		// .crossover()
		.mutation_rate(1.0)
		.mutation(SequenceMutator::from(&muts[..]))
		.init_genome(Box::new(|_ctx: &mut Context, fb: &mut FunctionBuilder| {
			// starting code for each genome
			fb.func_body().i32_const(1);
		}))
		.seed(seed)
		.build();
	ga.run();
}
