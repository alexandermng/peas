//!
//! Evolve code that sums 3 inputs
//!

use peas::{
	genetic::{Problem, Solution},
	mutations::NeutralAddInstr,
	selection::TournamentSelection,
	Agent, WasmGABuilder, WasmGenome,
};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use walrus::{ir::UnaryOp, FunctionBuilder};
use wasmtime::{Linker, Val};

struct Sum3 {
	tests: Vec<((i32, i32, i32), i32)>, // input-output pairs
}

impl Problem for Sum3 {
	type In = (i32, i32, i32);
	type Out = i32;

	fn fitness(&self, soln: impl Solution<Sum3>) -> f64 {
		let passed = self
			.tests
			.par_iter()
			.map(|t| soln.exec(t.0) == t.1) // pass test
			.sum();
		1.0 * passed / self.tests.len()
	}
}

impl Solution<Sum3> for Agent {
	fn exec(&self, input: (i32, i32, i32)) -> i32 {
		let linker = Linker::new(&self.engine);
		let mut store = Store::new(&self.engine, ());
		let instance = linker.instantiate(&mut store, &self.module).unwrap();
		let main = instance
			.get_typed_func::<Sum3::In, Sum3::Out>(&mut store, "main")
			.unwrap();
		let (a, b, c) = input;
		let result = main
			.call(&mut store, &[Val::I32(a), Val::I32(b), Val::I32(c)])
			.unwrap();
		result[0].i32().unwrap()
	}
}

fn main() {
	let tests = vec![
		((1, 3, 3), 7),
		((2, 3, 9), 14),
		((0, 1, 0), 1),
		((0, -1, 3), 2),
	];
	let prob = Sum3 { tests };
	let seed = thread_rng().gen();
	let ga = WasmGABuilder::default()
		.problem(prob)
		.pop_size(100)
		.selection(TournamentSelection { k: 3, p: 0.9 })
		.enable_elitism(true)
		.elitism_rate(0.1)
		.enable_crossover(false)
		// .crossover_rate(0.8)
		// .crossover()
		.mutation_rate(1.0)
		.mutation(vec![
			NeutralAddLocal::with_rate(0.01), // local variable
			NeutralAddOp::with_rate(0.3),
			SwapRoot::with_rate(0.1), // consts, locals, push onto stack
			SwapOp::with_rate(0.02),
			AddTee::with_rate(0.02),
		])
		.starter(|fb: &mut FunctionBuilder| {
			// starting code for each genome
			fb.func_body().i32_const(0);
		})
		.seed(seed)
		.build();
	ga.run();
}
