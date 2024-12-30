use rand::{
	distributions::{Distribution, Uniform},
	thread_rng, Rng,
};
use wasm_encoder::Instruction;

use crate::problems::{Problem, Solution};
// use rayon::prelude::*;

pub struct Polynom<const N: usize = 2> {
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
