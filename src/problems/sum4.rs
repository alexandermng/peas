//!
//! Evolve code that sums 4 inputs
//!

use rand::{
	distributions::{Distribution, Uniform},
	thread_rng, Rng,
};
use wasm_encoder::Instruction;

use crate::problems::{Problem, Solution};
// use rayon::prelude::*;

pub struct Sum4 {
	tests: Vec<((i32, i32, i32, i32), i32)>, // input-output pairs
}

impl Sum4 {
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

impl Problem for Sum4 {
	type In = (i32, i32, i32, i32);
	type Out = i32;

	fn fitness(&self, soln: impl Solution<Sum4>) -> f64 {
		let passed: f64 = self
			.tests
			// .par_iter()
			.iter()
			.map(|t| if soln.exec(t.0) == t.1 { 1.0 } else { 0.0 }) // pass test
			.sum();
		passed / (self.tests.len() as f64)
	}
}
