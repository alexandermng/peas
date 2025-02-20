//!
//! Evolve code that sums 4 inputs
//!

use derive_more::derive::Display;
use rand::{
	distr::{Distribution, Uniform},
	Rng,
};
use serde::{Deserialize, Deserializer, Serialize};
use wasm_encoder::Instruction;

use crate::problems::{Problem, Solution};
// use rayon::prelude::*;

#[derive(Serialize, Display, Debug, Clone)]
#[display("sum4")]
pub struct Sum4 {
	pub num_tests: usize,
	pub partial1_tests_rate: f64,
	pub partial2_tests_rate: f64,
	pub partial3_tests_rate: f64,

	#[serde(skip)]
	tests: Vec<((i32, i32, i32, i32), i32)>, // input-output pairs
}

impl Sum4 {
	pub fn new(num: usize, partial1_rate: f64, partial2_rate: f64, partial3_rate: f64) -> Self {
		// const partial1_rate: f64 = 0.02;
		// const partial2_rate: f64 = 0.04;
		// const partial3_rate: f64 = 0.08;
		#[cfg(debug_assertions)]
		{
			let total_partials = partial1_rate * 4.0 + partial2_rate * 6.0 + partial3_rate * 4.0;
			assert!(total_partials < 1.0, "Test rates cannot exceed 1.0!")
		};
		let mut tests = Vec::with_capacity(num);
		let partial1_tests_num = (num as f64 * partial1_rate) as usize;
		let partial2_tests_num = (num as f64 * partial2_rate) as usize;
		let partial3_tests_num = (num as f64 * partial3_rate) as usize;
		let dist = Uniform::new(-256, 256).unwrap();
		let rng = &mut rand::rng();

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

		Self {
			num_tests: num,
			partial1_tests_rate: partial1_rate,
			partial2_tests_rate: partial2_rate,
			partial3_tests_rate: partial3_rate,
			tests,
		}
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
			.try_fold(0.0, |acc, &(args, exp)| match soln.try_exec(args) {
				Ok(o) if o == exp => Ok(acc + 1.0), // passed test
				Ok(_) => Ok(acc),                   // no pass
				Err(_) => Err(0.0),                 // if any invalid, fail early 0 fitness
			})
			.unwrap_or(0.0);
		passed / (self.tests.len() as f64)
	}

	fn name(&self) -> &'static str {
		"sum4"
	}
}

impl<'de> Deserialize<'de> for Sum4 {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		#[derive(Deserialize)]
		struct Sum4Params {
			num_tests: usize,
			partial1_tests_rate: f64,
			partial2_tests_rate: f64,
			partial3_tests_rate: f64,
		}

		let params = Sum4Params::deserialize(deserializer)?;
		Ok(Sum4::new(
			params.num_tests,
			params.partial1_tests_rate,
			params.partial2_tests_rate,
			params.partial3_tests_rate,
		))
	}
}
