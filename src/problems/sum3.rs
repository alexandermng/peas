use std::{
	collections::HashMap,
	fs::File,
	io::{BufReader, Write},
};

use derive_more::derive::Display;
use eyre::Result;
use rand::{
	distr::{Distribution, Uniform},
	Rng,
};
use serde::{Deserialize, Deserializer, Serialize};
use wasm_encoder::Instruction;

use crate::problems::{Problem, Solution};
// use rayon::prelude::*;

#[derive(Serialize, Display, Debug, Clone)]
#[display("sum3")]
pub struct Sum3 {
	pub num_tests: usize,
	pub partial1_tests_rate: f64,
	pub partial2_tests_rate: f64,

	#[serde(skip)]
	tests: Vec<((i32, i32, i32), i32)>, // input-output pairs
}

impl Sum3 {
	pub fn new(num: usize, partial1_rate: f64, partial2_rate: f64) -> Self {
		// const partial1_rate: f64 = 0.1; // TODO find a way to default these
		// const partial2_rate: f64 = 0.2;
		let mut tests = Vec::with_capacity(num);
		let partial1_tests_num = (num as f64 * partial1_rate) as usize;
		let partial2_tests_num = (num as f64 * partial2_rate) as usize;
		let dist = Uniform::new(-256, 256).unwrap();
		let rng = &mut rand::rng();

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

		Self {
			num_tests: num,
			partial1_tests_rate: partial1_rate,
			partial2_tests_rate: partial2_rate,
			tests,
		}
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
			.try_fold(0.0, |acc, &(args, exp)| match soln.try_exec(args) {
				Ok(o) if o == exp => Ok(acc + 1.0), // passed test
				Ok(_) => Ok(acc),                   // no pass
				Err(_) => Err(0.0),                 // if any invalid, fail early 0 fitness
			})
			.unwrap_or(0.0);
		passed / (self.tests.len() as f64)
	}

	fn name(&self) -> &'static str {
		"sum3"
	}
}

impl<'de> Deserialize<'de> for Sum3 {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		#[derive(Deserialize)]
		struct Sum3Params {
			num_tests: usize,
			partial1_tests_rate: f64,
			partial2_tests_rate: f64,
		}

		let params = Sum3Params::deserialize(deserializer)?;
		Ok(Sum3::new(
			params.num_tests,
			params.partial1_tests_rate,
			params.partial2_tests_rate,
		))
	}
}
