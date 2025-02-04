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
#[display("polynomial")]
pub struct Polynom<const N: usize = 2> {
	pub num_tests: usize,
	pub partial1_tests_rate: f64,

	#[serde(skip)]
	tests: Vec<([i32; N], i32)>, // input-output pairs
}

impl Polynom<2> {
	/// Desired expression / solution
	pub fn expr(x: i32, y: i32) -> i32 {
		(x * x) + (x * y) + (y * y)
	}

	pub fn new(num: usize, partial1_rate: f64) -> Self {
		// const partial1_rate: f64 = 0.3;
		let mut tests = Vec::with_capacity(num);
		let partial1_tests_num = (num as f64 * partial1_rate) as usize;
		let dist = Uniform::new(-256, 256).unwrap();
		let rng = &mut rand::rng();

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

		Self {
			num_tests: num,
			partial1_tests_rate: partial1_rate,
			tests,
		}
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

	fn name(&self) -> &'static str {
		"poly2"
	}
}

impl<'de> Deserialize<'de> for Polynom<2> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		#[derive(Deserialize)]
		struct Polynom2Params {
			num_tests: usize,
			partial1_tests_rate: f64,
		}

		let params = Polynom2Params::deserialize(deserializer)?;
		Ok(Polynom::<2>::new(
			params.num_tests,
			params.partial1_tests_rate,
		))
	}
}
