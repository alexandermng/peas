mod polynom;
mod sum3;
mod sum4;

use std::fmt::Debug;

pub use polynom::Polynom;
pub use sum3::Sum3;
pub use sum4::Sum4;

use serde::{Deserialize, Serialize};

/// Represents a task or problem to be solved by a genetic algorithm's individual/agent. Should contain
/// problem parameters and necessary training data for evaluation.
pub trait Problem {
	type In; // type of inputs/arguments to the Agent (e.g. (i32, i32) )
	type Out; // type of outputs/results from the Agent (e.g. i32 )
		   // can add stuff like externals later

	/// Calculates a Solution's fitness, defined per-problem
	fn fitness(&self, soln: impl Solution<Self>) -> f64
	where
		Self: Sized;

	fn name(&self) -> &'static str;

	// TODO: vary_params somewhere?
}

/// A solution to a given problem.
pub trait Solution<P: Problem, E: Debug = eyre::Error>: Sync {
	/// Works the problem given the input arguments, returning the output.
	/// If the solution is fallible, this will panic. Use `try_exec` instead.
	fn exec(&self, args: P::In) -> P::Out {
		self.try_exec(args).unwrap()
	}

	/// Works the problem given the input arguments, returning the output.
	/// If the solution is fallible, returns an error.
	fn try_exec(&self, args: P::In) -> Result<P::Out, E>;
}

impl<P, T, E> Solution<P, E> for &T
where
	P: Problem,
	T: Solution<P, E>,
	E: Debug,
{
	fn try_exec(&self, args: P::In) -> Result<P::Out, E> {
		(**self).try_exec(args)
	}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "name")]
pub enum ProblemSet {
	Sum3(Sum3),
	Sum4(Sum4),
	Polynom2(Polynom<2>),
}

// TODO serialize using tag = Problem::name()

impl From<Sum3> for ProblemSet {
	fn from(value: Sum3) -> Self {
		ProblemSet::Sum3(value)
	}
}

impl From<Sum4> for ProblemSet {
	fn from(value: Sum4) -> Self {
		ProblemSet::Sum4(value)
	}
}

impl From<Polynom<2>> for ProblemSet {
	fn from(value: Polynom<2>) -> Self {
		ProblemSet::Polynom2(value)
	}
}

// TODO move... idk where.
use crate::wasm::{InnovNum, StackValType, WasmGene, WasmGenome};
use wasm_encoder::Instruction;
impl ProblemSet {
	pub fn init_genome(&self) -> WasmGenome {
		match self {
			ProblemSet::Sum3(sum3) => {
				let params = &[StackValType::I32, StackValType::I32, StackValType::I32];
				let result = &[StackValType::I32];
				let mut wg = WasmGenome::new(0, params, result);
				wg.genes
					.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
				wg
			}
			ProblemSet::Sum4(sum4) => {
				let params = &[
					StackValType::I32,
					StackValType::I32,
					StackValType::I32,
					StackValType::I32,
				];
				let result = &[StackValType::I32];
				let mut wg = WasmGenome::new(0, params, result);
				wg.genes
					.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
				wg
			}
			ProblemSet::Polynom2(polynom) => {
				let params = &[StackValType::I32, StackValType::I32];
				let result = &[StackValType::I32];
				let mut wg = WasmGenome::new(0, params, result);
				wg.genes
					.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
				wg
			}
		}
	}
}
