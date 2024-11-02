use serde::{Deserialize, Serialize};

use crate::genetic::{AsContext, Genome, Mutator, OnceMutator, Predicate, Problem, Selector};

/// Parameters for the program. Can be output to a file.
#[derive(Serialize)]
pub struct GenAlgParams {
	pub seed: u64, // set seed for the run
	pub pop_size: usize,
	pub num_generations: usize,
	pub mutation_rate: f64, // TODO consider
	pub elitism_rate: f64,
	pub crossover_rate: f64,
	pub enable_speciation: bool, // TODO add more?

	#[serde(skip_serializing)]
	pub log_file: Option<String>,
}

/// Input options to the
#[derive(Deserialize)]
pub struct GenAlgParamsOpts {
	// TODO take GenAlgParams and... make everything optional. and String-like.
}

/// Input command-line arguments
#[derive(clap::Parser, Debug)]
pub struct GenAlgParamsCLI {
	// TODO -F / --config [filename]
	// TODO -s / --seed [seed]
}

impl From<GenAlgParamsOpts> for GenAlgParams {
	fn from(value: GenAlgParamsOpts) -> Self {
		todo!()
	}
}
