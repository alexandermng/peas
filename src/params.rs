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

/// Input options to set the parameters. Can be read from a config file.
#[derive(Deserialize)]
pub struct GenAlgParamsOpts {
	// TODO take GenAlgParams and... make everything optional. and String-like.
	pub seed: Option<u64>,
	pub pop_size: Option<usize>,
	pub num_generations: Option<usize>,
	pub mutation_rate: Option<f64>, // TODO consider
	pub elitism_rate: Option<f64>,
	pub crossover_rate: Option<f64>,
	pub enable_speciation: Option<bool>, // TODO add more?

	pub log_file: Option<String>
}

/// Input command-line arguments
#[derive(clap::Parser, Debug)]
pub struct GenAlgParamsCLI {
	// TODO -F / --config [filename]
	pub config_filename: Option<String>,
	pub seed: Option<u64>
	// TODO -s / --seed [seed]
}

impl GenAlgParamsOpts {
	fn build(self) -> GenAlgParams {
		GenAlgParams{
			seed: self.seed.unwrap_or(0),
			pop_size: self.pop_size.unwrap_or(0),
			num_generations: self.num_generations.unwrap_or(0),
			mutation_rate: self.mutation_rate.unwrap_or(0.0),
			elitism_rate: self.elitism_rate.unwrap_or(0.0),
			crossover_rate: self.crossover_rate.unwrap_or(0.0),
			enable_speciation: self.enable_speciation.unwrap_or(false),

			log_file: self.log_file
		}
	}
}

impl From<GenAlgParamsOpts> for GenAlgParams {
	fn from(value: GenAlgParamsOpts) -> Self {
		value.build()
	}
}
