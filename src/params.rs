use serde::{Deserialize, Serialize};

use crate::genetic::{AsContext, Genome, Mutator, OnceMutator, Predicate, Problem, Selector};

/// Declare runtime parameters
#[derive(Serialize, Deserialize)]
pub struct GenAlgParams {
	pub seed: Option<u64>, // pre-set seed
	pub pop_size: usize,
	pub num_generations: usize,
	pub mutation_rate: f64, // TODO consider
	pub elitism_rate: f64,
	pub crossover_rate: f64,
	pub enable_speciation: bool, // TODO add more?

	#[serde(skip_serializing)]
	pub log_file: Option<String>,
}
