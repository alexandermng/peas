use bon::{builder, Builder};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize, Serializer};
use std::path::Path;
use std::{fs, marker::PhantomData};
use toml;

use crate::{
	genetic::{AsContext, Genome, Mutator, OnceMutator, Predicate, Problem, Selector},
	selection::TournamentSelection,
	wasm::{mutations::WasmMutation, Context, WasmGenome},
};

/// Actual parameters for the genetic algorithm. Can be saved as an output to a file and subsequently loaded in to replicate runs.
#[derive(Serialize, Deserialize, Builder)]
#[serde(bound = "
	G: Default,
	for<'m> M: Deserialize<'m> + Serialize,
	for<'s> S: Deserialize<'s> + Serialize
")]
pub struct GenAlgParams<G = WasmGenome, C = Context, M = WasmMutation, S = TournamentSelection>
where
	G: Genome<C>,
	C: AsContext,
	M: Mutator<G, C>,
	S: Selector<G, C>,
{
	pub seed: u64, // set seed for the run
	pub pop_size: usize,
	pub num_generations: usize,
	pub max_fitness: Option<f64>,

	pub mutators: Vec<M>, // enabled mutations. to be selected from
	pub mutation_rate: f64,

	pub selector: S,

	#[serde(skip)]
	pub init_genome: G,

	pub elitism_rate: f64,
	pub crossover_rate: f64,
	pub enable_speciation: bool,

	#[serde(skip)]
	pub output_dir: String, // enclosing output directory, containing this config and other logs

	#[doc(hidden)]
	#[serde(skip)]
	#[builder(skip)]
	_ctx: PhantomData<C>,
}

/// Input options to set the parameters. Can be read from a config file.
#[derive(Deserialize)]
pub struct GenAlgParamsOpts {
	// TODO take GenAlgParams and... make everything optional. and String-like.
	pub seed: Option<u64>,
	pub pop_size: Option<usize>,
	pub num_generations: Option<usize>,

	pub mutators: Vec<String>, // TODO enrich
	pub mutation_rate: Option<f64>,

	pub selector: Option<String>, // TODO enrich

	pub init_genome: String, // filename, required

	pub elitism_rate: Option<f64>,
	pub crossover_rate: Option<f64>,
	pub enable_speciation: Option<bool>, // TODO add more?

	pub log_file: Option<String>, // location of directory
}

/// Input command-line arguments
#[derive(clap::Parser, Debug)]
pub struct GenAlgParamsCLI {
	// TODO -F / --config [filename]  (get rid of this comment and add real description)
	#[arg(short = 'F', long = "config")]
	pub config: Option<String>,
	// TODO -s / --seed [seed] (get rid of this comment and add real description)
	#[arg(short = 's', long = "seed")]
	pub seed: Option<u64>,
}

// impl GenAlgParamsOpts {
// 	fn from_file(filename: &str) -> Self {
// 		let contents = fs::read_to_string(filename).unwrap();
// 		let config = toml::from_str(&contents);
// 		return config.unwrap();
// 	}

// 	pub fn build(self) -> GenAlgParams {
// 		let seed = self.seed.unwrap_or(thread_rng().gen());
// 		let log_file = self.log_file.unwrap_or_else(|| format!("trial_{}.log", 0)); // TODO actual timestamp
// 		GenAlgParams {
// 			seed,
// 			pop_size: self.pop_size.unwrap_or(100),
// 			num_generations: self.num_generations.unwrap_or(20),
// 			mutators: self.mutators.into_iter().map(|s| ),
// 			mutation_rate: self.mutation_rate.unwrap_or(1.0),
// 			selector: self.selector,       //FIXME?
// 			init_genome: self.init_genome, //FIXME?
// 			elitism_rate: self.elitism_rate.unwrap_or(0.05),
// 			crossover_rate: self.crossover_rate.unwrap_or(0.95),
// 			enable_speciation: self.enable_speciation.unwrap_or(false),

// 			output_dir: self.log_file.unwrap(),
// 		}
// 	}
// }

// impl From<GenAlgParamsOpts> for GenAlgParams {
// 	fn from(value: GenAlgParamsOpts) -> Self {
// 		value.build()
// 	}
// }
