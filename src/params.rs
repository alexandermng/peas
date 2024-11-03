use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::fs;
use toml;
use std::path::Path;

use crate::{
	genetic::{AsContext, Genome, Mutator, OnceMutator, Predicate, Problem, Selector},
	wasm::{mutations::WasmMutation, Context, WasmGenome},
};

/// Actual parameters for the genetic algorithm. Can be saved as an output to a file and subsequently loaded in to replicate runs.
#[derive(Serialize)]
pub struct GenAlgParams<G = WasmGenome, C = Context, M = WasmMutation>
where
	G: Genome<C>,
	C: AsContext,
	M: Mutator<G, C>,
{
	pub seed: u64, // set seed for the run
	pub pop_size: usize,
	pub num_generations: usize,

	pub mutators: Vec<M>,
	pub mutation_rate: f64, // TODO consider

	#[serde(skip_serializing)] // TODO serialize_with just name
	pub selector: Box<dyn Selector<G, C>>, // includes rate

	#[serde(skip_serializing)] // TODO serialize_with just name
	pub init_genome: G,

	pub elitism_rate: f64,
	pub crossover_rate: f64,
	pub enable_speciation: bool, // TODO add more?

	#[serde(skip_serializing)]
	pub log_file: String,
}

/// Input options to set the parameters. Can be read from a config file.
#[derive(Deserialize, Debug)]
pub struct GenAlgParamsOpts<G = WasmGenome, C = Context, M = WasmMutation> 
where 
G: Genome<C>,
C: AsContext,
M: Mutator<G, C>,
{
	// TODO take GenAlgParams and... make everything optional. and String-like.
	pub seed: Option<u64>,
	pub pop_size: Option<usize>,
	pub num_generations: Option<usize>,

	pub mutators: Vec<M>,
	pub mutation_rate: Option<f64>, // TODO consider

	#[serde[skip_deserializing]]
	//#[debug(skip)]
	pub selector: Box<dyn Selector<G, C>>,

	//#[debug(skip)]
	#[serde[skip_deserializing]]
	pub init_genome: G,

	pub elitism_rate: Option<f64>,
	pub crossover_rate: Option<f64>,
	pub enable_speciation: Option<bool>, // TODO add more?

	pub log_file: Option<String> //we'll see about this
}

/// Input command-line arguments
#[derive(clap::Parser, Debug)]
pub struct GenAlgParamsCLI {
	// TODO -F / --config [filename]  (get rid of this comment and add real description)
	#[arg(short='F',long="config")]
	pub config: Option<String>,
	// TODO -s / --seed [seed] (get rid of this comment and add real description)
	#[arg(short='s',long="seed")]
	pub seed: Option<u64>,
}



impl GenAlgParamsOpts {
	fn from_file(filename: &str) -> Self {
		let contents = fs::read_to_string(filename).unwrap();
		let config = toml::from_str(&contents);
		return config.unwrap();
	}

	pub fn build(self) -> GenAlgParams {
		let seed = self.seed.unwrap_or(thread_rng().gen());
		let log_file = self.log_file.unwrap_or_else(|| format!("trial_{}.log", 0)); // TODO actual timestamp
		GenAlgParams {
			seed,
			pop_size: self.pop_size.unwrap_or(100),
			num_generations: self.num_generations.unwrap_or(20),
			mutators: self.mutators.or(Vec::new()), //FIXME?
			mutation_rate: self.mutation_rate.unwrap_or(1.0),
			selector: self.selector, //FIXME?
			init_genome: self.init_genome, //FIXME?
			elitism_rate: self.elitism_rate.unwrap_or(0.05),
			crossover_rate: self.crossover_rate.unwrap_or(0.95),
			enable_speciation: self.enable_speciation.unwrap_or(false),

			log_file: self.log_file.unwrap(),
		}
	}
}

impl From<GenAlgParamsOpts> for GenAlgParams {
	fn from(value: GenAlgParamsOpts) -> Self {
		value.build()
	}
}
