use bon::{builder, Builder};
use derive_more::derive::{From, Into};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize, Serializer};
use std::path::Path;
use std::str::FromStr;
use std::{fs, marker::PhantomData};
use toml;

use crate::genetic::Results;
use crate::problems::{Problem, ProblemSet};
use crate::{
	genetic::{AsContext, Genome, Mutator, OnceMutator, Predicate, Selector},
	selection::TournamentSelection,
	wasm::{mutations::WasmMutationSet, Context, WasmGenome},
};

/// A full config for a genetic algorithm. Meant to be saved to a file.
#[derive(Serialize, Deserialize, Builder, Debug, Clone)]
#[serde(bound = "
	G: Default,
	R: Default,
	P: Serialize + for<'p> Deserialize<'p>,
	M: Serialize + for<'m> Deserialize<'m>,
	S: Serialize + for<'s> Deserialize<'s>,
")]
pub struct GenAlgConfig<G, C, P, R, M, S>
where
	G: Genome<C>,
	C: AsContext,
	R: Results,
	M: Mutator<G, C>,
	S: Selector<G, C>,
{
	pub problem: P,
	pub params: GenAlgParams<G, C, M, S>,

	/// Skipped when serializing, so you must build yourself
	#[serde(skip)]
	pub results: R,

	/// Enclosing output directory, containing this config (serialized as `config.toml`) and
	/// other logs.
	#[serde(skip)]
	pub output_dir: String,

	/// Name of csv file storing genome records. Defaults to `data.csv`. Will be found inside
	/// the output directory.
	#[serde(skip_serializing, default = "default_datafile")]
	#[builder(default = default_datafile())]
	pub datafile: String,

	/// Name of json file storing results. Defaults to `results.json`. Will be found inside the
	/// output directory.
	#[serde(skip_serializing, default = "default_resultsfile")]
	#[builder(default = default_resultsfile())]
	pub resultsfile: String,
}

pub(crate) fn default_datafile() -> String {
	"data.csv".into()
}

pub(crate) fn default_resultsfile() -> String {
	"results.json".into()
}

/// Actual parameters for the genetic algorithm. Can be saved as an output to a file and subsequently loaded in to replicate runs.
#[derive(Serialize, Deserialize, Builder, Debug)]
#[serde(bound = "
	G: Default,
	for<'m> M: Deserialize<'m> + Serialize,
	for<'s> S: Deserialize<'s> + Serialize
")]
pub struct GenAlgParams<G, C, M, S>
where
	G: Genome<C>,
	C: AsContext,
	M: Mutator<G, C>,
	S: Selector<G, C>,
{
	#[builder(into)]
	pub seed: SeedString, // set seed for the run, can convert into u64

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

	#[doc(hidden)]
	#[serde(skip)]
	#[builder(skip)]
	_ctx: PhantomData<C>,
}

impl<G, C, M, S> Clone for GenAlgParams<G, C, M, S>
where
	G: Genome<C> + Clone,
	C: AsContext,
	M: Mutator<G, C> + Clone,
	S: Selector<G, C> + Clone,
{
	fn clone(&self) -> Self {
		Self {
			seed: self.seed.clone(),
			pop_size: self.pop_size.clone(),
			num_generations: self.num_generations.clone(),
			max_fitness: self.max_fitness.clone(),
			mutators: self.mutators.clone(),
			mutation_rate: self.mutation_rate.clone(),
			selector: self.selector.clone(),
			init_genome: self.init_genome.clone(),
			elitism_rate: self.elitism_rate.clone(),
			crossover_rate: self.crossover_rate.clone(),
			enable_speciation: self.enable_speciation.clone(),
			_ctx: self._ctx.clone(),
		}
	}
}

/// Utility for (de)serializing a u64 seed to/from a string.
#[derive(Debug, From, Into, Clone, Copy)]
pub struct SeedString(u64);

// TODO: ^ impl FromStr

impl Serialize for SeedString {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: Serializer,
	{
		let s = self.0.to_string();
		serializer.serialize_str(&s)
	}
}
impl<'de> Deserialize<'de> for SeedString {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let s = String::deserialize(deserializer)?;
		// TODO some way of string to int
		let i: u64 = s.parse().map_err(serde::de::Error::custom)?;
		Ok(Self(i))
	}
}
