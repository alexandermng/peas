use bon::{builder, Builder};
use derive_more::derive::{From, Into};
use rand::Rng;
use serde::{Deserialize, Serialize, Serializer};
use std::fmt::Debug;
use std::path::Path;
use std::str::FromStr;
use std::{fs, marker::PhantomData};
use toml;

use crate::genetic::{GenAlg, Results};
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
	M: Serialize + for<'m> Deserialize<'m>,
	S: Serialize + for<'s> Deserialize<'s>,
")]
#[rustfmt::skip]
pub struct GenAlgConfig<G, C, M, S>
where
	G: Genome<C>,
	C: AsContext,
	M: Mutator<G, C>,
	S: Selector<G, C>,
{
	pub problem: ProblemSet, // TODO: Fix
	pub params: GenAlgParams<G, C, M, S>,

	/// Results to be output.
	#[serde(skip)]
	pub results: ResultsParams,

	/// Enclosing output directory, containing this config (serialized as `config.toml`) and
	/// other logs.
	#[serde(skip)]
	pub output_dir: String,

	// /// Name of csv file storing genome records. Defaults to `data.csv`. Will be found inside
	// /// the output directory.
	// #[serde(skip_serializing, default = "default_datafile")]
	// #[builder(default = default_datafile())]
	// pub datafile: String,
	// TODO put in ResultsParams

	// /// Name of json file storing results. Defaults to `results.json`. Will be found inside the
	// /// output directory.
	// #[serde(skip_serializing, default = "default_resultsfile")]
	// #[builder(default = default_resultsfile())]
	// pub resultsfile: String,
	// TODO put in ResultsParams
}

pub(crate) fn default_datafile() -> String {
	"data.csv".into()
}

pub(crate) fn default_resultsfile() -> String {
	"results.json".into()
}

pub trait Configurator: Serialize + for<'c> Deserialize<'c> {
	type Genome: Genome<Self::Ctx> + Default;
	type Ctx: AsContext;
	type ProblemSet: Serialize + for<'p> Deserialize<'p>; // Problem Set
	type MutationSet: Mutator<Self::Genome, Self::Ctx> + Serialize + for<'m> Deserialize<'m>;
	type SelectorSet: Selector<Self::Genome, Self::Ctx> + Serialize + for<'s> Deserialize<'s>;
	type Results: Results + Serialize;
	type Output: GenAlg<G = Self::Genome, C = Self::Ctx>;

	fn gen_params(
		&self,
	) -> GenAlgParams<Self::Genome, Self::Ctx, Self::MutationSet, Self::SelectorSet>;
	fn gen_results(&self) -> Self::Results;
	fn build() -> Self::Output;

	// TODO: ... merge? idk
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
	#[builder(default, into)]
	#[serde(default)]
	pub seed: SeedString, // set seed for the run, can convert into u64
	pub pop_size: usize,
	pub num_generations: usize,
	pub max_fitness: Option<f64>,

	pub mutators: Vec<M>, // enabled mutations. to be selected from
	pub mutation_rate: f64,

	pub selector: S,

	#[builder(default)]
	#[serde(default)]
	pub speciation: SpeciesParams,

	#[serde(skip)]
	pub init_genome: G,

	pub elitism_rate: f64,
	pub crossover_rate: f64,

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
			seed: self.seed,
			pop_size: self.pop_size,
			num_generations: self.num_generations,
			max_fitness: self.max_fitness,
			mutators: self.mutators.clone(),
			mutation_rate: self.mutation_rate,
			selector: self.selector.clone(),
			speciation: self.speciation.clone(),
			init_genome: self.init_genome.clone(),
			elitism_rate: self.elitism_rate,
			crossover_rate: self.crossover_rate,
			_ctx: self._ctx,
		}
	}
}

/// Parameters for results output.
#[derive(Serialize, Deserialize)]
pub struct ResultsParams {
	pub use_default: bool,

	#[serde(skip)]
	pub custom_results: Vec<crate::wasm::WasmGenAlgResults>,
	// TODO: fix this shit. below, too.
}

impl ResultsParams {
	pub fn from_single(
		custom: impl Results<Genome = WasmGenome, Ctx = Context> + Send + Sync + 'static,
	) -> Self {
		let c: Vec<crate::wasm::WasmGenAlgResults> = vec![Box::new(custom)];
		Self::from_custom(c)
	}

	pub fn from_custom(custom_results: Vec<crate::wasm::WasmGenAlgResults>) -> Self {
		Self {
			use_default: false,
			custom_results,
		}
	}
}

impl Default for ResultsParams {
	fn default() -> Self {
		Self {
			use_default: true,
			custom_results: vec![],
		}
	}
}

impl Debug for ResultsParams {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ResultsParams")
			.field("use_default", &self.use_default)
			.field("custom_results", &self.custom_results.len())
			.finish()
	}
}

impl Clone for ResultsParams {
	fn clone(&self) -> Self {
		Self {
			use_default: self.use_default.clone(),
			custom_results: vec![],
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

impl Default for SeedString {
	fn default() -> Self {
		Self(rand::rng().random())
	}
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SpeciesParams {
	pub enabled: bool,         // whether to use species
	pub threshold: f64,        // compatibility threshold
	pub fitness_sharing: bool, // whether to use explicit fitness sharing
}
