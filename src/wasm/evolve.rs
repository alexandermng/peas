use std::{
	any::Any,
	borrow::{Borrow, BorrowMut},
	cell::{RefCell, RefMut},
	collections::{HashMap, HashSet},
	fs,
	hash::{DefaultHasher, Hash, Hasher},
	marker::PhantomData,
	mem,
	ops::{Index, IndexMut, Range},
	path::{Path, PathBuf},
	time::Instant,
};

use bon::{builder, Builder};
use csv::Writer;
use eyre::{eyre, DefaultHandler, Error, OptionExt, Result, WrapErr};
use rand::{
	distr::{Distribution, Uniform},
	seq::{IndexedRandom, SliceRandom},
	Rng, SeedableRng,
};
use rand_pcg::Pcg64Mcg;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use slab::Slab;
use wasm_encoder::{Instruction, ValType};
use wasmtime::{Engine, Instance, Linker, Module, Store, WasmParams, WasmResults, WasmTy};

use crate::{
	genetic::{
		self, Configurable, ConfiguredGenAlg, GenAlg, Genome, Mutator, OnceMutator, Predicate,
		Selector, Species,
	},
	params::{self, ResultsParams},
	problems::{Problem, ProblemSet, Solution, Sum3},
	wasm::GeneDiff,
	Id,
};
use crate::{
	genetic::{AsContext, Results},
	params::{GenAlgConfig, GenAlgParams},
	selection::TournamentSelection,
	wasm::{
		genome::{InnovNum, StackValType, WasmGene, WasmGenome},
		mutations::AddOperation,
	},
};

use super::{
	mutations::{MutationLog, WasmMutationSet},
	species::{WasmSpecies, WasmSpeciesId},
	WasmGenomeId, WasmGenomeRecord,
};

/// Assembled phenotype of an individual in a genetic algorithm. Used as a solution to a problem.
#[derive(Clone)]
pub struct Agent {
	pub engine: Engine,
	pub module: Module,
	// ^ todo consider access. also starting state to seed Store<_>
	// pub fitness: f64,
	is_valid: bool,
}

impl Agent {
	fn new(engine: Engine, binary: &[u8]) -> Self {
		let mut is_valid = true;
		let module = match Module::from_binary(&engine, binary) {
			Ok(o) => o,
			Err(err) => {
				log::error!("Invalid Agent({binary:?}), {err}");
				// panic!("{err:?}")
				is_valid = false;
				Module::new(
					// default :/ TODO fixthis
					&engine,
					r#"
				(module
					(type (;0;) (func (param i32 i32 i32) (result i32)))
					(func (;0;) (type 0) (param i32 i32 i32) (result i32)
					i32.const 0
					)
					(export "main" (func 0))
				)
				"#,
				)
				.unwrap()
			}
		};
		Agent {
			engine,
			module,
			is_valid,
		}
	}
}

impl<P> Solution<P> for Agent
where
	P: Problem,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	fn try_exec(&self, args: P::In) -> eyre::Result<P::Out> {
		if !self.is_valid {
			return Err(eyre!("invalid module"));
		}
		let linker = Linker::new(&self.engine);
		let mut store = Store::new(&self.engine, ());
		let instance = linker.instantiate(&mut store, &self.module).unwrap();
		let main = instance
			.get_typed_func::<<P as Problem>::In, <P as Problem>::Out>(&mut store, "main")
			.unwrap();

		main.call(&mut store, args)
			.map_err(|e| eyre!("solution failed: {e:?}"))
	}
}

type InnovKey = MutationLog;

#[derive(Debug)]
pub struct Context {
	pub generation: usize, // current generation (0-indexed)
	pub max_fitness: f64,  // current top fitness
	pub avg_fitness: f64,  // current mean fitness

	pub(crate) rng: Pcg64Mcg,                // reproducible rng
	pub(crate) genomes: Slab<WasmGenome>,    // full genome backing store
	pub(crate) species: Slab<WasmSpecies>,   // species backing store
	innov_cnt: InnovNum,                     // innovation number count
	cur_innovs: HashMap<InnovKey, InnovNum>, // running log of current unique innovations, cleared per-generation.

	pub(crate) start_time: Instant,  // start time of run
	pub(crate) latest_time: Instant, // time since last generation
}

impl Context {
	pub fn new(seed: u64) -> Self {
		Self {
			generation: 0,
			max_fitness: 0.0,
			avg_fitness: 0.0,
			rng: Pcg64Mcg::seed_from_u64(seed),
			genomes: Slab::new(),
			species: Slab::new(),
			innov_cnt: InnovNum(0),
			cur_innovs: HashMap::new(),
			start_time: Instant::now(),
			latest_time: Instant::now(),
		}
	}

	pub fn new_genome(&mut self, genome: WasmGenome) -> WasmGenomeId {
		Id::from(self.genomes.insert(genome))
	}

	pub fn get_genome(&self, id: WasmGenomeId) -> &WasmGenome {
		self.genomes.get(id.into()).expect("valid genome id")
	}

	pub fn get_genome_mut(&mut self, id: WasmGenomeId) -> &mut WasmGenome {
		self.genomes.get_mut(id.into()).expect("valid genome id")
	}

	pub fn iter_genomes<I>(&self, ids: I) -> impl Iterator<Item = &WasmGenome>
	where
		I: IntoIterator,
		I::Item: Borrow<WasmGenomeId>,
	{
		ids.into_iter().map(move |id| self.get_genome(*id.borrow()))
	}

	// pub fn iter_genomes_mut<'a, I>(
	// 	&'a mut self,
	// 	ids: I,
	// ) -> impl Iterator<Item = &'a mut WasmGenome> + 'a
	// where
	// 	I: IntoIterator<Item = WasmGenomeId> + 'a,
	// {
	// 	let genomes = &mut self.genomes;
	// 	ids.into_iter()
	// 		.map(move |id| genomes.get_mut(id.into()).expect("valid id"))
	// }

	/// Creates a new species from a founding genome
	pub fn new_species(&mut self, founder: WasmGenomeId) -> WasmSpeciesId {
		let spec = WasmSpecies {
			representative: founder,
			members: vec![founder],
			fitness: self[founder].fitness,
			capacity: 1, // set later
			starting_generation: self.generation,
			archive: vec![founder],
		};
		Id::from(self.species.insert(spec))
	}

	pub fn get_species(&self, id: WasmSpeciesId) -> &WasmSpecies {
		self.species.get(id.into()).expect("valid species id")
	}

	pub fn get_species_mut(&mut self, id: WasmSpeciesId) -> &mut WasmSpecies {
		self.species.get_mut(id.into()).expect("valid species id")
	}

	pub fn iter_species<I>(&self, ids: I) -> impl Iterator<Item = &WasmSpecies>
	where
		I: IntoIterator,
		I::Item: Borrow<WasmSpeciesId>,
	{
		ids.into_iter()
			.map(move |id| self.get_species(*id.borrow()))
	}

	// /// Get or assign an innovation number to an innovation keyed by location of mutation and instruction added
	pub fn innov(&mut self, key: InnovKey) -> InnovNum {
		*self.cur_innovs.entry(key).or_insert_with(|| {
			let out = self.innov_cnt;
			*self.innov_cnt += 1;
			out
		})
	}

	pub fn num_innovs(&self) -> InnovNum {
		self.innov_cnt
	}
}

impl AsContext for Context {
	#[inline]
	fn rng(&mut self) -> &mut impl Rng {
		&mut self.rng
	}

	#[inline]
	fn generation(&self) -> usize {
		self.generation
	}
}

impl Index<WasmGenomeId> for Context {
	type Output = WasmGenome;

	fn index(&self, id: WasmGenomeId) -> &Self::Output {
		self.get_genome(id)
	}
}

impl IndexMut<WasmGenomeId> for Context {
	fn index_mut(&mut self, id: WasmGenomeId) -> &mut Self::Output {
		self.get_genome_mut(id)
	}
}

impl Index<&WasmGenomeId> for Context {
	type Output = WasmGenome;

	fn index(&self, id: &WasmGenomeId) -> &Self::Output {
		self.get_genome(*id)
	}
}

impl IndexMut<&WasmGenomeId> for Context {
	fn index_mut(&mut self, id: &WasmGenomeId) -> &mut Self::Output {
		self.get_genome_mut(*id)
	}
}

impl Index<WasmSpeciesId> for Context {
	type Output = WasmSpecies;

	fn index(&self, id: WasmSpeciesId) -> &Self::Output {
		self.get_species(id)
	}
}

impl IndexMut<WasmSpeciesId> for Context {
	fn index_mut(&mut self, id: WasmSpeciesId) -> &mut Self::Output {
		self.get_species_mut(id)
	}
}

impl Index<&WasmSpeciesId> for Context {
	type Output = WasmSpecies;

	fn index(&self, id: &WasmSpeciesId) -> &Self::Output {
		self.get_species(*id)
	}
}

impl IndexMut<&WasmSpeciesId> for Context {
	fn index_mut(&mut self, id: &WasmSpeciesId) -> &mut Self::Output {
		self.get_species_mut(*id)
	}
}

#[derive(Default, Debug, Serialize, Clone)]
pub struct DefaultWasmGenAlgResults {
	// pub configfile: Option<String>, // the config which this ran from
	#[serde(skip)]
	pub resultsfile: Option<String>, // this results.json file
	pub datafile: Option<String>, // collated gene data csv
	pub success: bool,            // whether it found a solution
	pub num_generations: usize,   // how many generations it ran for
	pub total_time: f64,          // total time of run, in seconds
	pub max_fitnesses: Vec<f64>,  // top fitness for each generation
	pub avg_fitnesses: Vec<f64>,  // mean fitness for each generation
	pub times: Vec<f64>,          // evaluation times for each generation, in seconds

	#[serde(skip)]
	pub hall_of_fame: Vec<WasmGenomeId>, // best genomes per generation

	#[serde(skip)]
	records: Vec<WasmGenomeRecord>, // genome records, saved to datafile
}

impl Results for DefaultWasmGenAlgResults {
	type Ctx = Context;
	type Genome = WasmGenome;

	fn initialize(&mut self, ctx: &mut Self::Ctx) {
		ctx.start_time = Instant::now();
		ctx.latest_time = Instant::now();
	}

	fn record_generation(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>]) {
		// Record timing
		self.times.push(ctx.latest_time.elapsed().as_secs_f64());

		// Update context stats
		self.max_fitnesses.push(ctx.max_fitness);
		self.avg_fitnesses.push(ctx.avg_fitness);

		// Records
		for (i, &p) in pop.iter().enumerate() {
			let id = ctx.generation * pop.len() + i; // TODO move id into genome itself; assign at reproduction-time
			self.records.push(WasmGenomeRecord::new(id, &ctx[p]));
		}

		// Log
		log::info!("Average Fitness: {:.5}", ctx.avg_fitness);
		log::info!("Top 10:");
		for &p in pop.iter().take(10) {
			log::info!(
				"\t{:.3} <-- {} (gen {}, len {}, species #{})",
				ctx[p].fitness,
				p,
				ctx[p].generation,
				ctx[p].len(),
				ctx[p].species.expect("has species")
			);
		}
		self.hall_of_fame.push(pop[0]);
	}

	fn record_success(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>]) {
		self.success = true;
	}

	fn finalize(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>], outdir: &Path) {
		self.total_time = ctx.start_time.elapsed().as_secs_f64();
		self.num_generations = ctx.generation + 1; // Because generations are 0-indexed.

		// data.csv
		if let Some(datafile) = &self.datafile {
			let csvfile = outdir.join(datafile);
			let mut wtr = Writer::from_path(csvfile).unwrap();
			for rec in &self.records {
				wtr.serialize(rec);
			}
			wtr.flush().unwrap();
		}

		// hall of fame
		for (eration, &id) in self.hall_of_fame.iter().enumerate() {
			let filename = outdir.join(format!("hof_gen{eration}_{id}.wasm"));
			fs::write(filename, ctx[id].emit());
		}

		// results.json
		if let Some(resultsfile) = &self.resultsfile {
			let resultsfile = outdir.join(resultsfile);
			let results = serde_json::to_string_pretty(&self).expect("results should serialize");
			fs::write(resultsfile, results);
		}
	}
}

pub type WasmGenAlgConfig<M, S> = GenAlgConfig<WasmGenome, Context, M, S>;
pub type WasmGenAlgParams<M, S> = GenAlgParams<WasmGenome, Context, M, S>;
pub type WasmGenAlgResults = Box<dyn Results<Genome = WasmGenome, Ctx = Context> + Send + Sync>;

/// A genetic algorithm for synthesizing WebAssembly modules to solve a problem. The problem must specify
/// a fitness function. Parametrized across the problem, the mutation type, and the selection type.
pub struct WasmGenAlg<P, M = WasmMutationSet, S = TournamentSelection>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	// Parameters
	pub problem: P,
	pub params: WasmGenAlgParams<M, S>,
	pub results: Vec<WasmGenAlgResults>,

	mutators: Vec<M>,        // copied from params
	selector: S,             // copied from params
	init_genome: WasmGenome, // copied from params

	seed: u64,       // copied from params
	outdir: PathBuf, // copied from config, or default
	// resultsfile: String, // copied from config, or default

	// Runtime use
	engine: Engine,
	ctx: RefCell<Context>,       // runtime context
	pop: Vec<WasmGenomeId>,      // current population of genomes
	species: Vec<WasmSpeciesId>, // current active species
}

// TODO refactor this mess entirely. it's horrendous.
impl<P, M, S> Configurable<WasmGenAlgConfig<M, S>> for WasmGenAlg<P, M, S>
where
	P: Problem + Into<ProblemSet> + Clone + Sync,
	M: Mutator<WasmGenome, Context> + Clone + for<'m> Deserialize<'m> + Serialize + 'static + Send,
	S: Selector<WasmGenome, Context> + Clone + for<'s> Deserialize<'s> + Serialize + 'static + Send,
{
	type Output = dyn GenAlg<G = WasmGenome, C = Context> + Send;

	// TODO: maybe move this? doesn't depend on P.
	fn from_config(config: WasmGenAlgConfig<M, S>) -> Box<Self::Output> {
		let WasmGenAlgConfig {
			problem,
			params,
			results: result_params,
			output_dir,
			// datafile,
			// resultsfile,
		} = config;
		let mut results: Vec<WasmGenAlgResults> = Vec::new();
		if result_params.use_default {
			results.push(Box::new({
				let mut r = DefaultWasmGenAlgResults::default();
				r.datafile = Some(params::default_datafile());
				r.resultsfile = Some(params::default_resultsfile());
				// TODO: ^ figure out why this is needed and replace. perhaps modify in default().
				r
			}));
		}
		results.extend(result_params.custom_results);

		match problem {
			ProblemSet::Sum3(p) => Box::new(WasmGenAlg::with_results(p, params, results)),
			ProblemSet::Sum4(p) => Box::new(WasmGenAlg::with_results(p, params, results)),
			ProblemSet::Polynom2(p) => Box::new(WasmGenAlg::with_results(p, params, results)),
		}
	}

	fn gen_config(&self) -> WasmGenAlgConfig<M, S> {
		let problem = self.problem.clone().into();
		WasmGenAlgConfig {
			problem,
			params: self.params.clone(),
			results: ResultsParams::default(), // TODO fix
			output_dir: self.outdir.to_string_lossy().to_string(),
			// datafile: self.results.datafile.clone().unwrap_or_default(),
			// resultsfile: self.results.resultsfile.clone().unwrap_or_default(),
		}
		// clone for simplicity... can fix later
	}

	fn log_config(&mut self) {
		let configfile = self.outdir.join("config.toml");
		let config = self.gen_config();
		let config = toml::to_string(&config).expect("config should serialize");
		fs::write(configfile, config);
	}
}

impl<P, M, S> WasmGenAlg<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context> + Clone,
	S: Selector<WasmGenome, Context> + Clone,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	/// Create a new WasmGenAlg based on the given problem, parameters, and default results.
	pub fn new(problem: P, params: WasmGenAlgParams<M, S>) -> Self {
		Self::with_results(
			problem,
			params,
			vec![Box::new(DefaultWasmGenAlgResults::default())],
		)
	}

	/// Create a new WasmGenAlg based on the given problem, parameters, and any desired results.
	pub fn with_results(
		problem: P,
		params: WasmGenAlgParams<M, S>,
		results: Vec<WasmGenAlgResults>,
	) -> Self {
		let size = params.pop_size;
		let seed: u64 = params.seed.into();
		let mutators = params.mutators.clone();
		let selector = params.selector.clone();
		let init_genome = params.init_genome.clone();
		let stop_cond: Box<dyn Predicate<WasmGenome, Context>> = match params.max_fitness {
			Some(x) => Box::new(move |ctx: &mut Context, _: &[WasmGenomeId]| -> bool {
				ctx.max_fitness >= x
			}),
			None => Box::new(|_: &mut Context, _: &[WasmGenomeId]| false),
		};
		let outdir = {
			let mut p = PathBuf::from("data");
			p.push(problem.name());
			p.push(format!("trial_{}", seed));
			p
		};

		WasmGenAlg {
			problem,
			params,
			results,

			mutators,
			selector,
			init_genome,
			seed,
			outdir,
			// resultsfile: params::default_resultsfile(),

			// Runtime use
			engine: Engine::default(),
			ctx: RefCell::new(Context::new(seed)),
			pop: Vec::with_capacity(size),
			species: Vec::new(),
		}
	}
}

impl<P, M, S> WasmGenAlg<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	/// Add another results object before running.
	pub fn register_results(&mut self, added_results: WasmGenAlgResults) {
		self.results.push(added_results);
	}

	/// Get the runtime context
	pub fn context(&self) -> std::cell::Ref<Context> {
		self.ctx.borrow()
	}

	/// Get the runtime context mutably
	pub fn context_mut(&self) -> std::cell::RefMut<Context> {
		self.ctx.borrow_mut()
	}

	/// Distribute new population into species, after they've been evaluated, at the end of every generation.
	/// This should happen BEFORE selection/reproduction in the next generation, because it may affect the
	/// fitness of its members.
	pub fn speciate(&mut self) {
		// TODO move to genalg?

		// 1. Empty each species, reducing to a representative
		let mut map = HashMap::<WasmGenomeId, WasmSpeciesId>::new();
		for &specid in &self.species {
			let membs = {
				let mut ctx = self.context_mut();
				mem::take(&mut ctx[specid].members) // clears members from species
			}; // TODO refactor into species method
			let rep = *membs
				.choose(self.ctx.get_mut().rng())
				.expect("non-empty species");
			{
				// update representative
				let mut ctx = self.ctx.borrow_mut();
				let mut spec = &mut ctx[specid];
				spec.representative = rep;
			}
			map.insert(rep, specid);
		}

		// 2. For each individual of new pop, assign to a species (or create new)
		let mut species_added = Vec::new();
		for &indiv in &self.pop {
			let mut ctx = self.ctx.borrow_mut();
			let genom = &ctx[indiv];
			let mut found = false;
			for (&rep, &specid) in &map {
				let rep = &ctx[rep];
				let dist = genom.dist(rep);
				if dist < self.params.speciation.threshold {
					ctx[specid].add_genome(indiv);
					ctx[indiv].species = Some(specid);
					found = true;
					break;
				}
			}
			if !found {
				let gen = ctx.generation;
				let specid = ctx.new_species(indiv);
				ctx[indiv].species = Some(specid);
				map.insert(indiv, specid);
				species_added.push(specid);
			}
		}
		let num_added = species_added.len();
		self.species.extend(species_added);

		// 3. Adjust fitnesses of each individual based on species size
		if self.params.speciation.fitness_sharing {
			for &indiv in &self.pop {
				let mut ctx = self.ctx.borrow_mut();
				let genom = &ctx[indiv];
				let spec = &ctx[genom.species.expect("species assigned")];
				let adj = spec.adjusted_fitness(genom.fitness);
				ctx[indiv].adjusted_fitness = Some(adj);
			}
		}

		// 4. Prune extinct species.
		let mut num_extinct = 0;
		self.species.retain(|&specid| {
			let ctx = self.ctx.borrow();
			let size = ctx[specid].size();
			if size == 0 {
				// log::debug!("Species #{} died out.", specid);
				num_extinct += 1;
				false
			} else {
				true // keep
			}
		});

		// 5. For each species, set capacities for next generation based on proportion of adjusted fitness. Ensure population count.
		let total_fitness: f64 = self
			.pop
			.iter()
			.map(|&g| self.ctx.borrow()[g].fitness())
			.sum();
		let mut count = 0;
		for &specid in &self.species {
			let f: f64 = {
				let ctx = self.ctx.borrow();
				ctx[specid].members.iter().map(|&g| ctx[g].fitness()).sum()
			};
			let cap = if total_fitness == 0.0 {
				(self.params.pop_size as f64 / self.species.len() as f64).ceil() as usize // special case: if no total fitness, everyone is equally bad, so distribute equally among all species
			} else if f == 0.0 {
				0
			} else {
				(f / total_fitness * self.params.pop_size as f64).ceil() as usize
				// Always overproduce, allow for small fitness species to survive. Will prune later down.
			};
			let mut ctx = self.ctx.borrow_mut();
			ctx[specid].fitness = f;
			ctx[specid].capacity = cap;
			log::debug!(
				"Species #{specid} (size {}) set to {} capacity ({:.3} / {:.3} => {:.3} fitness)",
				ctx[specid].size(),
				cap,
				f,
				total_fitness,
				f / total_fitness,
			);
			count += cap;
		}
		if count > self.params.pop_size {
			// Overcounted with ceil earlier, so pruning down randomly, weighted by capacity
			let num = count - self.params.pop_size;
			let capmap: Vec<_> = {
				let mut ctx = self.ctx.borrow_mut();
				self.species
					.iter()
					.map(|&id| (id, ctx[id].capacity as f64))
					.collect()
			};
			let ids: Vec<_> = capmap
				.choose_multiple_weighted(self.ctx.get_mut().rng(), num, |(_, cap)| *cap)
				.expect("we have some species")
				.map(|(id, _)| *id)
				.collect();
			let mut ctx = self.ctx.borrow_mut();
			for id in ids {
				ctx[id].capacity -= 1;
			}
			count -= num;
		}
		debug_assert_eq!(
			count, self.params.pop_size,
			"total intended capacity should match population size"
		);

		// 5. Prune species with no future capacity due to low fitness.
		let mut num_pruned = 0;
		self.species.retain(|&specid| {
			let ctx = self.ctx.borrow();
			let cap = ctx[specid].capacity;
			if cap == 0 {
				// log::debug!("Species #{} pruned for 0 capacity.", specid);
				num_pruned += 1;
				false
			} else {
				true // keep
			}
		});

		log::info!(
			"Speciated {} individuals into {} species (+{} added, -{} extinct, -{} pruned).",
			self.pop.len(),
			self.species.len(),
			num_added,
			num_extinct,
			num_pruned,
		);
	}
}

impl<P, M, S> GenAlg for WasmGenAlg<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
	P::In: WasmParams,
	P::Out: WasmResults,
	Self: Configurable<WasmGenAlgConfig<M, S>>, /* TODO relax / reimplement */
{
	type G = WasmGenome;
	type C = Context;

	fn run(&mut self) {
		log::info!("Beginning GA trial with seed {}", self.seed);

		fs::create_dir_all(&self.outdir).unwrap();
		// config.toml
		self.log_config();

		let (genome0, species0) = {
			// evaluate base fitness before use
			let mut ctx = self.ctx.borrow_mut();
			let mut g = ctx.new_genome(self.init_genome.clone());
			let agent = Agent::new(self.engine.clone(), &ctx[g].emit());
			ctx[g].fitness = self.problem.fitness(agent);
			// set species
			let specid = ctx.new_species(g);
			ctx[g].species = Some(specid);
			(g, specid)
		};
		self.pop.extend((0..self.params.pop_size).map(|_| genome0));
		if self.params.speciation.enabled {
			self.species.push(species0);
			let mut ctx = self.ctx.borrow_mut();
			ctx[species0].members = self.pop.clone(); // all clones are in this species
			ctx[species0].capacity = self.params.pop_size;
			log::info!(
				"Created Species #{} with {} members, {} capacity.",
				species0,
				ctx[species0].size(),
				ctx[species0].capacity
			);
		}

		for r in &mut self.results {
			r.initialize(self.ctx.get_mut());
		}

		while self.epoch() {}

		for r in &mut self.results {
			r.finalize(self.ctx.get_mut(), &self.pop, &self.outdir);
		}
	}

	fn epoch(&mut self) -> bool {
		let generation = self.ctx.borrow().generation; // current generation
		log::info!("Creating generation {}.", generation);

		let mut elites: Vec<WasmGenomeId> = Vec::new(); // all elites

		if self.params.speciation.enabled {
			self.pop.clear(); // clear, population should consist of species members after last-gen's speciation

			// Elitism, Speciation and Crossover independently on each species
			for &specid in &self.species {
				let (mut membs, mut cap_remaining) = {
					let mut ctx = self.ctx.borrow_mut();
					(mem::take(&mut ctx[specid].members), ctx[specid].capacity)
				};
				let origlen = membs.len();
				log::debug!(
					"Reproducing Species #{specid} from {origlen} members to {cap_remaining} capacity."
				);

				/* Elitism */
				let elitism_cnt =
					((self.params.elitism_rate * (origlen as f64)) as usize).min(cap_remaining); // count based on current size, but capped at cap
				let special_elites = if elitism_cnt > 0 {
					let mut ctx = self.ctx.borrow_mut();
					membs.sort_unstable_by(|a, b| {
						f64::partial_cmp(&ctx[b].fitness, &ctx[a].fitness).unwrap()
					});
					log::debug!("Retaining {elitism_cnt} elites from Species #{specid}.");
					membs[0..elitism_cnt].to_vec()
				} else {
					Vec::new()
				}; // elites for this species
				elites.extend(&special_elites);
				cap_remaining -= elitism_cnt;

				/* Selection */
				membs = if membs.len() <= 2 {
					membs
				} else {
					let mut ctx = self.ctx.borrow_mut();
					self.selector.select(&mut ctx, membs)
				};
				let parent_cnt = membs.len();
				log::debug!("Selected {parent_cnt} survivors from Species #{specid}.");

				/* Crossover */
				let mut children = Vec::new();
				let crossover_cnt = (self.params.crossover_rate * (cap_remaining as f64)) as usize;
				if crossover_cnt > 0 && membs.len() >= 2 {
					let indices: Vec<_> = {
						let mut ctx = self.ctx.borrow_mut();
						let dist = Uniform::new(0, parent_cnt).unwrap();
						(0..crossover_cnt)
							.map(|_| {
								let a = dist.sample(&mut ctx.rng);
								let b = dist.sample(&mut ctx.rng);
								(a, b)
							})
							.collect()
					}; // crossover_cnt pairs of indices into parents

					for (a, b) in indices {
						let child = self.crossover(membs[a], membs[b]);
						let child = self.ctx.get_mut().new_genome(child);
						children.push(child);
					}
					debug_assert_eq!(children.len(), crossover_cnt);

					log::debug!("Reproduced {crossover_cnt} children from {parent_cnt} parents in Species #{specid}.");
					cap_remaining -= crossover_cnt;
				}

				/* Mutation */
				children.extend((0..cap_remaining).map(|_| {
					let mut ctx = self.ctx.borrow_mut();
					membs // clone a random parent
						.choose(&mut ctx.rng)
						.expect("non-empty species")
				})); // fill with clones to max capacity
				let mut children = children
					.into_iter()
					.map(|g| self.mutate(g))
					.collect::<Vec<_>>();

				/* Re-add to pop */
				let mut ctx = self.ctx.borrow_mut();
				ctx[specid].members = membs; // save old members for speciation
				self.pop.extend(children);
			}
			debug_assert!(
				self.pop.len() <= self.params.pop_size,
				"should be under capacity ({} > {})",
				self.pop.len(),
				self.params.pop_size,
			);
			self.pop.extend(elites);
			debug_assert!(
				self.pop.len() == self.params.pop_size,
				"should be fully populated ({} != {})",
				self.pop.len(),
				self.params.pop_size,
			);
		} else {
			// Regular selection + crossover
			let mut nextgen: Vec<Id<Self::G>> = Vec::with_capacity(self.params.pop_size);

			let elitism_cnt = (self.params.elitism_rate * (self.params.pop_size as f64)) as usize;
			if elitism_cnt > 0 {
				log::info!("Passing {elitism_cnt} elites to next generation.");
				elites.extend(&self.pop[0..elitism_cnt])
			};

			// TODO extract selection
			let mut selected = {
				let mut ctx = self.ctx.borrow_mut();
				self.selector.select(&mut ctx, mem::take(&mut self.pop)) // pop empty after selection
			};
			log::info!(
				"Selected {} individuals from current population.",
				selected.len()
			);

			let mut crossover_cnt =
				(self.params.crossover_rate * (self.params.pop_size as f64)) as usize;
			if crossover_cnt > 0 {
				log::info!("Crossing over {crossover_cnt} individuals.");

				let indices: Vec<_> = {
					let mut ctx = self.ctx.borrow_mut();
					let dist = Uniform::new(0, selected.len()).unwrap();
					(0..crossover_cnt)
						.map(|_| {
							let a = dist.sample(&mut ctx.rng);
							let b = dist.sample(&mut ctx.rng);
							(a, b)
						})
						.collect()
				};

				for (a, b) in indices {
					let child = self.crossover(selected[a], selected[b]);
					let child = self.ctx.get_mut().new_genome(child);
					nextgen.push(child);
				}
			} else {
				nextgen.append(&mut selected);
			}

			// Mutation
			let needed = self.params.pop_size - elites.len() - nextgen.len(); // amount needed
			log::info!(
				"Mutating {} unique genomes (+{needed} copy-variants).",
				nextgen.len()
			);
			let fill: Vec<_> = nextgen
				.choose_multiple(&mut self.ctx.get_mut().rng, needed)
				.cloned()
				.collect(); // fill to capacity
			nextgen.extend(fill);
			self.pop = nextgen
				// .into_par_iter() // OPT
				.into_iter()
				.map(|g| self.mutate(g))
				.collect();

			// Add to next generation!
			self.pop.extend(elites);
			debug_assert!(
				self.pop.len() == self.params.pop_size,
				"should be fully populated"
			);
		}

		// TODO move to some "record_create_generation" or even just "record_generation" itself
		{
			let mut ctx = self.ctx.borrow_mut();
			log::info!(
				"Created generation {} ({} species). {} new innovations; {} total innovations.",
				ctx.generation,
				self.species.len(),
				ctx.cur_innovs.len(),
				ctx.innov_cnt,
			);
			log::debug!("Innovations were: {:?}", ctx.cur_innovs);
			ctx.cur_innovs.clear();
		}

		/* Evaluation */
		log::info!("Evaluating generation {generation}.");
		self.evaluate();
		log::info!("Done evaluating generation {generation}.");

		/* Speciation */
		if self.params.speciation.enabled {
			log::info!("Speciating generation {generation}.");
			self.speciate();
		}

		/* Logging */
		let mut ctx = self.ctx.borrow_mut();
		self.pop
			.sort_unstable_by(|a, b| f64::partial_cmp(&ctx[b].fitness, &ctx[a].fitness).unwrap());
		let pop = &self.pop[..]; // read-only, guaranteed for hooks to be sorted by raw fitness

		// Update stats for results recording
		ctx.max_fitness = ctx[pop[0]].fitness;
		ctx.avg_fitness =
			ctx.iter_genomes(pop).map(WasmGenome::fitness).sum::<f64>() / (pop.len() as f64);
		ctx.latest_time = Instant::now(); // TODO consider moving, get rid of timing inside ctx

		// Run results recording
		for r in &mut self.results {
			r.record_generation(&mut ctx, pop);
		}
		if self.params.speciation.enabled {
			for r in &mut self.results {
				r.record_speciation(&mut ctx, &self.species);
			}
		}

		// Test stop condition
		if let &Some(mf) = &self.params.max_fitness {
			if ctx.max_fitness >= mf {
				log::info!(
					"Success achieving max fitness ({mf:.3}). Results are in ./{}",
					self.outdir.to_string_lossy(),
				);
				for r in &mut self.results {
					r.record_success(&mut ctx, pop);
				}
				return false; // exit
			}
		}
		if ctx.generation >= self.params.num_generations {
			log::info!(
				"Completed {} generations. Results are in ./{}",
				ctx.generation,
				self.outdir.to_string_lossy()
			);
			return false;
		}

		// Update parameters for next generation
		ctx.generation += 1;
		self.mutators
			.iter_mut()
			.for_each(|m| m.vary_params(&mut ctx, pop));
		self.selector.vary_params(&mut ctx, pop);
		// self.crossover.vary_params(&mut ctx, pop);

		true // continue!
	}

	fn evaluate(&mut self) {
		let mut ctx = self.ctx.borrow_mut();
		let agents: Vec<_> = ctx
			.iter_genomes(&self.pop)
			// OPT can I par_iter here? I think I need WasmGenome: Send / Sync
			.map(WasmGenome::emit)
			.map(|b| Agent::new(self.engine.clone(), &b))
			.collect();
		let fitnai: Vec<_> = agents.par_iter().map(|a| self.problem.fitness(a)).collect();
		for (&g, f) in Iterator::zip(self.pop.iter(), fitnai) {
			ctx[g].fitness = f;
		}
	}

	fn mutate(&self, indiv: Id<Self::G>) -> Id<Self::G> {
		let mut ctx = self.ctx.borrow_mut();
		let indiv = ctx[indiv].clone();
		let indiv = self
			.mutators
			.choose(ctx.rng())
			.expect("non-empty mutators")
			.mutate(&mut *ctx, indiv);
		ctx.new_genome(indiv)
	}

	fn select(&self, pop: &[Id<Self::G>]) -> Vec<Id<Self::G>> {
		let mut ctx = self.ctx.borrow_mut();
		self.selector.select(&mut *ctx, pop.to_vec())
	}

	fn crossover(&self, a: Id<Self::G>, b: Id<Self::G>) -> Self::G {
		let (par_a, par_b) = {
			let ctx = self.ctx.borrow();
			(ctx[a].clone(), ctx[b].clone())
		};
		// TODO OPT don't clone. would have to change reproduce context since it also needs ctx
		let mut ctx = self.ctx.borrow_mut();
		// SAFETY: reproducing does not mutate the parents in ctx's genome backing
		par_a.reproduce_into(&par_b, &mut *ctx)
	}
}
