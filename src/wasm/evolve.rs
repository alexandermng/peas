use std::{
	any::Any,
	borrow::BorrowMut,
	cell::RefCell,
	collections::{HashMap, HashSet},
	fs,
	hash::{DefaultHasher, Hash, Hasher},
	mem,
	ops::Range,
};

use eyre::{eyre, DefaultHandler, OptionExt, Result};
use rand::{
	distributions::Uniform,
	prelude::Distribution,
	seq::{index::sample, IteratorRandom, SliceRandom},
	thread_rng, Rng, SeedableRng,
};
use rand_pcg::Pcg64Mcg;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use wasm_encoder::{Instruction, ValType};
use wasmtime::{Engine, Instance, Linker, Module, Store, WasmParams, WasmResults, WasmTy};

use crate::{
	genetic::AsContext,
	params::GenAlgParams,
	wasm::{
		genome::{InnovNum, StackValType, WasmGene, WasmGenome},
		mutations::AddOperation,
	},
};
use crate::{
	genetic::{self, GenAlg, Genome, Mutator, OnceMutator, Predicate, Problem, Selector, Solution},
	wasm::GeneDiff,
};

use super::mutations::MutationLog;

/// Assembled phenotype of an individual in a genetic algorithm. Used as a solution to a problem.
#[derive(Clone)]
pub struct Agent {
	pub engine: Engine,
	pub module: Module,
	// ^ todo consider access. also starting state to seed Store<_>
	// pub fitness: f64,
}

impl Agent {
	fn new(engine: Engine, binary: &[u8]) -> Self {
		let module = match Module::from_binary(&engine, binary) {
			Ok(o) => o,
			Err(err) => {
				log::error!("Invalid Agent: {binary:?}");
				panic!("{err:?}")
			}
		};
		Agent { engine, module }
	}
}

impl<P> Solution<P> for Agent
where
	P: Problem,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	fn exec(&self, args: P::In) -> P::Out {
		let linker = Linker::new(&self.engine);
		let mut store = Store::new(&self.engine, ());
		let instance = linker.instantiate(&mut store, &self.module).unwrap();
		let main = instance
			.get_typed_func::<<P as Problem>::In, <P as Problem>::Out>(&mut store, "main")
			.unwrap();

		main.call(&mut store, args).unwrap()
	}
}

impl<P, T> Solution<P> for &T
where
	P: Problem,
	T: Solution<P>,
{
	fn exec(&self, args: P::In) -> P::Out {
		(**self).exec(args)
	}
}

type InnovKey = MutationLog;

#[derive(Debug)]
pub struct Context {
	pub generation: usize, // current generation
	pub max_fitness: f64,  // current top fitness
	pub avg_fitness: f64,  // current mean fitness

	pub(crate) rng: Pcg64Mcg,                // reproducible rng
	innov_cnt: InnovNum,                     // innovation number count
	cur_innovs: HashMap<InnovKey, InnovNum>, // running log of current unique innovations, cleared per-generation.
}

impl Context {
	pub fn new(seed: u64) -> Self {
		Self {
			generation: 0,
			max_fitness: 0.0,
			avg_fitness: 0.0,
			rng: Pcg64Mcg::seed_from_u64(seed),
			innov_cnt: InnovNum(0),
			cur_innovs: HashMap::new(),
		}
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

/// A genetic algorithm for synthesizing WebAssembly modules to solve a problem. The problem must specify
/// a fitness function. Parametrized across the problem, the mutation type, and the selection type.
pub struct WasmGenAlg<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	// Parameters
	problem: P,
	mutator: M,
	selector: S,
	init_genome: OnceMutator<WasmGenome, Context>,
	stop_cond: Box<dyn Predicate<WasmGenome, Context>>,
	params: GenAlgParams,
	seed: u64,
	log_file: String,

	// Runtime use
	engine: Engine,
	ctx: RefCell<Context>,
	pop: Vec<WasmGenome>, // population of genomes
}

impl<P, M, S> WasmGenAlg<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	pub fn run(&mut self) {
		self.seed = self.params.seed.unwrap(); // TODO default
		self.log_file = match &self.params.log_file {
			Some(s) => s.clone(),
			None => format!("trial_{}.log", self.seed),
		};

		log::info!("Beginning GA trial with seed {}", self.seed);
		self.init();

		fs::create_dir(&self.log_file).unwrap();

		while self.epoch() {}
	}

	fn init(&mut self) {
		let params = &[StackValType::I32, StackValType::I32, StackValType::I32]; // TODO: fix hardcode
		let result = &[StackValType::I32];
		for i in 0..self.params.pop_size {
			let mut wg = WasmGenome::new(params, result);
			wg = self.init_genome.mutate(self.ctx.get_mut(), wg);
			self.pop.push(wg);
		}
	}

	fn new(
		params: GenAlgParams,
		problem: P,
		mutator: M,
		selector: S,
		init_genome: OnceMutator<WasmGenome, Context>,
		stop_cond: Box<dyn Predicate<WasmGenome, Context>>,
	) -> Self {
		let seed = params.seed.unwrap_or_else(|| thread_rng().gen());
		let log_file = params
			.log_file
			.clone()
			.unwrap_or_else(|| format!("trial_{}.log", seed));
		let size = params.pop_size;
		WasmGenAlg {
			problem,
			mutator,
			selector,
			init_genome,
			stop_cond,
			params,
			seed,
			log_file,

			// Runtime use
			engine: Engine::default(),
			ctx: RefCell::new(Context::new(seed)),
			pop: Vec::with_capacity(size),
		}
	}

	fn epoch_speciation() {
		todo!();
	}
}

impl<P, M, S> GenAlg for WasmGenAlg<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	type G = WasmGenome;
	type C = Context;

	fn epoch(&mut self) -> bool {
		log::info!("Evaluating generation {}.", self.ctx.borrow().generation);
		self.evaluate();

		self.pop
			.sort_unstable_by(|a, b| f64::partial_cmp(&b.fitness, &a.fitness).unwrap());

		{
			let mut ctx = self.ctx.borrow_mut();
			let pop = &self.pop[..]; // read-only, guaranteed for hooks to be sorted by fitness

			// Update context stats
			ctx.max_fitness = pop[0].fitness;
			ctx.avg_fitness = pop.iter().map(WasmGenome::fitness).sum::<f64>() / (pop.len() as f64);

			// Logging
			{
				log::info!("Average Fitness: {}", ctx.avg_fitness);
				log::info!("Top 10:");
				for p in self.pop.iter().take(10) {
					log::info!("\t{} <-- {}", p.fitness, p);
				}
				let filename = format!("{}/gen_{}.wasm", self.log_file, ctx.generation);
				let topguy = pop[0].emit();
				fs::write(filename, topguy);
			}

			// Test stop condition
			if self.stop_cond.test(&mut ctx, pop) {
				log::info!(
					"Stop condition met, exiting. Results are in trial_{}.log/",
					self.seed
				);
				return false;
			} else if ctx.generation >= self.params.num_generations {
				log::info!(
					"Completed {} generations. Results are in trial_{}.log/",
					ctx.generation,
					self.seed
				);
				return false;
			}

			// Update parameters for next generation
			ctx.generation += 1;
			self.selector.vary_params(&mut ctx, pop);
			self.selector.vary_params(&mut ctx, pop);
			// self.crossover.vary_params(&mut ctx, pop);

			log::info!("Creating generation {}.", ctx.generation);
		}

		let mut nextgen: Vec<Self::G> = Vec::with_capacity(self.params.pop_size);

		let elitism_cnt = (self.params.elitism_rate * (self.params.pop_size as f64)) as usize;
		let elites = if elitism_cnt > 0 {
			log::info!("Passing {elitism_cnt} elites to next generation.");
			self.pop[0..elitism_cnt].to_vec()
		} else {
			Vec::new()
		};

		//TODO speciation
		if self.params.enable_speciation {
			WasmGenAlg::<P, M, S>::epoch_speciation();
		}

		// TODO extract selection
		let mut selected = self
			.selector
			.select(self.ctx.get_mut(), mem::take(&mut self.pop)); // pop empty after selection
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
				let dist = Uniform::new(0, selected.len());
				(0..crossover_cnt)
					.map(|_| {
						let a = dist.sample(&mut ctx.rng);
						let b = dist.sample(&mut ctx.rng);
						(a, b)
					})
					.collect()
			};

			for (a, b) in indices {
				let child = self.crossover(&selected[a], &selected[b]);
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

		let mut ctx = self.ctx.borrow_mut();
		log::info!(
			"Created generation {} (size {}). {} new innovations; {} total innovations.",
			ctx.generation,
			self.params.pop_size,
			ctx.cur_innovs.len(),
			ctx.innov_cnt,
		);
		log::debug!("Innovations were: {:?}", ctx.cur_innovs);
		ctx.cur_innovs.clear();
		true
	}

	fn evaluate(&mut self) {
		let agents: Vec<_> = self
			.pop
			.iter() // OPT can I par_iter here? I think I need WasmGenome: Send / Sync
			.map(WasmGenome::emit)
			.map(|b| Agent::new(self.engine.clone(), &b))
			.collect();
		let fitnai: Vec<_> = agents.par_iter().map(|a| self.problem.fitness(a)).collect();
		for (g, f) in Iterator::zip(self.pop.iter_mut(), fitnai) {
			g.fitness = f;
		}
	}

	fn mutate(&self, indiv: Self::G) -> Self::G {
		let mut ctx = self.ctx.borrow_mut();
		self.mutator.mutate(&mut *ctx, indiv)
	}

	fn select(&self) -> HashSet<usize> {
		// let mut rng = self.rng.borrow_mut();
		// (0..self.pop_size)
		// 	.choose_multiple(&mut *rng, self.selection_cnt)
		// 	.into_iter()
		// 	.collect()
		todo!()
	}

	fn crossover(&self, par_a: &Self::G, par_b: &Self::G) -> Self::G {
		let mut ctx = self.ctx.borrow_mut();
		par_a.reproduce(par_b, &mut *ctx)
	}
}

pub struct WasmGenAlgBuilder<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	problem: Option<P>,
	pop_size: Option<usize>,
	selection: Option<S>,
	enable_elitism: Option<bool>,
	elitism_rate: Option<f64>,
	enable_crossover: Option<bool>,
	enable_speciation: Option<bool>,
	crossover_rate: Option<f64>,
	// crossover: Option<C>
	mutation_rate: Option<f64>,
	mutation: Option<M>,
	starter: Option<OnceMutator<WasmGenome, Context>>,
	stop_cond: Option<Box<dyn Predicate<WasmGenome, Context>>>,
	generations: Option<usize>,
	seed: Option<u64>,
	log_file: Option<String>,
}

impl<P, M, S> WasmGenAlgBuilder<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	pub fn build(self) -> WasmGenAlg<P, M, S> {
		let selection = self.selection.unwrap();
		let mutation = self.mutation.unwrap();
		let problem = self.problem.unwrap();
		let stop_cond = self
			.stop_cond
			.unwrap_or_else(|| Box::new(|_: &mut Context, _: &[WasmGenome]| true)); // optional
		let init_genome = self.starter.unwrap();

		let params = GenAlgParams {
			// TODO assert crossover_rate + elitism_rate <= 1.0
			pop_size: self.pop_size.unwrap(),

			elitism_rate: self.elitism_rate.unwrap_or_default(), // optional, default 0.0
			enable_speciation: self.enable_speciation.unwrap_or_default(), // optional, default false
			crossover_rate: self.crossover_rate.unwrap_or_default(), // optional, default 0.0
			mutation_rate: self.mutation_rate.unwrap(),          // TODO: use mutation rate
			num_generations: self.generations.unwrap(),
			seed: self.seed,
			log_file: self.log_file,
		};

		WasmGenAlg::new(params, problem, mutation, selection, init_genome, stop_cond)
	}

	pub fn problem(mut self, problem: P) -> Self {
		self.problem = Some(problem);
		self
	}

	pub fn pop_size(mut self, size: usize) -> Self {
		self.pop_size = Some(size);
		self
	}

	pub fn selection(mut self, sel: S) -> Self {
		self.selection = Some(sel);
		self
	}

	pub fn enable_elitism(mut self, en: bool) -> Self {
		self.enable_elitism = Some(en);
		self
	}

	pub fn elitism_rate(mut self, rate: f64) -> Self {
		assert!((0.0..=1.0).contains(&rate), "rate must be in [0, 1]");
		self.elitism_rate = Some(rate);
		self
	}

	pub fn enable_crossover(mut self, en: bool) -> Self {
		self.enable_crossover = Some(en);
		self
	}

	pub fn enable_speciation(mut self, en: bool) -> Self {
		self.enable_speciation = Some(en);
		self
	}

	pub fn crossover_rate(mut self, rate: f64) -> Self {
		assert!((0.0..=1.0).contains(&rate), "rate must be in [0, 1]");
		self.crossover_rate = Some(rate);
		self
	}

	pub fn mutation_rate(mut self, rate: f64) -> Self {
		assert!((0.0..=1.0).contains(&rate), "rate must be in [0, 1]");
		self.mutation_rate = Some(rate);
		self
	}

	pub fn mutation(mut self, mu: M) -> Self {
		self.mutation = Some(mu);
		self
	}

	pub fn init_genome(mut self, init: OnceMutator<WasmGenome, Context>) -> Self {
		self.starter = Some(init);
		self
	}

	pub fn stop_condition(mut self, pred: Box<dyn Predicate<WasmGenome, Context>>) -> Self {
		self.stop_cond = Some(pred);
		self
	}

	pub fn generations(mut self, gens: usize) -> Self {
		self.generations = Some(gens);
		self
	}

	pub fn seed(mut self, seed: u64) -> Self {
		self.seed = Some(seed);
		self
	}
}

impl<P, M, S> Default for WasmGenAlgBuilder<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	fn default() -> Self {
		Self {
			problem: None,
			pop_size: None,
			selection: None,
			enable_elitism: None,
			elitism_rate: None,
			enable_crossover: None,
			enable_speciation: None,
			crossover_rate: None,
			mutation_rate: None,
			mutation: None,
			starter: None,
			stop_cond: None,
			generations: None,
			seed: None,
			log_file: None,
		}
	}
}
