#![allow(unused)] // for now

pub mod genetic;
pub mod genome;
pub mod mutations;
pub mod selection;

use std::{borrow::BorrowMut, cell::RefCell, collections::HashSet, fs, mem};

use eyre::{eyre, OptionExt, Result};
use rand::{
	seq::{IteratorRandom, SliceRandom},
	Rng, SeedableRng,
};
use rand_pcg::Pcg64Mcg;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use walrus::{CustomSection, FunctionBuilder, LocalFunction, ValType};
use wasmtime::{Engine, Instance, Linker, Module, Store, WasmParams, WasmResults, WasmTy};
// use wasm_encoder::{
// 	CodeSection, Function, FunctionSection, Instruction, Module, TypeSection, ValType,
// };

pub use crate::genome::WasmGenome;

use crate::genetic::{GenAlg, Genome, Mutator, Problem, Selector, Solution};
use crate::mutations::NeutralAddOp;

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
		let module = Module::from_binary(&engine, binary).unwrap();
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

pub struct Context {
	pub generation: usize,
	pub(crate) rng: Pcg64Mcg,
}

/// A genetic algorithm for synthesizing WebAssembly modules.
pub struct WasmGA<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	// Parameters
	problem: P,
	pop_size: usize,
	selection: S,
	enable_elitism: bool,
	elitism_rate: f64,
	enable_crossover: bool,
	crossover_rate: f64,
	// crossover: C
	mutation_rate: f64,
	mutation: M,
	init_genome: Box<dyn Mutator<WasmGenome, Context>>,
	num_generations: usize,
	seed: u64,

	// Runtime use
	engine: Engine,
	ctx: RefCell<Context>,
	pop: Vec<WasmGenome>, // population of genomes
	agents: Vec<Agent>,   // corresponding agents
}

impl<P, M, S> WasmGA<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	pub fn run(&mut self) {
		self.init();

		let dirname = format!("trial_{}.log", self.seed);
		fs::create_dir(dirname).unwrap();

		for n in 0..self.num_generations {
			log::info!("Starting generation {}.", self.ctx.borrow().generation);

			self.epoch();

			// TODO logging hooks
			log::info!("Finished generation {}.", self.ctx.borrow().generation);
			self.ctx.get_mut().generation += 1;

			// TODO end condition / max fitness
		}
	}

	fn init(&mut self) {
		let params = &[ValType::I32, ValType::I32, ValType::I32]; // hardcoded for now, TODO fix
		let result = &[ValType::I32];
		for i in 0..self.pop_size {
			let mut wg = WasmGenome::new(params, result);
			wg = self.init_genome.mutate(self.ctx.get_mut(), wg);
			self.pop.push(wg);
		}
	}
}

impl<P, M, S> GenAlg for WasmGA<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	type G = WasmGenome;

	fn epoch(&mut self) {
		self.evaluate();

		self.pop
			.sort_unstable_by(|a, b| f64::partial_cmp(&b.fitness, &a.fitness).unwrap());

		log::info!("Top 10:");
		for p in self.pop.iter().take(10) {
			log::info!("\t{} <-- {}", p.fitness, p);
		}
		let filename = format!(
			"trial_{}.log/gen_{}.wasm",
			self.seed,
			self.ctx.borrow().generation
		);
		self.pop[0]
			.module
			.get_mut()
			.emit_wasm_file(filename)
			.unwrap();

		let mut nextgen: Vec<WasmGenome> = Vec::with_capacity(self.pop_size);

		let elitism_cnt = (self.elitism_rate * (self.pop_size as f64)) as usize;
		if self.enable_elitism {
			log::info!("Passing {elitism_cnt} elites to next generation.");
			for i in 0..elitism_cnt {
				nextgen.push(self.pop[i].clone());
			}
		}

		// TODO extract selection
		// self.selection.select(self.ctx.get_mut(), &self.pop)
		let mut selected: Vec<WasmGenome> = mem::take(&mut self.pop) // pop empty after selection
			.into_iter()
			.choose_multiple(&mut self.ctx.get_mut().rng, self.pop_size - nextgen.len());
		log::info!(
			"Selected {} individuals from current population.",
			selected.len()
		);

		if self.enable_crossover {
			let crossover_cnt = (self.crossover_rate * (self.pop_size as f64)) as usize;
			log::info!("Crossing over {crossover_cnt} individuals.");
			todo!() // TODO crossover selected
		} else {
			nextgen.append(&mut selected);
		}

		// Mutation
		if self.enable_elitism {
			self.pop = nextgen.drain(0..elitism_cnt).collect();
		}
		log::info!("Mutating {} genomes.", nextgen.len());
		nextgen = nextgen
			// .into_par_iter() // OPT
			.into_iter()
			.map(|g| self.mutate(g))
			.collect(); // OPT- collect into pop directly? rust stupid

		// Add to next generation!
		self.pop.append(&mut nextgen);
	}

	fn evaluate(&mut self) {
		// NOTE: we may not actually need self.agents here lol...
		self.agents = self
			.pop
			.iter() // OPT can I par_iter here?
			.map(WasmGenome::emit)
			.map(|b| Agent::new(self.engine.clone(), &b))
			.collect();
		let fitnai: Vec<_> = self
			.agents
			.par_iter()
			.map(|a| self.problem.fitness(a))
			.collect();
		for (g, f) in Iterator::zip(self.pop.iter_mut(), fitnai) {
			g.fitness = f;
		}
	}

	fn mutate(&self, indiv: Self::G) -> Self::G {
		let mut ctx = self.ctx.borrow_mut();
		self.mutation.mutate(&mut *ctx, indiv)
	}

	fn select(&self) -> HashSet<usize> {
		// let mut rng = self.rng.borrow_mut();
		// (0..self.pop_size)
		// 	.choose_multiple(&mut *rng, self.selection_cnt)
		// 	.into_iter()
		// 	.collect()
		todo!()
	}

	fn crossover(&self, a: &Self::G, b: &Self::G) -> Self::G {
		todo!()
	}
}

pub struct WasmGABuilder<P, M, S>
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
	crossover_rate: Option<f64>,
	// crossover: Option<C>
	mutation_rate: Option<f64>,
	mutation: Option<M>,
	starter: Option<Box<dyn Mutator<WasmGenome, Context>>>,
	generations: Option<usize>,
	seed: Option<u64>,
}

impl<P, M, S> WasmGABuilder<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	pub fn build(self) -> WasmGA<P, M, S> {
		let size = self.pop_size.unwrap();
		let seed = self.seed.unwrap();
		WasmGA {
			problem: self.problem.unwrap(),
			pop_size: size,
			selection: self.selection.unwrap(),
			enable_elitism: self.enable_elitism.unwrap(),
			elitism_rate: self.elitism_rate.unwrap(),
			enable_crossover: self.enable_crossover.unwrap(),
			crossover_rate: self.crossover_rate.unwrap(),
			mutation_rate: self.mutation_rate.unwrap(),
			mutation: self.mutation.unwrap(),
			init_genome: self.starter.unwrap(),
			num_generations: self.generations.unwrap(),
			seed,

			// Runtime use
			engine: Engine::default(),
			ctx: RefCell::new(Context {
				generation: 0,
				rng: <Pcg64Mcg as SeedableRng>::seed_from_u64(seed),
			}),
			pop: Vec::with_capacity(size),
			agents: Vec::with_capacity(size),
		}
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
		self.elitism_rate = Some(rate);
		self
	}

	pub fn enable_crossover(mut self, en: bool) -> Self {
		self.enable_crossover = Some(en);
		self
	}

	pub fn crossover_rate(mut self, rate: f64) -> Self {
		self.crossover_rate = Some(rate);
		self
	}

	pub fn mutation_rate(mut self, rate: f64) -> Self {
		self.mutation_rate = Some(rate);
		self
	}

	pub fn mutation(mut self, mu: M) -> Self {
		self.mutation = Some(mu);
		self
	}

	pub fn init_genome(mut self, init: Box<dyn Mutator<WasmGenome, Context>>) -> Self {
		self.starter = Some(init);
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

impl<P, M, S> Default for WasmGABuilder<P, M, S>
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
			elitism_rate: Some(0.0),
			enable_crossover: None,
			crossover_rate: Some(0.0),
			mutation_rate: None,
			mutation: None,
			starter: None,
			generations: None,
			seed: None,
		}
	}
}
