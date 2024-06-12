#![allow(unused)] // for now

pub mod genetic;
pub mod genome;
pub mod mutations;
pub mod selection;

use std::{
	any::Any,
	borrow::BorrowMut,
	cell::RefCell,
	collections::{HashMap, HashSet},
	fs,
	hash::{DefaultHasher, Hash, Hasher},
	mem,
};

use eyre::{eyre, DefaultHandler, OptionExt, Result};
use genetic::Predicate;
use genome::InnovNum;
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

#[derive(Debug)]
struct InnovKey(InnovNum, walrus::ir::Instr);
impl PartialEq for InnovKey {
	fn eq(&self, other: &Self) -> bool {
		let mut me = DefaultHasher::new();
		let mut them = me.clone();
		self.hash(&mut me);
		other.hash(&mut them);
		me.finish() == them.finish()
		// ehhhh we hope here :pray:
	}
}
impl Eq for InnovKey {}
impl Hash for InnovKey {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		self.0.hash(state);
		mem::discriminant(&self.1).hash(state);
		use walrus::ir;
		use walrus::ir::Instr::*;
		match self.1 {
			Binop(ir::Binop { op }) => mem::discriminant(&op).hash(state),
			Unop(ir::Unop { op }) => mem::discriminant(&op).hash(state),
			Const(ir::Const { value }) => {
				mem::discriminant(&value).hash(state);
				use ir::Value::*;
				match value {
					I32(v) => v.hash(state),
					I64(v) => v.hash(state),
					F32(v) => v.to_le_bytes().hash(state),
					F64(v) => v.to_le_bytes().hash(state),
					V128(v) => v.hash(state),
				}
			}
			LocalGet(ir::LocalGet { local }) => local.hash(state),
			LocalSet(ir::LocalSet { local }) => local.hash(state),
			LocalTee(ir::LocalTee { local }) => local.hash(state),
			GlobalGet(ir::GlobalGet { global }) => global.hash(state),
			GlobalSet(ir::GlobalSet { global }) => global.hash(state),
			Load(ir::Load { memory, kind, arg }) => {
				mem::discriminant(&kind).hash(state);
				memory.hash(state);
				arg.align.hash(state);
				arg.offset.hash(state);
			}
			_ => todo!("no hashy for you"), // this is so stupid
		};
	}
}

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
			innov_cnt: 0,
			cur_innovs: HashMap::new(),
		}
	}

	/// Get or assign an innovation number to an innovation keyed by location of mutation and instruction added
	pub fn innov(&mut self, loc: InnovNum, instr: walrus::ir::Instr) -> InnovNum {
		*self
			.cur_innovs
			.entry(InnovKey(loc, instr))
			.or_insert_with(|| {
				let out = self.innov_cnt;
				self.innov_cnt += 1;
				out
			})
	}

	pub fn num_innovs(&self) -> InnovNum {
		self.innov_cnt
	}
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
	stop_cond: Box<dyn Predicate<WasmGenome, Context>>,
	num_generations: usize, // maximum generations to go
	seed: u64,

	// Runtime use
	engine: Engine,
	ctx: RefCell<Context>,
	pop: Vec<WasmGenome>, // population of genomes
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
		log::info!("Beginning GA trial with seed {}", self.seed);
		self.init();

		let dirname = format!("trial_{}.log", self.seed);
		fs::create_dir(dirname).unwrap();

		while self.epoch() {}
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
			log::info!("Average Fitness: {}", ctx.avg_fitness);
			log::info!("Top 10:");
			for p in self.pop.iter().take(10) {
				log::info!("\t{} <-- {}", p.fitness, p);
			}
			let filename = format!("trial_{}.log/gen_{}.wasm", self.seed, ctx.generation);
			pop[0] // best
				.module
				.borrow_mut()
				.emit_wasm_file(filename) // still dont know why this is mut tbh
				.unwrap();
			// TODO logging hook

			// Test stop condition
			if self.stop_cond.test(&mut ctx, pop) {
				log::info!(
					"Stop condition met, exiting. Results are in trial_{}.log/",
					self.seed
				);
				return false;
			} else if ctx.generation >= self.num_generations {
				log::info!(
					"Completed {} generations. Results are in trial_{}.log/",
					ctx.generation,
					self.seed
				);
				return false;
			}

			// Update parameters for next generation
			ctx.generation += 1;
			self.selection.vary_params(&mut ctx, pop);
			self.mutation.vary_params(&mut ctx, pop);
			// self.crossover.vary_params(&mut ctx, pop);

			log::info!("Creating generation {}.", ctx.generation);
		}

		let mut nextgen: Vec<Self::G> = Vec::with_capacity(self.pop_size);

		let elitism_cnt = (self.elitism_rate * (self.pop_size as f64)) as usize;
		if self.enable_elitism {
			log::info!("Passing {elitism_cnt} elites to next generation.");
			for i in 0..elitism_cnt {
				nextgen.push(self.pop[i].clone());
			}
		}

		// TODO extract selection
		let mut selected = self
			.selection
			.select(self.ctx.get_mut(), mem::take(&mut self.pop)); // pop empty after selection
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
			self.pop = nextgen[0..elitism_cnt].to_vec(); // clone elites
		}
		let needed = self.pop_size - self.pop.len() - nextgen.len(); // amount needed
		log::info!(
			"Mutating {} unique genomes (+{needed} copy-variants).",
			nextgen.len()
		);
		let fill: Vec<_> = nextgen
			.choose_multiple(&mut self.ctx.get_mut().rng, needed)
			.cloned()
			.collect(); // fill to capacity
		nextgen.extend(fill);
		nextgen = nextgen
			// .into_par_iter() // OPT
			.into_iter()
			.map(|g| self.mutate(g))
			.collect(); // OPT- collect into pop directly? rust stupid

		// Add to next generation!
		self.pop.append(&mut nextgen);
		assert!(self.pop.len() == self.pop_size, "should be fully populated");

		log::info!(
			"Created generation {} (size {}).",
			self.ctx.get_mut().generation,
			self.pop_size
		);
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
	stop_cond: Option<Box<dyn Predicate<WasmGenome, Context>>>,
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
			elitism_rate: self.elitism_rate.unwrap_or_default(), // optional
			enable_crossover: self.enable_crossover.unwrap(),
			crossover_rate: self.crossover_rate.unwrap_or_default(), // optional
			mutation_rate: self.mutation_rate.unwrap(),
			mutation: self.mutation.unwrap(),
			init_genome: self.starter.unwrap(),
			stop_cond: self
				.stop_cond
				.unwrap_or_else(|| Box::new(|_: &mut Context, _: &[WasmGenome]| true)), // optional
			num_generations: self.generations.unwrap(),
			seed,

			// Runtime use
			engine: Engine::default(),
			ctx: RefCell::new(Context::new(seed)),
			pop: Vec::with_capacity(size),
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
			elitism_rate: None,
			enable_crossover: None,
			crossover_rate: None,
			mutation_rate: None,
			mutation: None,
			starter: None,
			stop_cond: None,
			generations: None,
			seed: None,
		}
	}
}
