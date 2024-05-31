#![allow(unused)] // for now

pub mod genetic;
pub mod mutations;
pub mod selection;

use std::{borrow::BorrowMut, cell::RefCell, collections::HashSet, mem};

use rand::{
	seq::{IteratorRandom, SliceRandom},
	Rng, SeedableRng,
};
use rand_pcg::Pcg64Mcg;
use selection::Selector;
use walrus::{FunctionBuilder, LocalFunction, ValType};
use wasmtime::{Engine, Instance, Module, Store};
// use wasm_encoder::{
// 	CodeSection, Function, FunctionSection, Instruction, Module, TypeSection, ValType,
// };

use crate::genetic::{GenAlg, Genome, Problem, Solution};
use crate::mutations::{Mutation, NeutralAddInstr};

/// The genome of a Wasm agent/individual, with additional genetic data. Can generate a Wasm Agent: bytecode whose
/// phenotype is the solution for a particular problem.
#[derive(Debug)]
pub struct WasmGenome {
	module: walrus::Module, // in progress module
	func: LocalFunction,    // mutatable main function
	markers: Vec<usize>,    // markers for main, by instruction

	fitness: Option<f64>, // None if not run before
}

impl WasmGenome {
	fn new() -> Self {
		let config = walrus::ModuleConfig::new();
		let mut module = walrus::Module::with_config(config);
		let params = [ValType::I32]; // hardcode for now, TODO make it part of Problem
		let result = [ValType::I32];
		let mut func = FunctionBuilder::new(&mut module.types, &params, &result);
		let args: Vec<_> = params.iter().map(|&p| module.locals.add(p)).collect();
		let func = func.local_func(args);
		WasmGenome {
			module,
			func,
			markers: vec![], // TODO consider type
			fitness: None,
		}
	}

	fn emit(&self) -> Agent {
		todo!() // TODO
	}
}

impl Genome for WasmGenome {
	fn dist(&self, other: &Self) -> f64 {
		todo!() // use markers
	}
}

/// Assembled phenotype of an individual in a genetic algorithm. Used as a solution to a problem.
#[derive(Clone)]
pub struct Agent {
	pub engine: Engine,
	pub module: Module,
	// ^ todo consider access. also starting state to seed Store<_>
	pub fitness: f64,
}

// impl<P: Problem> Solution<P> for Agent {
// 	fn solve(&self, args: P::In) -> P::Out {
// 		todo!() // i can only do this if I can genericify it
// 	}
// }

/// A genetic algorithm for synthesizing WebAssembly modules.
pub struct WasmGA<P, M, S>
where
	P: Problem,
	M: Mutation,
	S: Selector<WasmGenome>,
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
	starter: Box<dyn Fn(&mut FunctionBuilder)>,
	num_generations: usize,
	seed: u64,

	// Runtime use
	generation: usize,
	rng: RefCell<Pcg64Mcg>,
	pop: Vec<WasmGenome>, // population of genomes
	agents: Vec<Agent>,   // corresponding agents
}

impl<P, M, S> WasmGA<P, M, S>
where
	P: Problem,
	M: Mutation,
	S: Selector<WasmGenome>,
{
	pub fn run(&mut self) {
		self.init();
		for n in 0..self.num_generations {
			self.epoch()

			// TODO logging
		}
	}

	fn init(&mut self) {
		for i in 0..self.pop_size {}
	}
}

impl<P, M, S> GenAlg for WasmGA<P, M, S>
where
	P: Problem,
	M: Mutation,
	S: Selector<WasmGenome>,
{
	type G = WasmGenome;

	fn epoch(&mut self) {
		self.evaluate();
		self.pop
			.sort_unstable_by(|a, b| f64::partial_cmp(&a.fitness, &b.fitness).unwrap()); // from now on, we're sorted

		let mut nextgen = Vec::with_capacity(self.pop_size);
		let selected = self.select(); // Selection

		// TODO elitism, skip all

		if self.do_crossover {
			// Crossover
			let mut rng = self.rng.borrow_mut();
			while nextgen.len() < nextgen.capacity() {
				let a = *selected.iter().choose(&mut *rng).unwrap();
				let b = *selected.iter().choose(&mut *rng).unwrap();
				nextgen.push(self.crossover(&self.pop[a], &self.pop[b]));
			}
		} else {
			for (i, v) in mem::take(&mut self.pop).into_iter().enumerate() {
				if selected.contains(&i) {
					nextgen.push(v)
				}
			}
		}

		// Mutation
		for gn in &mut nextgen {
			*gn = self.mutate(mem::take(gn));
		}

		// if no crossover, then must fill to cap with mutated variants
		while nextgen.len() < nextgen.capacity() {
			let indiv = {
				let mut rng = self.rng.borrow_mut();
				nextgen.choose(&mut *rng).unwrap().clone()
			};
			nextgen.push(self.mutate(indiv));
		}

		self.pop = nextgen;
	}

	fn evaluate(&mut self) {
		todo!() // TODO... evaluate Agent based on problem P
	}

	fn mutate(&self, indiv: Self::G) -> Self::G {
		let mut rng = self.rng.borrow_mut();
		let mutator = NeutralAddInstr {};
		mutator.mutate(&indiv.genes, &mut *rng);

		todo!()
	}

	fn select(&self) -> HashSet<usize> {
		let mut rng = self.rng.borrow_mut();
		(0..self.pop_size)
			.choose_multiple(&mut *rng, self.selection_cnt)
			.into_iter()
			.collect()
	}

	fn crossover(&self, a: &Self::G, b: &Self::G) -> Self::G {
		todo!()
	}
}

#[derive(Default)]
pub struct WasmGABuilder<P, M, S>
where
	P: Problem,
	M: Mutation,
	S: Selector<WasmGenome>,
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
	starter: Option<Box<dyn Fn(&mut FunctionBuilder)>>,
	generations: Option<usize>,
	seed: Option<u64>,
}

impl<P, M, S> WasmGABuilder<P, M, S>
where
	P: Problem,
	M: Mutation,
	S: Selector<WasmGenome>,
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
			starter: self.starter.unwrap(),
			num_generations: self.generations.unwrap(),
			seed,

			generation: 0,
			rng: RefCell::new(<Pcg64Mcg as SeedableRng>::seed_from_u64(seed)),
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

	pub fn starter(mut self, st: Box<dyn Fn(&mut FunctionBuilder)>) -> Self {
		self.starter = Some(st);
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
