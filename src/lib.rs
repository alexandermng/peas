#![allow(unused)] // for now

pub mod genetic;

use std::{cell::RefCell, collections::HashSet, mem};

use fastrand::Rng;
use genetic::{GeneticAlg, Genome};
use wasm_encoder::{
	CodeSection, Function, FunctionSection, Instruction, Module, TypeSection, ValType,
};

/// The genome of a Wasm agent/individual, with additional genetic data. Can generate a Wasm Agent: bytecode whose
/// phenotype is the solution for a particular problem.
#[derive(Clone, Debug, Default)]
pub struct WasmGenome {
	genes: Vec<u8>, // flat genome
	fitness: f64,   // associated fitness after being run
}

impl WasmGenome {
	/* nothing */
}

impl Genome for WasmGenome {
	fn dist(&self, other: &Self) -> f64 {
		todo!() // use markers
	}
}

/// A genetic algorithm for synthesizing WebAssembly modules.
pub struct WasmGA {
	// Parameters
	pub seed: u64,            // rng seed for entire alg
	pub pop_size: usize,      // fixed population size
	pub selection_cnt: usize, // count of how many to select for crossover
	pub do_crossover: bool,   // whether to do crossover or not
	pub do_elitism: bool,     // whether to preserve some number of top performers without mutation
	pub elitism_cnt: usize,   // count of how many to carry unmodified to next generation

	// Runtime use
	rng: RefCell<Rng>,
	pop: Vec<WasmGenome>, // population
}

impl WasmGA {
	pub fn new(seed: u64, pop_size: usize, selection_cnt: usize, elitism_cnt: usize) -> Self {
		WasmGA {
			seed,
			rng: RefCell::new(Rng::with_seed(seed)),
			pop_size,
			pop: Vec::new(),
			selection_cnt,
			do_crossover: false,
			do_elitism: (elitism_cnt > 0),
			elitism_cnt,
		}
	}
}

impl GeneticAlg for WasmGA {
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
				let a = *rng.choice(&selected).unwrap();
				let b = *rng.choice(&selected).unwrap();
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
				rng.choice(&mut nextgen).unwrap().clone()
			};
			nextgen.push(self.mutate(indiv));
		}

		self.pop = nextgen;
	}

	fn evaluate(&mut self) {
		todo!()
	}

	fn fitness(&self, indiv: &Self::G) -> f64 {
		indiv.fitness
	}

	fn mutate(&self, indiv: Self::G) -> Self::G {
		todo!()
	}

	fn select(&self) -> HashSet<usize> {
		todo!()
	}

	fn crossover(&self, a: &Self::G, b: &Self::G) -> Self::G {
		todo!()
	}
}
