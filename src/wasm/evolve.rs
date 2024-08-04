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

use crate::genetic::{
	self, GenAlg, Genome, Mutator, OnceMutator, Predicate, Problem, Selector, Solution,
};
use crate::wasm::{
	genome::{InnovNum, StackValType, WasmGene, WasmGenome},
	mutations::NeutralAddOp,
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
	pop_size: usize,
	selection: S,
	enable_elitism: bool,
	elitism_rate: f64,
	enable_crossover: bool,
	enable_speciation: bool,
	crossover_rate: f64,
	// crossover: C
	mutation_rate: f64,
	mutation: M,
	init_genome: OnceMutator<WasmGenome, Context>,
	stop_cond: Box<dyn Predicate<WasmGenome, Context>>,
	num_generations: usize, // maximum generations to go
	seed: u64,

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
		log::info!("Beginning GA trial with seed {}", self.seed);
		self.init();

		let dirname = format!("trial_{}.log", self.seed);
		fs::create_dir(dirname).unwrap();

		while self.epoch() {}
	}

	fn init(&mut self) {
		let params = &[StackValType::I32, StackValType::I32, StackValType::I32]; // TODO: fix hardcode
		let result = &[StackValType::I32];
		for i in 0..self.pop_size {
			let mut wg = WasmGenome::new(params, result);
			wg = self.init_genome.mutate(self.ctx.get_mut(), wg);
			self.pop.push(wg);
		}
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
			// pop[0] // best
			// 	.module
			// 	.borrow_mut()
			// 	.emit_wasm_file(filename) // still dont know why this is mut tbh
			// 	.unwrap();
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

		//TODO speciation
		if self.enable_speciation {
			todo!();
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
			let mut crossover_cnt = (self.crossover_rate * (self.pop_size as f64)) as usize;
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

		let mut ctx = self.ctx.borrow_mut();
		log::info!(
			"Created generation {} (size {}). {} new innovations; {} total innovations.",
			ctx.generation,
			self.pop_size,
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

	fn crossover(&self, par_a: &Self::G, par_b: &Self::G) -> Self::G {
		let len_a = par_a.len();
		let len_b = par_b.len();
		let mut matches: Vec<Range<usize>> = Vec::new(); // ranges into par_a
		let mut disjoint: Vec<(Range<usize>, Range<usize>)> = Vec::new(); // (a, b)

		let (mut cur_a, mut cur_b) = (0, 0); // current indices
		let (mut last_a, mut last_b) = (0, 0); // index of last matches or disjoints
		let mut matching = true; // true if currently matching, otherwise disjoint
		while cur_a < len_a && cur_b < len_b {
			if par_a[cur_a] == par_b[cur_b] {
				if !matching {
					disjoint.push((last_a..cur_a, last_b..cur_b)); // push disjoint range
					matching = true; // start matching range
					(last_a, last_b) = (cur_a, cur_b);
				}
				// cont matching range
				cur_a += 1;
				cur_b += 1;
			}
			// current disjoint
			if matching {
				matches.push(last_a..cur_a); // push matching range
				matching = false; // start disjoint range
				(last_a, last_b) = (cur_a, cur_b);
			}
			// cont disjoint range
			if let Some(mat_b) = par_b[last_b..].iter().position(|b| *b == par_a[cur_a]) {
				cur_b = mat_b; // found match, go next
				continue;
			}
			cur_b += 1;
		}
		if cur_a != len_a || cur_b != len_b {
			disjoint.push((cur_a..len_a, cur_b..len_b)); // at least one unfinished, disjoint at end
		}
		log::debug!("Found ranges: matching {matches:?} and disjoint {disjoint:?}");

		let mut ctx = self.ctx.borrow_mut();
		let mut child: Vec<WasmGene> = Vec::with_capacity(len_a);
		// will be [mat, dis, mat, dis,... mat?]
		for (mat, (dis_a, dis_b)) in
			Iterator::zip(matches.iter().cloned(), disjoint.iter().cloned())
		{
			child.extend(par_a[mat].iter().cloned()); // get matching
										  // then disjoint
			let genes_a = &par_a[dis_a];
			let genes_b = &par_b[dis_b];
			let choice = if ctx.rng.gen_bool(0.5) {
				genes_a
			} else {
				genes_b
			};
			// let choice = *[&par_a[dis_a], &par_b[dis_b]].choose(&mut ctx.rng).unwrap(); // choose one or the other
			child.extend(choice.iter().cloned());
		}
		if matches.len() > disjoint.len() {
			let last = matches.last().unwrap().clone();
			child.extend(par_a[last].iter().cloned());
		}

		WasmGenome {
			genes: child,
			fitness: 0.0,
			params: par_a.params.clone(),
			result: par_a.result.clone(),
			locals: par_a.locals.clone(),
		}
		/*
		first, extract the "genes" part of each wasm genome
		Then, starting at the first gene, repeat this algorithm:
		If the corresponding genes match, clone into the child
		If they do not match, check which one has the lower innovation number
		Iterate through the other genome until a matching gene is found
		If a matching gene is not found, repeat this step with the next lowest innovation number gene until a match is found
		Once a match is found, the captured genes are chosen based on which parent has the higher fitness (currently random) or randomly if fitness is equal
		*/
		// let len_a = par_a.len();
		// let len_b = par_b.len();
		// let mut child: Vec<WasmGene> = Vec::new();

		// let mut idx_a = 0; // cur idx in a
		// let mut last_a = 0; // idx of last match in a
		// let mut idx_b = 0; // cur idx in b
		// let mut last_b = 0; // idx of last match in b

		// while idx_a < len_a && idx_b < len_b {
		// 	if par_a[idx_a] == par_b[idx_b] {
		// 		// add match to child
		// 		child.push(par_a[idx_a]);
		// 		last_a = idx_a;
		// 		last_b = idx_b;
		// 		idx_a += 1;
		// 		idx_b += 1;
		// 	} else {
		// 		let mut found = false;
		// 		let mut temp_A = idx_a;
		// 		let mut temp_B = idx_b;

		// 		if par_a[idx_a].marker > par_b[idx_b].marker {
		// 			while idx_a < len_a && !found {
		// 				if WasmGene::eq(&par_a[idx_a], &par_b[idx_b]) {
		// 					found = true;
		// 				} else {
		// 					idx_a += 1;
		// 				}
		// 			}
		// 		} else {
		// 			while idx_b < len_b && !found {
		// 				if WasmGene::eq(&par_a[idx_a], &par_b[idx_b]) {
		// 					found = true;
		// 				} else {
		// 					idx_b += 1;
		// 				}
		// 			}
		// 		}

		// 		if !found {
		// 			idx_a = temp_A;
		// 			idx_b = temp_B;
		// 			if par_a[idx_a].marker > par_b[idx_b].marker {
		// 				idx_b += 1;
		// 			} else {
		// 				idx_a += 1;
		// 			}
		// 			if !WasmGene::eq(&par_a[idx_a], &par_b[idx_b]) {
		// 				continue;
		// 			}
		// 		}

		// 		//success
		// 		let mut rng = rand::thread_rng();
		// 		if rng.gen_bool(0.5) {
		// 			child.extend(par_a[last_a + 1..idx_a].iter().cloned());
		// 		} else {
		// 			child.extend(par_b[last_b + 1..idx_b].iter().cloned());
		// 		}
		// 		continue;
		// 	}
		// }

		// if idx_a < len_a {
		// 	child.extend(par_a[last_a + 1..len_a].iter().cloned());
		// } else if idx_b < len_b {
		// 	child.extend(par_b[last_b + 1..len_b].iter().cloned());
		// }

		// //can probably be replaced with "new" function after further work on that
		// WasmGenome {
		// 	genes: child,
		// 	fitness: 0.0,
		// 	params: todo!(),
		// 	result: todo!(),

		// 	locals: todo!(),
		// }
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
}

impl<P, M, S> WasmGenAlgBuilder<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	pub fn build(self) -> WasmGenAlg<P, M, S> {
		let size = self.pop_size.unwrap();
		let seed = self.seed.unwrap();
		WasmGenAlg {
			problem: self.problem.unwrap(),
			pop_size: size,
			selection: self.selection.unwrap(),
			enable_elitism: self.enable_elitism.unwrap_or_default(), // optional, default false
			elitism_rate: self.elitism_rate.unwrap_or_default(),     // optional, default 0.0
			enable_crossover: self.enable_crossover.unwrap(),
			enable_speciation: self.enable_speciation.unwrap_or_default(), // optional, default false
			crossover_rate: self.crossover_rate.unwrap_or_default(),       // optional, default 0.0
			mutation_rate: self.mutation_rate.unwrap(),                    // TODO: use mutation rate
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
		}
	}
}
