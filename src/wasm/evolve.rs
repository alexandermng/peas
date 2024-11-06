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

use bon::{builder, Builder};
use csv::Writer;
use eyre::{eyre, DefaultHandler, Error, OptionExt, Result, WrapErr};
use rand::{
	distributions::Uniform,
	prelude::Distribution,
	seq::{index::sample, IteratorRandom, SliceRandom},
	thread_rng, Rng, SeedableRng,
};
use rand_pcg::Pcg64Mcg;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use wasm_encoder::{Instruction, ValType};
use wasmtime::{Engine, Instance, Linker, Module, Store, WasmParams, WasmResults, WasmTy};

use crate::{
	genetic::AsContext,
	params::{GenAlgParams, GenAlgParamsOpts},
	selection::TournamentSelection,
	wasm::{
		genome::{InnovNum, StackValType, WasmGene, WasmGenome},
		mutations::AddOperation,
	},
};
use crate::{
	genetic::{self, GenAlg, Genome, Mutator, OnceMutator, Predicate, Problem, Selector, Solution},
	wasm::GeneDiff,
};

use super::{
	mutations::{MutationLog, WasmMutation},
	WasmGenomeRecord,
};

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
	fn try_exec(&self, args: P::In) -> eyre::Result<P::Out> {
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

type WasmGenAlgParams<M, S> = GenAlgParams<WasmGenome, Context, M, S>;

/// A genetic algorithm for synthesizing WebAssembly modules to solve a problem. The problem must specify
/// a fitness function. Parametrized across the problem, the mutation type, and the selection type.
pub struct WasmGenAlg<P, M, S>
where
	P: Problem,
	M: Mutator<WasmGenome, Context>,
	S: Selector<WasmGenome, Context>,
{
	// Parameters
	pub problem: P,
	pub params: WasmGenAlgParams<M, S>,

	mutators: Vec<M>,                                   // copied from params
	selector: S,                                        // copied from params
	init_genome: WasmGenome,                            // copied from params
	stop_cond: Box<dyn Predicate<WasmGenome, Context>>, // TODO... idk fix

	seed: u64, // copied from params

	// Runtime use
	engine: Engine,
	ctx: RefCell<Context>,
	pop: Vec<WasmGenome>, // population of genomes
	records: Vec<WasmGenomeRecord>,
}

impl<P, M, S> WasmGenAlg<P, M, S>
where
	P: Problem + Sync,
	M: Mutator<WasmGenome, Context> + Clone + for<'m> Deserialize<'m> + Serialize,
	S: Selector<WasmGenome, Context> + Clone + for<'s> Deserialize<'s> + Serialize,
	P::In: WasmParams,
	P::Out: WasmResults,
{
	pub fn new(params: WasmGenAlgParams<M, S>, problem: P) -> Self {
		let size = params.pop_size;
		let seed = params.seed;
		let mutators = params.mutators.clone();
		let selector = params.selector.clone();
		let init_genome = params.init_genome.clone();
		let stop_cond: Box<dyn Predicate<WasmGenome, Context>> = match params.max_fitness {
			Some(x) => Box::new(move |ctx: &mut Context, _: &[WasmGenome]| -> bool {
				ctx.max_fitness >= x
			}),
			None => Box::new(|_: &mut Context, _: &[WasmGenome]| false),
		};
		WasmGenAlg {
			problem,
			params,

			mutators,
			selector,
			init_genome,
			stop_cond,
			seed,

			// Runtime use
			engine: Engine::default(),
			ctx: RefCell::new(Context::new(seed)),
			pop: Vec::with_capacity(size),
			records: Vec::new(),
		}
	}

	pub fn run(&mut self) {
		log::info!("Beginning GA trial with seed {}", self.seed);

		fs::create_dir(&self.params.output_dir).unwrap();

		self.pop
			.extend((0..self.params.pop_size).map(|_| self.init_genome.clone()));

		while self.epoch() {}

		// config.toml
		let configfile = format!("{}/config.toml", self.params.output_dir);
		let config = toml::to_string(&self.params).expect("params should serialize");
		fs::write(configfile, config);

		// data.csv
		let csvfile = format!("{}/{}", self.params.output_dir, self.params.datafile);
		let mut wtr = Writer::from_path(csvfile).unwrap();
		for rec in &self.records {
			wtr.serialize(rec);
		}
		wtr.flush().unwrap();
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

			// Records
			for (i, p) in pop.iter().enumerate() {
				let id = ctx.generation * self.params.pop_size + i; // TODO move id into genome itself; assign at reproduction-time
				self.records.push(WasmGenomeRecord::new(id, p));
			}

			// Logging
			{
				log::info!("Average Fitness: {}", ctx.avg_fitness);
				log::info!("Top 10:");
				for p in pop.iter().take(10) {
					log::info!("\t{} <-- {}", p.fitness, p);
				}
				let filename = format!("{}/gen_{}.wasm", self.params.output_dir, ctx.generation);
				let topguy = pop[0].emit();
				fs::write(filename, topguy);
			}

			// Test stop condition
			if self.stop_cond.test(&mut ctx, pop) {
				log::info!(
					"Stop condition met, exiting. Results are in ./{}",
					self.params.output_dir
				);
				return false;
			} else if ctx.generation >= self.params.num_generations {
				log::info!(
					"Completed {} generations. Results are in ./{}",
					ctx.generation,
					self.params.output_dir
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
			todo!();
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
		self.mutators
			.choose(ctx.rng())
			.expect("non-empty mutators")
			.mutate(&mut *ctx, indiv)
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
