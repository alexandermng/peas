//! Genetic Algorithm Types

use std::{cell::RefCell, marker::PhantomData, path::Path};

use downcast_rs::{impl_downcast, Downcast};
use rand::Rng;

use crate::Id;

/// Represents a genetic algorithm, which must implement these genetic operators, for a given Genome type.
/// It holds parameters for these operators and keeps track of its current population through generations.
pub trait GenAlg {
	type G: Genome<Self::C>;
	type C: AsContext;

	/// Run the full algorithm.
	fn run(&mut self) -> Vec<DynResults<Self::G, Self::C>>;

	// fn reset(&mut self);

	/// Executes an epoch/generation by performing:
	/// 1. Evaluation/Simulation, calculating fitnesses
	/// 2. Selection
	/// 3. Crossover, if enabled
	/// 4. Mutation
	///
	/// Returns whether the algorithm should continue (ie. false if stop condition was triggered).
	fn epoch(&mut self) -> bool;

	// Evaluates the population against the given problem/simulation, updating fitnesses.
	fn evaluate(&mut self);

	/// Generates a mutated version of an individual.
	fn mutate(&self, indiv: Id<Self::G>) -> Id<Self::G>;

	/// Selects the parents for the next generation.
	fn select(&self, pop: &[Id<Self::G>]) -> Vec<Id<Self::G>>;

	/// Crosses over / recombines two parent individuals to generate a new child individual.
	fn crossover(&self, a: Id<Self::G>, b: Id<Self::G>) -> Self::G;
}
pub trait ConfiguredGenAlg<C>: GenAlg + Configurable<C> {}
impl<T, C> ConfiguredGenAlg<C> for T where T: GenAlg + Configurable<C> {}

pub trait Configurable<C> {
	type Output: GenAlg + ?Sized;
	/// Creates the object from a config
	fn from_config(config: C) -> Box<Self::Output>;

	/// Recreates the config of the object
	fn gen_config(&self) -> C;

	/// Outputs the config to a file
	fn log_config(&mut self);
}

/// A species in a genetic algorithm
pub trait Species<G, C>
where
	G: Genome<C>,
	C: AsContext,
{
	// fn epoch() ???

	// NOTE: uses the enclosing genalg's selection
	// fn select(&mut self, ctx: &mut C, num: usize) -> Vec<Id<G>>;

	// NOTE: uses the enclosing genalg's reproduction
	// fn crossover(&mut self, num: usize, parents: Vec<&G>) -> Vec<G>;

	fn add_genome(&mut self, g: Id<G>);

	/// Adjust individual fitnesses based on average fitness of species
	fn adjusted_fitness(&self, fitness: f64) -> f64;

	fn fitness(&self) -> f64;
	fn representative(&self) -> Id<G>;
	fn size(&self) -> usize;
}

/// The Genome of an individual in a Genetic Algorithm.
pub trait Genome<C>
where
	C: AsContext,
{
	/// The distance between two Genomes, used to measure compatibility for crossover.
	fn dist(&self, other: &Self) -> f64;

	/// The fitness of this Genome after evaluated.
	fn fitness(&self) -> f64;

	/// Crossover with another parent into a new offspring.
	fn reproduce(&self, other: &Self, ctx: &mut C) -> Self;

	/// Crossover with another parent into a new offspring, consuming this parent.
	fn reproduce_into(self, other: &Self, ctx: &mut C) -> Self
	where
		Self: Sized,
	{
		self.reproduce(other, ctx)
	}
}

/// Genetic Algorithm Context
pub trait AsContext {
	/// Get the RNG
	fn rng(&mut self) -> &mut impl Rng;

	/// Current generation number
	fn generation(&self) -> usize;

	// TODO: params?

	// TODO: require Index<Id<G>, Output = G>?
}

// TODO add conversion
// impl<T> AsContext for T
// where
// 	T: AsMut<AsContext>,
// {
// 	fn rng(&mut self) -> &mut impl Rng {
// 		self.as_mut().rng()
// 	}

// 	fn generation(&mut self) -> usize {
// 		self
// 	}
// }

/// Aggregates results from the run. Define hooks to record data.
pub trait Results: Downcast {
	type Genome: Genome<Self::Ctx>;
	type Ctx: AsContext;

	/// Initialize and create enclosing files. Start any timers here.
	fn initialize(&mut self, ctx: &mut Self::Ctx) {}

	/// Called once every generation, after evaluation and before crafting of next generation.
	fn record_generation(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>]) {}

	/// Called once every generation, after evaluation and before crafting of next generation.
	/// TODO fix by adding a default associated type or smthn
	fn record_speciation(
		&mut self,
		ctx: &mut Self::Ctx,
		species: &[Id<crate::wasm::species::WasmSpecies>],
	) {
	}

	/// Called upon the algorithm hitting its stop condition. Not called when algorithm completes specified generations.
	fn record_success(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>]) {}

	/// Finalize results and write to files. `outdir` should contain any artefacts.
	fn finalize(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>], outdir: &Path) {}
}
impl_downcast!(Results assoc Genome, Ctx where Genome: crate::genetic::Genome<Ctx>, Ctx: AsContext);

pub type DynResults<G, C> = Box<dyn Results<Genome = G, Ctx = C> + Send + Sync>;

// /// Default impl for Results. See trait-level docs.
// #[derive(Default, Debug, Serialize, Clone)]
// pub struct DefaultResults<G> {
// 	success: bool, // whether it found a solution
// 	num_generations: usize, // how many generations it ran for
// 	max_fitnesses: Vec<u64>, // top fitness for each generation
// 	avg_fitnesses: Vec<u64>, // mean fitness for each generation
// }

// impl<G> Results<G> for DefaultResults<G> {
// 	fn record_generation(&mut self, alg: &mut G) {
// 			// TODO need alg to expose population
// 	}

// 	fn record_success(&mut self, alg: &mut G) {
// 			self.success = true;
// 	}

// 	fn finalize(&mut self, alg: &mut G) {
// 			self.num_generations = 0; // TODO need alg to expose params
// 	}
// }

/// The selection operator in a Genetic Algorithm. To be called once per generation, with optional
/// parameter variation after evaluation each generation.
pub trait Selector<G, C>
where
	G: Genome<C>,
	C: AsContext,
{
	fn select(&self, ctx: &mut C, pop: Vec<Id<G>>) -> Vec<Id<G>>;
	fn vary_params(&mut self, ctx: &mut C, pop: &[Id<G>]) {}
}

/// The mutation operator in a Genetic Algorithm. Called per-individual, with optional parameter
/// variation after evaluation each generation.
pub trait Mutator<G, C>
where
	G: Genome<C>,
	C: AsContext,
{
	fn mutate(&self, ctx: &mut C, indiv: G) -> G;
	fn vary_params(&mut self, ctx: &mut C, pop: &[Id<G>]) {}
}

// /// The crossover operator in a Genetic Algorithm. Called per-pair, with optional parameter variation
// /// after evaluation each generation.
// pub trait Recombiner<G, C>
// where
// 	G: Genome<C>,
// 	C: AsContext,
// {
// 	fn crossover(&self, ctx: &mut C, par_a: G, par_b: &G) -> G;
// 	fn vary_params(&mut self, ctx: &mut C, pop: &[G]) {}
// }

/// View and determine something about a Genetic Algorithm. Used for stop conditions.
pub trait Predicate<G, C>
where
	G: Genome<C>,
	C: AsContext,
{
	fn test(&mut self, ctx: &mut C, pop: &[Id<G>]) -> bool;
}

/// View the Genetic Algorithm in-progress. Used for logging.
pub trait Peeker<G, C>
where
	G: Genome<C>,
	C: AsContext,
{
	fn peek(&mut self, ctx: &mut C, pop: &[Id<G>]);
}

/// Mutator meant to be called only once (noop after). Used for gene initialization.
pub struct OnceMutator<G, C, T = Box<dyn FnMut(&mut C, G) -> G>>
where
	G: Genome<C>,
	C: AsContext,
	T: FnMut(&mut C, G) -> G,
{
	inner: RefCell<T>,
	_phan: PhantomData<fn(&mut C, G) -> G>,
}

impl<T, G, C> OnceMutator<G, C, T>
where
	G: Genome<C>,
	C: AsContext,
	T: FnMut(&mut C, G) -> G,
{
	pub fn new(mutator: T) -> Self {
		Self {
			inner: RefCell::new(mutator),
			_phan: PhantomData,
		}
	}
}

impl<T, G, C> Mutator<G, C> for OnceMutator<G, C, T>
where
	G: Genome<C>,
	C: AsContext,
	T: FnMut(&mut C, G) -> G,
{
	/// Runs closure on first mutation, and is a no-op on subsequent calls.
	fn mutate(&self, ctx: &mut C, indiv: G) -> G {
		let mut f = self.inner.borrow_mut();
		(f)(ctx, indiv)
	}
}

impl<T, G, C> From<T> for OnceMutator<G, C>
where
	G: Genome<C>,
	C: AsContext,
	T: FnMut(&mut C, G) -> G + 'static,
{
	fn from(value: T) -> Self {
		let value: Box<dyn FnMut(&mut C, G) -> G> = Box::new(value);
		OnceMutator::new(value)
	}
}

/***** Blanket Impls *****/

/* bleh, no specialization... */

// impl<T, C, G> Selector<G, C> for T
// where
// 	G: Genome,
// 	T: Fn(&mut C, Vec<G>) -> Vec<G>,
// {
// 	fn select(&self, ctx: &mut C, pop: Vec<G>) -> Vec<G> {
// 		(self)(ctx, pop)
// 	}
// }

// impl<T, C, G> Mutator<G, C> for T
// where
// 	G: Genome,
// 	T: Fn(&mut C, G) -> G,
// {
// 	fn mutate(&self, ctx: &mut C, indiv: G) -> G {
// 		(self)(ctx, indiv)
// 	}
// }

// impl<T, C, G> Recombiner<G, C> for T
// where
// 	G: Genome,
// 	T: Fn(&mut C, G, &G) -> G,
// {
// 	fn crossover(&self, ctx: &mut C, par_a: G, par_b: &G) -> G {
// 		(self)(ctx, par_a, par_b)
// 	}
// }

impl<T, C, G> Predicate<G, C> for T
where
	G: Genome<C>,
	C: AsContext,
	T: FnMut(&mut C, &[Id<G>]) -> bool,
{
	fn test(&mut self, ctx: &mut C, pop: &[Id<G>]) -> bool {
		(self)(ctx, pop)
	}
}

impl<T, C, G> Peeker<G, C> for T
where
	G: Genome<C>,
	C: AsContext,
	T: FnMut(&mut C, &[Id<G>]),
{
	fn peek(&mut self, ctx: &mut C, pop: &[Id<G>]) {
		(self)(ctx, pop);
	}
}
