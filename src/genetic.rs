//! Genetic Algorithm Types

use std::collections::HashSet;

use rand::Rng;

/// Represents a genetic algorithm, which must implement these genetic operators, for a given Genome type.
/// It holds parameters for these operators and keeps track of its current population through generations.
pub trait GenAlg {
	type G: Genome;

	/// Executes an epoch/generation by performing:
	/// 1. Evaluation/Simulation, calculating fitnesses
	/// 2. Selection
	/// 3. Crossover, if enabled
	/// 4. Mutation
	fn epoch(&mut self);

	// Evaluates the population against the given problem/simulation, updating fitnesses.
	fn evaluate(&mut self);

	/// Consumes and generates a mutated version of an individual.
	fn mutate(&self, indiv: Self::G) -> Self::G;

	/// Selects the parents for the next generation. Returns the indices of the individuals to select.
	fn select(&self) -> HashSet<usize>;

	/// Crosses over / recombines two parent individuals to generate a new child individual.
	fn crossover(&self, a: &Self::G, b: &Self::G) -> Self::G;
}

/// The Genome of an individual in a Genetic Algorithm.
pub trait Genome {
	/// The distance between two Genomes, used to measure compatibility for crossover.
	fn dist(&self, other: &Self) -> f64;

	// TODO add fitness
}

/// Represents a task or problem to be solved by a WebAssembly module. Should contain
/// problem parameters and necessary training data for evaluation.
pub trait Problem {
	type In; // type of inputs/arguments to the Agent (e.g. (i32, i32) )
	type Out; // type of outputs/results from the Agent (e.g. i32 )
		  // can add stuff like externals later

	/// Calculates a Solution's fitness, defined per-problem
	fn fitness(&self, soln: impl Solution<Self>) -> f64
	where
		Self: Sized; // no "dyn Problem"s
}

/// A solution to a given problem.
pub trait Solution<P: Problem>: Sync {
	/// Works the problem given the input arguments, returning the output
	fn exec(&self, args: P::In) -> P::Out;
}

/// The selection operator in a Genetic Algorithm
pub trait Selector<G: Genome, C> {
	fn select<'a>(&self, ctx: &mut C, pop: &'a [G]) -> Vec<&'a G>;
}

/// The mutation operator in a Genetic Algorithm
pub trait Mutator<G: Genome, C> {
	fn mutate(&self, ctx: &mut C, indiv: G) -> G;
}

/// The crossover operator in a Genetic Algorithm
pub trait Recombiner<G: Genome, C> {
	fn crossover(&self, ctx: &mut C, par_a: G, par_b: &G) -> G;
}
