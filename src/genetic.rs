//! Genetic Algorithm Types

use std::collections::HashSet;

/// Represents a genetic algorithm, which must implement these genetic operators, for a given Genome type.
/// It holds parameters for these operators and keeps track of its current population through generations.
pub trait GeneticAlg {
	type G: Genome;

	/// Executes an epoch/generation by performing:
	/// 1. Evaluation/Simulation, calculating fitnesses
	/// 2. Selection
	/// 3. Crossover, if enabled
	/// 4. Mutation
	fn epoch(&mut self);

	// Evaluates the population against the given problem/simulation, updating fitnesses.
	fn evaluate(&mut self);

	/// Calculates the fitness of an individual after evaluation.
	fn fitness(&self, indiv: &Self::G) -> f64;

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
}
