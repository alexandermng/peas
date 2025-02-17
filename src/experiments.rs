pub mod speciation;

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
	genetic::{AsContext, GenAlg, Genome, Mutator, Selector},
	params::GenAlgParams,
};

/// An experiment to be run. Launches and manages multiple runs/trials.
pub trait Experiment {
	type Genome: Genome<Self::Ctx> + Default;
	type Ctx: AsContext;
	type ProblemSet: Serialize + for<'p> Deserialize<'p>; // Problem Set
	type MutationSet: Mutator<Self::Genome, Self::Ctx> + Serialize + for<'m> Deserialize<'m>;
	type SelectorSet: Selector<Self::Genome, Self::Ctx> + Serialize + for<'s> Deserialize<'s>;
	// type Results: Results + Serialize;
	type GA: GenAlg<G = Self::Genome, C = Self::Ctx>;

	// fn create_ga(&self, )

	/// Run the experiment
	fn run(&mut self);

	// TODO be able to save/restore from state
	// fn suspend()
	// fn resume()

	/// The base/control parameters for the run. Note that the seed must be set manually.
	fn base_params(
		&self,
	) -> GenAlgParams<Self::Genome, Self::Ctx, Self::MutationSet, Self::SelectorSet>;

	/// The output directory
	fn outdir(&self) -> &Path;
}

// pub struct ExperimentConfig {}
