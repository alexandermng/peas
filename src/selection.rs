//! Selection Operator Implementations
//! See [genetic::Selector] for more info.

use rand::{distributions::Uniform, Rng};

use crate::{
	genetic::{Genome, Selector},
	Context,
};

/// Tournament Selection
/// https://en.wikipedia.org/wiki/Tournament_selection
pub struct TournamentSelection {
	pub k: usize, // tournament size
	pub p: f64,   // probability rate
}

impl<G: Genome> Selector<G, Context> for TournamentSelection {
	fn select<'a>(&self, ctx: &mut Context, pop: &'a [G]) -> Vec<&'a G> {
		let unif = Uniform::new(0, 10);
		// ctx.rng.sample(unif)
		todo!() // TODO impl
	}
}
