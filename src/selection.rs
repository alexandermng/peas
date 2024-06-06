//! Selection Operator Implementations
//! See [genetic::Selector] for more info.

use rand::Rng;

use crate::genetic::{Genome, Selector};

/// Tournament Selection
/// https://en.wikipedia.org/wiki/Tournament_selection
pub struct TournamentSelection {
	pub k: usize, // tournament size
	pub p: f64,   // probability rate
}

impl<G: Genome, C> Selector<G, C> for TournamentSelection {
	fn select<'a>(&self, ctx: &mut C, pop: &'a [G]) -> Vec<&'a G> {
		todo!() // TODO impl
	}
}
