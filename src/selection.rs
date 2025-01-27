//! Selection Operator Implementations
//! See [genetic::Selector] for more info.

use std::ops::Index;

use rand::{
	distributions::{Bernoulli, Distribution, Uniform},
	seq::{IteratorRandom, SliceRandom},
	Rng,
};
use serde::{Deserialize, Serialize};

use crate::{
	genetic::{AsContext, Genome, Selector},
	wasm::Context,
	Id,
};

/// Tournament Selection
/// https://en.wikipedia.org/wiki/Tournament_selection
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TournamentSelection {
	pub rate: f64,     // selection rate determining selection pressure (in 0..1)
	k: usize,          // tournament size
	p: f64,            // probability rate
	replacement: bool, // whether to replace chosen individuals
}

impl TournamentSelection {
	pub fn new(rate: f64, tournament_size: usize, probability: f64, replacement: bool) -> Self {
		assert!(
			(0.0..=1.0).contains(&rate),
			"selection rate must be in [0,1]"
		);
		assert!(
			(0.0..=1.0).contains(&probability),
			"probability rate must be in [0,1]"
		);
		Self {
			rate,
			k: tournament_size,
			p: probability,
			replacement,
		}
	}
}

impl<G, C> Selector<G, C> for TournamentSelection
where
	G: Genome<C> + Clone,
	C: AsContext + Index<Id<G>, Output = G>,
{
	fn select(&self, ctx: &mut C, mut pop: Vec<Id<G>>) -> Vec<Id<G>> {
		let mut out = vec![];
		let bern = Bernoulli::new(self.p).unwrap();
		let selection_cnt = (self.rate * (pop.len() as f64)) as usize;
		while out.len() < selection_cnt {
			let mut tourney: Vec<_> = pop
				.iter() // length k (or however many left)
				.copied()
				.enumerate()
				.choose_multiple(ctx.rng(), self.k);
			tourney.sort_unstable_by(|(_, a), (_, b)| {
				f64::total_cmp(&ctx[*b].fitness(), &ctx[*a].fitness())
			});
			let winner = (0..tourney.len())
				.find(|_| bern.sample(ctx.rng())) // first with chance p, second with p*(1-p), ...
				.unwrap_or(tourney.len() - 1);
			let winner = tourney[winner].0; // actual index
			let winner = if self.replacement {
				pop[winner]
			} else {
				pop.swap_remove(winner)
			};
			out.push(winner);
		}
		out
	}
}
