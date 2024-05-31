use rand::Rng;

use crate::genetic::Genome;

/// The selection operator in a Genetic Algorithm
pub trait Selector<G: Genome> {
	type Iter: for<'a> Iterator<Item = &'a G>; // TODO figure out
	fn pull(&self, rng: &mut impl Rng, pop: &[G]) -> Self::Iter;
}
