use crate::{genetic::Species, Id};

use super::{Context, WasmGenome, WasmGenomeId};

pub type WasmSpeciesId = Id<WasmSpecies>;

#[derive(Debug, Clone)]
pub struct WasmSpecies {
	/// representative genome of the current generation
	pub representative: WasmGenomeId,
	/// current members
	pub members: Vec<WasmGenomeId>,
	/// sum of all members' adjusted fitnesses
	pub fitness: f64,
	/// maximum number of members for next generation, as determined by the genalg
	pub capacity: usize,

	/// when this species was created
	pub starting_generation: usize,

	pub(crate) archive: Vec<WasmGenomeId>, // record of all past members
}

impl Species<WasmGenome, Context> for WasmSpecies {
	fn add_genome(&mut self, g: WasmGenomeId) {
		self.archive.push(g);
		self.members.push(g);
	}

	fn adjusted_fitness(&self, fitness: f64) -> f64 {
		// explicit fitness sharing (Goldberg and Richardson, 1987)
		fitness / self.size() as f64
	}

	fn fitness(&self) -> f64 {
		self.fitness
	}

	fn representative(&self) -> WasmGenomeId {
		self.members[0]
	}

	fn size(&self) -> usize {
		self.members.len()
	}
}
