use crate::genetic::Species;

use super::{Context, WasmGenome};

type SpeciesId = usize;

/// Represents a Species of WasmGenomes.
struct WasmSpecies {
	id: SpeciesId,
	representative: WasmGenome,
	avg_fitness: f64,
	top_fitness: f64,

	pub members: Vec<WasmGenome>,
}

impl Species<WasmGenome, Context> for WasmSpecies {
	fn select(&mut self, num: usize) -> Vec<&WasmGenome> {
		todo!()
	}

	fn crossover(&mut self, num: usize, parents: Vec<&WasmGenome>) -> Vec<WasmGenome> {
		todo!()
	}

	fn add_genome(&mut self, g: WasmGenome) {
		todo!()
	}

	fn adjust_fitness(&mut self) {
		todo!()
	}

	fn fitness(&self) -> f64 {
		todo!()
	}

	fn representative(&self) -> &WasmGenome {
		todo!()
	}

	fn size(&self) -> usize {
		todo!()
	}
}
