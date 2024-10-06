#![allow(unused)] // for now

pub mod genetic;
pub mod params;
pub mod selection;
pub mod wasm {
	mod evolve;
	pub mod genome;
	pub mod graph;
	pub mod ir;
	pub mod mutations;

	pub use evolve::*;
	pub use genome::*;
}

pub mod prelude {
	pub use crate::genetic::{Problem, Solution};
	pub use crate::wasm::*;
}
