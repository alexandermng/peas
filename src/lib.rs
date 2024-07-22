#![allow(unused)] // for now

pub mod genetic;
pub mod selection;
pub mod wasm {
	mod evolve;
	pub mod genome;
	pub mod graph;
	pub mod mutations;
	// pub mod ir;

	pub use evolve::*;
	pub use genome::*;
}

pub use crate::wasm::*;
