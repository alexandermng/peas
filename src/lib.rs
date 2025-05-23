#![allow(unused)] // for now

pub mod genetic;
pub mod params;
pub mod selection;
pub mod wasm {
	mod evolve;
	pub mod genome;
	pub mod graph;
	pub mod mutations;
	pub mod species;
	// pub mod ir;

	pub use evolve::*;
	pub use genome::*;
}
pub mod experiments;
pub mod problems;

pub mod prelude {
	pub use crate::problems::{Problem, Solution};
	pub use crate::wasm::*;
}

use std::hash::Hash;

use derive_more::derive::{Display, From, Into};

/// Identification type for unique identification from a backing store.
#[derive(Debug, Display)]
#[display("{index}")]
pub struct Id<T> {
	index: usize,

	_mark: std::marker::PhantomData<T>,
}

impl<T> Copy for Id<T> {}
impl<T> Clone for Id<T> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<T> PartialEq for Id<T> {
	fn eq(&self, other: &Self) -> bool {
		self.index == other.index
	}
}
impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		self.index.hash(state);
	}
}

impl<T> From<usize> for Id<T> {
	fn from(index: usize) -> Self {
		Self {
			index,
			_mark: std::marker::PhantomData,
		}
	}
}

impl<T> From<Id<T>> for usize {
	fn from(id: Id<T>) -> usize {
		id.index
	}
}
