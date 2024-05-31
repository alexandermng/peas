//! Possible mutations to our program

use eyre::Result;
use rand::Rng;
use wasm_encoder::{CodeSection, Function, Module, ValType};
use wasmparser::{
	BinaryReader, BinaryReaderError, CodeSectionReader, FunctionBody, Operator, WasmFeatures,
};

/// Wasm Mutation
pub trait Mutation {
	fn mutate(&self, genes: &[u8], rng: &mut impl Rng) -> Result<Vec<u8>>;
}

pub struct NeutralAddInstr;
impl Mutation for NeutralAddInstr {
	fn mutate(&self, genes: &[u8], rng: &mut impl Rng) -> Result<Vec<u8>> {
		// TODO move to algorithm mutate()
		let reader = BinaryReader::new(genes, 0, WasmFeatures::all());
		let reader = CodeSectionReader::new(reader)?;

		// For now, we only have one function
		assert!(reader.count() == 1, "we only support one function for now");

		let fnreaders: Vec<FunctionBody> = reader
			.into_iter()
			.collect::<Result<Vec<_>, BinaryReaderError>>()?;

		let funny = &fnreaders[0];
		let locals = funny.get_locals_reader()?;
		let ops = funny.get_operators_reader()?;

		for op in ops.into_iter() {
			let op = op?;
			print!("op is {op:?}")
		}

		todo!()
	}
}

// TODO mutation sequence; mutation or; (mutation, rate)

// impl<I> Mutation for I
// where
// 	I: IntoIterator<Item = dyn Mutation>,
// {
// 	fn mutate(&self, genes: &[u8], rng: &mut impl Rng) -> Result<Vec<u8>> {
// 		todo!()
// 	}
// }
