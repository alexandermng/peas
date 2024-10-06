//! Intermediate Representations
//!
//! Our modeling of Wasm. Acts as a subset of our supported operations.
//!

use wasm_encoder::{Instruction, ValType as WasmValType};

use super::WasmGenome;

/// The type of a value
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValType {
	Bool,
	U8, // unsigned integer
	I8, // signed integer
	U32,
	I32,
	U64,
	I64,
	F32, // float
	F64,
	// TODO mem/ref types
}

impl From<ValType> for WasmValType {
	fn from(value: ValType) -> Self {
		use ValType::*;
		match value {
			Bool | U8 | I8 | U32 | I32 => WasmValType::I32,
			U64 | I64 => WasmValType::I64,
			F32 => WasmValType::F32,
			F64 => WasmValType::F64,
		}
	}
}

/// Encode some items given a module context
pub trait EncodeInContext<C = WasmGenome> {
	fn encode(&self, ctx: &C, sink: &mut Vec<u8>);
}

/// Create an instruction given a module context
pub trait IntoInstruction<C = WasmGenome> {
	fn into_instr<'a>(self, ctx: &'a C) -> Instruction<'a>;
}

#[derive(Clone, Copy, Debug)]
pub enum I32Op {
	Add,
	Sub,
	Mul,
	And,
	Or,
	// Xor,
}

impl IntoInstruction for I32Op {
	fn into_instr<'a>(self, ctx: &'a WasmGenome) -> Instruction<'a> {
		match self {
			I32Op::Add => Instruction::I32Add,
			I32Op::Sub => Instruction::I32Sub,
			I32Op::Mul => Instruction::I32Mul,
			I32Op::And => Instruction::I32And,
			I32Op::Or => Instruction::I32Or,
			// I32Op::Xor => Instruction::I32Xor,
		}
	}
}
