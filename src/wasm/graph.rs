//! Graph representation of the genome.
//!
//! Layers similar to a CFG-over-SSA type graph. Nodes are sequences of instructions, typed by inputs and outputs.
//! Outputs are stored as variables.

use petgraph::prelude::*;
use wasm_encoder::{BlockType, Encode, Instruction};

use super::{
	ir::{EncodeInContext, I32Op, IntoInstruction, ValType},
	InnovNum, WasmGenome,
};

pub type GeneGraph = DiGraph<GeneNode, VarEdge>;
// TODO: encode.

/// A node in a graph representing a gene.
#[derive(Clone, Debug)]
struct GeneNode<I: IntoInstruction = I32Op> {
	pub instrs: Vec<I>,
	pub marker: InnovNum,
	// OPT can we have this in sorted order the same as the graph? uses more mem but mbe more efficient?
	// or one global graph?
	params: Vec<ValType>, // pop type
	ret: Vec<ValType>,    // push type
}

impl GeneNode {
	pub fn ty(&self) -> (&[ValType], &[ValType]) {
		(&self.params, &self.ret)
	}

	pub fn ty_eq(&self, other: &Self) -> bool {
		Iterator::eq(self.params.iter(), other.params.iter())
			&& Iterator::eq(self.ret.iter(), other.ret.iter())
	}
}

impl EncodeInContext for GeneNode {
	fn encode(&self, ctx: &WasmGenome, sink: &mut Vec<u8>) {
		Instruction::Block(BlockType::Empty).encode(sink);
		for i in &self.instrs {
			i.into_instr(&ctx).encode(sink);
		}
		Instruction::End.encode(sink);
	}
}

impl PartialEq for GeneNode {
	fn eq(&self, other: &Self) -> bool {
		self.marker == other.marker
	}
}
impl Eq for GeneNode {}
impl PartialOrd for GeneNode {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		usize::partial_cmp(&self.marker, &other.marker)
	}
}
impl Ord for GeneNode {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		usize::cmp(&self.marker, &other.marker)
	}
}

/// Edge representing an output variable of its node source,
/// being passed to its node destination.
#[derive(Clone, Debug)]
struct VarEdge {
	pub marker: InnovNum, // unique for each edge
	pub local_idx: usize, // index of variable in function locals
	pub enabled: bool,
}
