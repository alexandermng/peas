//! Graph representation of the genome.
//!
//! Layers similar to a CFG-over-SSA type graph. Nodes are sequences of instructions, typed by inputs and outputs.
//! Outputs are stored as variables.

use std::ops::{Deref, DerefMut};

use petgraph::{algo::toposort, prelude::*};
use wasm_encoder::{BlockType, Encode, Instruction};

use super::{
	ir::{EncodeInContext, I32Op, IntoInstruction, ValType},
	Context, EdgeMarker, InnovNum, NodeMarker, WasmGenome,
};

/// Acts as the core of a Wasm Genome, a graph that emits Wasm.
#[derive(Clone, Default, Debug)]
pub struct GeneGraph {
	inner: DiGraphMap<NodeMarker, VarEdge>,
	sources: Vec<NodeMarker>, // special parameter input nodes to the function
	sinks: Vec<NodeMarker>, // special parameter output nodes from the function, at least 1 (return value)
}

impl GeneGraph {
	// pub fn from_externals(sources: Vec<NodeMarker>, sinks: Vec<NodeMarker>) -> Self {
	// 	Self {
	// 		inner: DiGraphMap::new(),
	// 		sources,
	// 		sinks,
	// 	}
	// }
}

impl Deref for GeneGraph {
	type Target = DiGraphMap<NodeMarker, VarEdge>;
	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}
impl DerefMut for GeneGraph {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

impl EncodeInContext<(&Context, &WasmGenome)> for GeneGraph {
	fn encode(&self, ctx: &(&Context, &WasmGenome), sink: &mut Vec<u8>) {
		let &(ctx, genome) = ctx;
		let graph = &self.inner;
		let nodes = toposort(graph, None).expect("should have no cycles");
		for id in nodes {
			let node = genome.get_node(ctx, id);
			for (_, _, e) in graph.edges_directed(id, Incoming) {
				e.encode(genome, sink); // push params
			}
			match node {
				Node::Gene(g) => {
					g.encode(genome, sink); // leaves output on stack
					Instruction::LocalSet((*id) as u32).encode(sink);
				}
				Node::Param(_) => {}
				Node::Result => {} // whatever's left on stack
			}
		}
	}
}

/// A Genetic Algorithm's backing pool for Nodes. Found from the genetic algorithm context,
/// with the index acting as the innovation number
pub type GeneNodePool = Vec<Node>;

/// Node data in a GeneGraph. There are special paramater source and result sink nodes,
#[derive(Clone, Debug)]
pub enum Node {
	Gene(GeneNode),
	Param(u32), // special source/input node: resolves inline to a `local.get idx` push to stack
	Result, // special unique sink/output node: must be placed last in compilation; only has one input

	        // Source(usize), // special source node: resolves inline to `global.get idx`
	        // Sink(usize),   // special node: resolves inline to a `global.set idx`
}

impl EncodeInContext for Node {
	fn encode(&self, ctx: &WasmGenome, sink: &mut Vec<u8>) {
		match self {
			Node::Gene(g) => g.encode(ctx, sink),
			Node::Param(idx) => Instruction::LocalGet(*idx).encode(sink),
			Node::Result => Instruction::Nop.encode(sink),
		};
	}
}

/// A node in a graph representing a gene. Found in a `GeneNodePool`, indexed by innovation number.
/// Encodes to a block which leaves one value on the stack.
#[derive(Clone, Debug)]
pub struct GeneNode<I: IntoInstruction = I32Op> {
	pub instrs: Vec<I>,
	pub params: Vec<ValType>, // pop type
	pub ret: Vec<ValType>,    // push type
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
		Instruction::Block(BlockType::Result(self.ret[0].into())).encode(sink);
		for i in &self.instrs {
			i.into_instr(&ctx).encode(sink);
		}
		Instruction::End.encode(sink);
	}
}

/// Edge representing an output variable of its node source,
/// being passed to its node destination.
#[derive(Clone, Debug)]
struct VarEdge {
	pub marker: EdgeMarker, // unique for each edge
	pub ty: ValType,        // type of variable
	pub local_idx: u32, // index of variable in function locals (aligns with the NodeMarker it's from)
	pub enabled: bool,
}

impl EncodeInContext for VarEdge {
	fn encode(&self, ctx: &WasmGenome, sink: &mut Vec<u8>) {
		if !self.enabled {
			return;
		}
		Instruction::LocalGet(self.local_idx).encode(sink);
	}
}
