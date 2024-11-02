//! WebAssembly Genome

use std::{
	borrow::Cow,
	cell::{Ref, RefCell},
	cmp,
	collections::{HashMap, HashSet},
	fmt::{Debug, Display},
	fs,
	ops::{Deref, DerefMut, Range, RangeBounds},
};

use eyre::{eyre, Result};
use petgraph::{
	graph::NodeIndex,
	visit::{IntoEdgeReferences, IntoNodeReferences, Visitable},
};
use rand::Rng;
use wasm_encoder::{
	CodeSection, Encode, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
	PrimitiveValType, TypeSection, ValType,
};

use crate::{
	genetic::{AsContext, Genome},
	wasm::ir::EncodeInContext,
};

use super::{graph::GeneNode, Context};
use super::{
	graph::{GeneGraph, Node},
	ir::{I32Op, ValType as StackValType},
};

pub type NodeMarker = InnovNum;

pub type EdgeMarker = InnovNum;

/// A global unique id within a Genetic Algorithm
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct InnovNum(pub usize);
impl Deref for InnovNum {
	type Target = usize;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}
impl DerefMut for InnovNum {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}
impl Display for InnovNum {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		Display::fmt(&self.0, f)
	}
}

/// The genome of a Wasm agent/individual, with additional genetic data. Can generate a Wasm Agent: bytecode whose
/// phenotype is the solution for a particular problem.
#[derive(Clone, Debug)]
pub struct WasmGenome {
	pub genes: GeneGraph,
	pub fitness: f64,
	pub params: Vec<StackValType>,
	pub result: Vec<StackValType>,

	// pub(crate) locals: Vec<StackValType>, // local variable types. includes params
	locals: Vec<NodeMarker>, // output of each Node as a local variable
}

impl WasmGenome {
	/// Create a new WasmGenome with the given signature for the main function.
	pub fn new(ctx: &mut Context, params: &[StackValType], result: &[StackValType]) -> Self {
		let params = params.to_vec();
		let result = result.to_vec();
		let mut out = WasmGenome {
			// ...hardcode. MUST update context on init before first gen
			genes: GeneGraph::default(),
			fitness: 0.0,
			params,
			result,
			locals: Vec::new(),
		};
		for i in 0..out.params.len() {
			out.add_node(ctx, Node::Param(i as u32));
		}
		out.add_node(ctx, Node::Result);
		out
	}

	pub fn get_node<'a>(&self, ctx: &'a Context, innov: NodeMarker) -> &'a Node {
		&ctx.node_pool[*innov]
	}

	/// Adds a Node to this structure, backed in the context. Returns the NodeMarker associated with it.
	pub fn add_node(&mut self, ctx: &mut Context, node: Node) -> NodeMarker {
		let innovnum = InnovNum(ctx.node_pool.len());
		match &node {
			Node::Gene(_) | Node::Param(_) => self.locals.push(innovnum), // note locals has 1-1 corresp with NodeMarkers, so locals[innov] works
			Node::Result => {}
		};
		ctx.node_pool.push(node);
		innovnum
	}

	pub fn emit(&self, ctx: &Context) -> Vec<u8> {
		let mut modu = Module::new();
		let types = {
			let mut ts = TypeSection::new();
			ts.function(
				self.params.iter().map(|&v| v.into()),
				self.result.iter().map(|&v| v.into()),
			);
			ts
		};
		let funcidx = 0;
		let funcs = {
			let mut fs = FunctionSection::new();
			fs.function(funcidx);
			fs
		};
		let expos = {
			let mut es = ExportSection::new();
			es.export("main", ExportKind::Func, 0);
			es
		};
		let codes = {
			let mut cs = CodeSection::new();
			let locals = self.locals.iter().enumerate().map(|(i, &v)| {
				(
					i as u32,
					match &ctx.node_pool[*v] {
						Node::Gene(g) => g.ret[0],
						Node::Param(p) => self.params[*p as usize],
						Node::Result => self.result[0], // may be unused tbh
					}
					.into(), // my ValType into encoder ValType
				)
			});
			let mut func = Function::new(locals); // main
			let mut sink = func.into_raw_body();
			self.genes.encode(&(ctx, self), &mut sink);
			Instruction::End.encode(&mut sink);
			cs.raw(&sink);
			// cs.function(&func);
			cs
		};
		modu.section(&types)
			.section(&funcs)
			.section(&expos)
			.section(&codes);
		let out = modu.finish();
		#[cfg(debug_assertions)]
		if let Err(e) = wasmparser::validate(&out) {
			log::error!("Outputting to ./invalid.wasm");
			fs::write("invalid.wasm", &out).unwrap();
			panic!("{e:?}");
		}
		out
	}
}

// TODO move or make configurable
const DIST_COEFF_EXCESS: f64 = 0.3;
const DIST_COEFF_DISJOINT: f64 = 0.5;

impl Genome<Context> for WasmGenome {
	fn dist(&self, other: &Self) -> f64 {
		// NOTE: just edges? what about nodes? gotta check in NEAT impl.
		// OPT bitset? how to deal with highest common?
		let ge_a: HashSet<InnovNum> = self
			.genes
			.edge_references()
			.map(|(_, _, w)| w.marker)
			.collect();
		let ge_b: HashSet<InnovNum> = other
			.genes
			.edge_references()
			.map(|(_, _, w)| w.marker)
			.collect();
		let highest_common = *ge_a.union(&ge_b).max().expect("graph non-empty");

		let n = cmp::max(ge_a.len(), ge_b.len()) as f64;
		let (disjoint_ge, excess_ge) =
			ge_a.symmetric_difference(&ge_b)
				.fold((0.0, 0.0), |acc, &i| {
					if i > highest_common {
						(acc.0, acc.1 + 1.0)
					} else {
						(acc.0 + 1.0, acc.1)
					}
				});

		// delta = c1*Excess/N + c2*Disjoint/N
		DIST_COEFF_EXCESS * excess_ge / n + DIST_COEFF_DISJOINT * disjoint_ge / n
	}

	fn fitness(&self) -> f64 {
		self.fitness
	}

	fn reproduce(&self, other: &Self, mut ctx: &mut Context) -> Self {
		log::debug!("Crossing over:\n\ta = {self:?}\n\tb = {other:?}");
		let par_a = &self.genes;
		let par_b = &other.genes;
		// let gn_a: HashMap<InnovNum, NodeIndex<u32>> = par_a
		// 	.node_references()
		// 	.map(|(idx, innov)| (innov, idx))
		// 	.collect();

		// TODO edit to only include the disjoint from the MORE FIT parent (self/par_a).
		let mut child = par_a.clone();
		child.extend(par_b.edge_references());

		WasmGenome {
			genes: child,
			fitness: 0.0,
			params: self.params.clone(),
			result: self.result.clone(),
			locals: self.locals.clone(), // TODO fix
		}
	}
}

impl Display for WasmGenome {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Genome[ todo ]")
		// for b in self.emit() {
		// 	write!(f, "{:X}", b)?;
		// }
	}
}
