//! WebAssembly Genome

use std::{
	borrow::Cow,
	cell::{Ref, RefCell},
	cmp,
	collections::HashSet,
	fmt::{Debug, Display},
	fs,
	ops::{Deref, DerefMut, Range, RangeBounds},
};

use eyre::{eyre, Result};
use petgraph::visit::IntoNodeReferences;
use rand::Rng;
use wasm_encoder::{
	CodeSection, Encode, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
	PrimitiveValType, TypeSection, ValType,
};

use crate::{
	genetic::{AsContext, Genome},
	wasm::ir::EncodeInContext,
};

use super::Context;
use super::{graph::GeneGraph, ir::ValType as StackValType};

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

/// A gene of the Wasm Genome, holding its type and historical marker.
#[derive(Clone)]
pub struct WasmGene<'a> {
	pub instr: Instruction<'a>,
	pub marker: InnovNum,

	// pub(crate) for now until new() stable
	pub(crate) popty: Cow<'a, [StackValType]>, // OPT: global append-only cache
	pub(crate) pushty: Cow<'a, [StackValType]>,
}

impl<'a> WasmGene<'a> {
	pub fn new(instr: Instruction<'a>, marker: InnovNum) -> Self {
		use Instruction::*;
		use StackValType::*;
		let (popty, pushty) = match &instr {
			I32Add | I32Sub | I32Mul | I32DivS | I32RemS | I32And | I32Or | I32Xor => {
				(vec![I32, I32], vec![I32])
			}
			I32Eqz => (vec![I32], vec![Bool]),
			I32Const(_) => (vec![], vec![I32]),
			// TODO fill rest... also consider adding an argument informing about current stack
			_ => unimplemented!("instruction type not supported"),
		};
		Self {
			instr,
			popty: popty.into(),
			pushty: pushty.into(),
			marker,
		}
	}

	/// Get the type of this instruction
	pub fn ty(&self) -> (&[StackValType], &[StackValType]) {
		(&self.popty, &self.pushty)
	}

	/// Check type-equality.
	pub fn ty_eq(&self, other: &Self) -> bool {
		Iterator::eq(self.popty.iter(), other.popty.iter())
			&& Iterator::eq(self.pushty.iter(), other.pushty.iter())
	}
}

impl<'a> PartialEq for WasmGene<'a> {
	fn eq(&self, other: &Self) -> bool {
		self.marker == other.marker
	}
}
impl<'a> Eq for WasmGene<'a> {}

impl<'a> Debug for WasmGene<'a> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Gene[{}]:", self.marker)?;
		self.instr.fmt(f)
	}
}

/// A comparison of two gene ranges among two genomes
#[derive(Debug, Clone)]
pub(crate) enum GeneDiff {
	Match(Range<usize>),
	Disjoint(Range<usize>, Range<usize>),
}

/// The genome of a Wasm agent/individual, with additional genetic data. Can generate a Wasm Agent: bytecode whose
/// phenotype is the solution for a particular problem.
#[derive(Clone, Debug, Default)]
pub struct WasmGenome {
	pub genes: GeneGraph,
	pub fitness: f64,
	pub params: Vec<StackValType>,
	pub result: Vec<StackValType>,

	pub(crate) locals: Vec<StackValType>, // local variable types. includes params
}

impl WasmGenome {
	/// Create a new WasmGenome with the given signature for the main function.
	pub fn new(params: &[StackValType], result: &[StackValType]) -> Self {
		let params = params.to_vec();
		let result = result.to_vec();
		let locals = params.clone();
		WasmGenome {
			genes: GeneGraph::new(),
			fitness: 0.0,
			params,
			result,

			locals,
		}
	}

	pub fn emit(&self) -> Vec<u8> {
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
			let locals = self
				.locals
				.iter()
				.enumerate()
				.map(|(i, &v)| (i as u32, v.into()));
			let mut func = Function::new(locals); // main
			let mut sink = func.into_raw_body();
			for node in self.genes.node_weights() {
				node.encode(&self, &mut sink);
				todo!();
				// add instructions from genome
				// func.instruction(&g.instr);
			}
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

	/// Show the gene intersection ranges of this genome and another. Returns a tuple of (matches, disjoint),
	/// where match ranges are of this genome's matching genes and disjoint ranges are a tuple of corresponding
	/// (self, other) genes. Genes will be [mat, dis, mat, dis... mat?], starting with 0..0 if it starts disjoint.
	pub(crate) fn diff(&self, other: &Self) -> Vec<GeneDiff> {
		return todo!();
		let len_a = self.len(); // par_a = self
		let len_b = other.len(); // par_b = other
		let mut diff = Vec::new();

		let (mut cur_a, mut cur_b) = (0, 0); // current indices
		let (mut last_a, mut last_b) = (0, 0); // index of last matches or disjoints
		let mut was_matching = true; // true if current range is matching, otherwise disjoint
		loop {
			let valid = cur_a < len_a && cur_b < len_b;
			let matching = valid && (self[cur_a] == other[cur_b]); // short-circuit (useless when invalid)
			match (valid, was_matching, matching) {
				/* invalid */
				(false, false, _) => {
					// out of bounds while disjoint
					diff.push(GeneDiff::Disjoint(last_a..len_a, last_b..len_b)); // final disjoint
					break;
				}
				(false, true, _) => {
					// out of bounds while matching
					diff.push(GeneDiff::Match(last_a..cur_a)); // final match
					if cur_a == len_a && cur_b == len_b {
						break; // clean finish, no extra disjoint bits
					}
					was_matching = false;
					(last_a, last_b) = (cur_a, cur_b);
					// pass to (false, false, _)
				}
				/* valid */
				(true, true, false) => {
					// was matching, but now disjoint
					// log::debug!("\t\tPushing match {last_a}..{cur_a}");
					diff.push(GeneDiff::Match(last_a..cur_a)); // push matching range
					was_matching = false; // start disjoint range
					(last_a, last_b) = (cur_a, cur_b);
					// pass to (_, false, false)
				}
				(true, false, true) => {
					// was disjoint, but now matching
					diff.push(GeneDiff::Disjoint(last_a..cur_a, last_b..cur_b)); // push disjoint range
															 // log::debug!("\t\tPushing disjoint ({last_a}..{cur_a}, {last_b}..{cur_b})");
					was_matching = true; // start matching range
					(last_a, last_b) = (cur_a, cur_b);
					// pass to (_, true, true)
				}
				(true, false, false) => {
					// cont valid disjoint
					if let Some(mat_b) = other[last_b..].iter().position(|b| *b == self[cur_a]) {
						cur_b = last_b + mat_b; // found match, go next
						debug_assert!(self[cur_a] == other[cur_b], "not real match");
						continue; // pass to (true, false, true)
					}
					cur_a += 1;
					// may invalidate a, => pass to (false, false, _)
					// else pass to (true, false, ?)
				}
				(true, true, true) => {
					// cont valid match
					cur_a += 1;
					cur_b += 1;
					// may invalidate a or b, => pass to (false, true, _)
					// else pass to (true, true, ?)
				}
			}
		}
		log::debug!("Found diff ranges for (0..{len_a}, 0..{len_b}): {diff:?}");
		diff
	}
}

// TODO move or make configurable
const DIST_COEFF_EXCESS: f64 = 0.3;
const DIST_COEFF_DISJOINT: f64 = 0.5;

impl Genome<Context> for WasmGenome {
	fn dist(&self, other: &Self) -> f64 {
		// NOTE: just edges? what about nodes? gotta check in NEAT impl.
		// OPT bitset? how to deal with highest common?
		let ge_a: HashSet<InnovNum> = self.genes.edge_weights().map(|w| w.marker).collect();
		let ge_b: HashSet<InnovNum> = other.genes.edge_weights().map(|w| w.marker).collect();
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
		let par_a = self.genes;
		let par_b = other.genes;
		let gn_a: HashSet<InnovNum> = par_a.node_weights()
		

		let mut child = par_a.clone();
		for nod in par_b.node_weights() {
			if 
		}
		self.genes.node_weights()

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
		f.write_str("Genome[")?;
		for b in self.emit() {
			write!(f, "{:X}", b)?;
		}
		f.write_str("]")
	}
}
