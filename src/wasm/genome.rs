//! WebAssembly Genome

use std::{
	borrow::Cow,
	cell::{Ref, RefCell},
	cmp,
	fmt::{Debug, Display},
	fs,
	ops::{Deref, DerefMut, Range, RangeBounds},
};

use eyre::{eyre, Context as _, Result};
use rand::Rng;
use serde::Serialize;
use wasm_encoder::{
	CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
	PrimitiveValType, TypeSection, ValType,
};
use wasmparser::{Operator, Parser};

use crate::genetic::{AsContext, Genome};

use super::Context;

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

/// The type of a value on the stack
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StackValType {
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

impl From<StackValType> for ValType {
	fn from(value: StackValType) -> Self {
		use StackValType::*;
		match value {
			Bool | U8 | I8 | U32 | I32 => ValType::I32,
			U64 | I64 => ValType::I64,
			F32 => ValType::F32,
			F64 => ValType::F64,
		}
	}
}

//mapping isn't one-to-one so idk if this is fine
impl From<ValType> for StackValType {
	fn from(value: ValType) -> Self {
		match value {
			ValType::I32 => StackValType::I32,
			ValType::I64 => StackValType::I64,
			ValType::F32 => StackValType::I32,
			ValType::F64 => StackValType::I64,
			_ => todo!(),
		}
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
	pub genes: Vec<WasmGene<'static>>,
	pub fitness: f64,      // the fitness of the resulting Agent
	pub generation: usize, // the generation this was created (+1 from its parents)

	pub params: Vec<StackValType>,
	pub result: Vec<StackValType>,
	pub(crate) locals: Vec<StackValType>, // local variable types. includes params
}

impl WasmGenome {
	/// Create a new WasmGenome with the given signature for the main function.
	pub fn new(generation: usize, params: &[StackValType], result: &[StackValType]) -> Self {
		let params = params.to_vec();
		let result = result.to_vec();
		let locals = params.clone();
		WasmGenome {
			genes: Vec::new(),
			fitness: 0.0,
			generation,

			params,
			result,
			locals,
		}
	}

	pub fn from_binary(binary: &[u8]) -> Result<Self> {
		let mut genes = Vec::new();
		let mut params = Vec::new();
		let mut result = Vec::new();
		let mut locals = Vec::new();

		let par = Parser::new(0);
		for payload in par.parse_all(binary) {
			use wasmparser::Payload;
			match payload? {
				Payload::CodeSectionStart { count, range, size } => {}
				Payload::CodeSectionEntry(body) => {
					let oprdr = body.get_operators_reader()?;
					for (i, op) in oprdr.into_iter().enumerate() {
						let instr =
							Instruction::try_from(op?).wrap_err("could not convert instruction")?;
						// let instr: Instruction<'static> = instr.to_owned();
						println!("Instr is {instr:?}");
						// genes.push(WasmGene::new(instr, InnovNum(i)));
						todo!() // idk why this doesn't work...
					}
					for loc in body.get_locals_reader()? {
						// TODO
					}
				}
				Payload::End(_) => break,
				_ => {}
			};
		}
		// TODO: 1. find export "main" and use type to determine params/result;
		//			 2. parse locals
		//			 3. parse code section

		Ok(WasmGenome {
			genes,
			fitness: 0.0,
			generation: 0,
			params,
			result,
			locals,
		})
	}

	/// Insert genes into the genome after a specified index.
	pub fn insert<I>(&mut self, idx: usize, genes: I)
	where
		I: IntoIterator<Item = WasmGene<'static>>,
	{
		let _: Vec<_> = self.genes.splice((idx + 1)..(idx + 1), genes).collect();
	}

	/// Replace genes in a range with the given genes. Returns removed genes.
	pub fn replace<R, I>(&mut self, range: R, genes: I) -> Vec<WasmGene<'static>>
	where
		R: RangeBounds<usize>,
		I: IntoIterator<Item = WasmGene<'static>>,
	{
		self.genes.splice(range, genes).collect()
	}

	// /// Retrieve the local position by global innovation number.
	// pub fn get_instr(&self, inno: InnovNum) -> usize {
	// 	self.markers
	// 		.iter()
	// 		.find_map(|&(pos, no)| (inno == no).then_some(pos))
	// 		.unwrap()
	// }

	// pub fn get_inno(&self, pos: usize) -> InnovNum {
	// 	self.markers
	// 		.iter()
	// 		.find_map(|&(po, no)| (pos == po).then_some(no))
	// 		.unwrap()
	// }

	// /// Mark a new innovation (pushed to end)
	// pub fn mark_at(&mut self, pos: usize, inno: InnovNum) {
	// 	self.markers.push((pos, inno));
	// }

	pub fn emit(&self) -> Vec<u8> {
		let mut modu = Module::new();
		let types = {
			let mut ts = TypeSection::new();
			ts.ty().function(
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
			let mut func = Function::new([]); // TODO add locals (map from self.locals)
			for g in &self.genes {
				// add instructions from genome
				func.instruction(&g.instr);
			}
			func.instruction(&Instruction::End);
			cs.function(&func);
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
		let diff = self.diff(other);
		let n = cmp::max(self.len(), other.len()) as f64; // max num genes
		let (num_excess, num_disjoint) =
			diff.into_iter()
				.fold((0, 0), |(acc_e, acc_d), gd| match gd {
					GeneDiff::Match(_) => (acc_e, acc_d),
					GeneDiff::Disjoint(a, b) if a.is_empty() => (acc_e + b.len(), acc_d),
					GeneDiff::Disjoint(a, b) if b.is_empty() => (acc_e + a.len(), acc_d),
					GeneDiff::Disjoint(a, b) => (acc_e, acc_d + b.len()),
				});
		(DIST_COEFF_EXCESS * (num_excess as f64) + DIST_COEFF_DISJOINT * (num_disjoint as f64)) / n
	}

	fn fitness(&self) -> f64 {
		self.fitness
	}

	fn reproduce(&self, other: &Self, mut ctx: &mut Context) -> Self {
		let par_a = self;
		let par_b = other;
		log::debug!("Crossing over:\n\ta = {par_a:?}\n\tb = {par_b:?}");
		let diff = par_a.diff(par_b);

		let mut child: Vec<WasmGene> = Vec::with_capacity(par_a.len());
		for d in diff {
			match d {
				GeneDiff::Match(mat) => {
					child.extend(par_a[mat].iter().cloned());
				}
				GeneDiff::Disjoint(a, b) => {
					let choice = if ctx.rng().gen_bool(0.5) {
						&par_a[a]
					} else {
						&par_b[b]
					};
					child.extend(choice.iter().cloned());
				}
			}
		}

		WasmGenome {
			genes: child,
			fitness: 0.0,
			generation: self.generation + 1,
			params: par_a.params.clone(),
			result: par_a.result.clone(),
			locals: par_a.locals.clone(),
		}
	}
}

impl Deref for WasmGenome {
	type Target = [WasmGene<'static>];
	fn deref(&self) -> &Self::Target {
		&self.genes
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

/// Data for logging a genome
#[derive(Serialize, Debug, Clone)]
pub struct WasmGenomeRecord {
	pub generation: usize,
	pub id: usize,
	pub fitness: f64,
	pub length: usize,
}

impl WasmGenomeRecord {
	pub fn new(id: usize, genome: &WasmGenome) -> Self {
		Self {
			generation: genome.generation,
			id,
			fitness: genome.fitness,
			length: genome.genes.len(),
		}
	}
}
