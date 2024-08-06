//! WebAssembly Genome

use std::{
	borrow::Cow,
	cell::{Ref, RefCell},
	fmt::{Debug, Display},
	ops::{Deref, DerefMut, Range, RangeBounds},
};

use eyre::{eyre, Result};
use wasm_encoder::{
	CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
	PrimitiveValType, TypeSection, ValType,
};

use crate::genetic::Genome;

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

/// The genome of a Wasm agent/individual, with additional genetic data. Can generate a Wasm Agent: bytecode whose
/// phenotype is the solution for a particular problem.
#[derive(Clone, Debug, Default)]
pub struct WasmGenome {
	pub genes: Vec<WasmGene<'static>>,
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
			genes: Vec::new(),
			fitness: 0.0,
			params,
			result,

			locals,
		}
	}

	pub fn from_binary(binary: &[u8]) -> Result<Self> {
		// let mut module = walrus::Module::from_buffer(binary)
		// 	.map_err(|e| eyre!("could not create module: {e}"))?;
		// let func = module
		// 	.exports
		// 	.get_func("main")
		// 	.map_err(|e| eyre!("cannot find main function: {e}"))?;
		// Ok(WasmGenome {
		// 	module: RefCell::new(module),
		// 	func,
		// 	markers: vec![],
		// 	fitness: 0.0,
		// })
		todo!()
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
		assert!(
			wasmparser::validate(&out).is_ok(),
			"generated invalid module"
		);
		out
	}

	/// Show the gene intersection ranges of this genome and another. Returns a tuple of (matches, disjoint),
	/// where match ranges are of this genome's matching genes and disjoint ranges are a tuple of corresponding
	/// (self, other) genes. Genes will be [mat, dis, mat, dis... mat?], starting with 0..0 if it starts disjoint.
	pub(crate) fn intersect(
		&self,
		other: &Self,
	) -> (Vec<Range<usize>>, Vec<(Range<usize>, Range<usize>)>) {
		let len_a = self.len(); // par_a = self
		let len_b = other.len(); // par_b = other
		let mut matches = Vec::new();
		let mut disjoint = Vec::new();

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
					disjoint.push((last_a..len_a, last_b..len_b)); // final disjoint
					break;
				}
				(false, true, _) => {
					// out of bounds while matching
					matches.push(last_a..cur_a); // final match
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
					matches.push(last_a..cur_a); // push matching range
					was_matching = false; // start disjoint range
					(last_a, last_b) = (cur_a, cur_b);
					// pass to (_, false, false)
				}
				(true, false, true) => {
					// was disjoint, but now matching
					disjoint.push((last_a..cur_a, last_b..cur_b)); // push disjoint range
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
		log::debug!("Found ranges for (0..{len_a}, 0..{len_b}): matching {matches:?} and disjoint {disjoint:?}");
		(matches, disjoint)
	}
}

impl Genome for WasmGenome {
	fn dist(&self, other: &Self) -> f64 {
		todo!() // use markers
	}

	fn fitness(&self) -> f64 {
		self.fitness
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
