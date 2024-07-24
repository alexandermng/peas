//! WebAssembly Genome

use std::{
	borrow::Cow,
	cell::{Ref, RefCell},
	fmt::{Debug, Display},
	ops::{Deref, DerefMut},
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

/// A gene of the Wasm Genome, holding its type and historical marker.
#[derive(Clone, Debug)]
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

/// The genome of a Wasm agent/individual, with additional genetic data. Can generate a Wasm Agent: bytecode whose
/// phenotype is the solution for a particular problem.
#[derive(Clone, Debug, Default)]
pub struct WasmGenome {
	pub genes: Vec<WasmGene<'static>>,
	pub fitness: f64,
	pub params: Vec<ValType>,
	pub result: Vec<ValType>,

	pub(crate) locals: Vec<StackValType>, // local variable types. includes params
}

impl WasmGenome {
	/// Create a new WasmGenome with the given signature for the main function.
	pub fn new(params: &[ValType], result: &[ValType]) -> Self {
		// let mut config = walrus::ModuleConfig::default();
		// config
		// 	.generate_name_section(false)
		// 	.generate_producers_section(false)
		// 	// .preserve_code_transform(true) // huh?
		// 	;
		// let mut module = walrus::Module::with_config(config);
		// let mut func = FunctionBuilder::new(&mut module.types, params, result); // empty body
		// let args: Vec<_> = params.iter().map(|&p| module.locals.add(p)).collect();
		// let func = func.finish(args, &mut module.funcs);
		// module.exports.add("main", func);
		// WasmGenome {
		// 	module: RefCell::new(module),
		// 	func,
		// 	markers: vec![],
		// 	fitness: 0.0,
		// }

		//Not sure what default values to put here; should be a blank initialization i think
		WasmGenome {
			genes: Vec::new(),
			fitness: 0.0,
			params: Vec::new(),
			result: Vec::new(),

			locals: Vec::new(), // TODO clone params
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

	// pub(crate) fn func<'a>(&'a self) -> Ref<walrus::LocalFunction> {
	// 	Ref::map(self.module.borrow(), |m| {
	// 		m.funcs.get(self.func).kind.unwrap_local()
	// 	})
	// }

	// pub fn func_mut(&mut self) -> &mut walrus::LocalFunction {
	// 	self.module
	// 		.get_mut()
	// 		.funcs
	// 		.get_mut(self.func)
	// 		.kind
	// 		.unwrap_local_mut()
	// }

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
			ts.function(self.params.clone(), self.result.clone()); // TODO fix hardcode //DONE
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
		modu.finish()
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

impl Display for WasmGenome {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Genome[")?;
		for b in self.emit() {
			// TODO OPT: shed chaff, just core main code
			write!(f, "{:X}", b)?;
		}
		f.write_str("]")
	}
}
