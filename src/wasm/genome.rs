//! WebAssembly Genome

use std::{
	cell::{Ref, RefCell},
	fmt::{Debug, Display},
	ops::{Deref, DerefMut},
};

use eyre::{eyre, Result};
use walrus::{CustomSection, FunctionBuilder, ValType};

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

/// The genome of a Wasm agent/individual, with additional genetic data. Can generate a Wasm Agent: bytecode whose
/// phenotype is the solution for a particular problem.
pub struct WasmGenome {
	pub(crate) module: RefCell<walrus::Module>, // in progress module
	pub(crate) func: walrus::FunctionId,        // mutatable main (LocalFunction) in module
	pub(crate) markers: Vec<(usize, InnovNum)>, // markers for main, mapping their local pos to innovation number
	// TODO WE STILL HAVE A PROBLEM. SHIT MOVES AROUND.
	pub fitness: f64,
}

impl WasmGenome {
	/// Create a new WasmGenome with the given signature for the main function.
	pub fn new(params: &[ValType], result: &[ValType]) -> Self {
		let mut config = walrus::ModuleConfig::default();
		config
			.generate_name_section(false)
			.generate_producers_section(false)
			// .preserve_code_transform(true) // huh?
			;
		let mut module = walrus::Module::with_config(config);
		let mut func = FunctionBuilder::new(&mut module.types, params, result); // empty body
		let args: Vec<_> = params.iter().map(|&p| module.locals.add(p)).collect();
		let func = func.finish(args, &mut module.funcs);
		module.exports.add("main", func);
		WasmGenome {
			module: RefCell::new(module),
			func,
			markers: vec![],
			fitness: 0.0,
		}
	}

	pub fn from_binary(binary: &[u8]) -> Result<Self> {
		let mut module = walrus::Module::from_buffer(binary)
			.map_err(|e| eyre!("could not create module: {e}"))?;
		let func = module
			.exports
			.get_func("main")
			.map_err(|e| eyre!("cannot find main function: {e}"))?;
		Ok(WasmGenome {
			module: RefCell::new(module),
			func,
			markers: vec![],
			fitness: 0.0,
		})
	}

	pub(crate) fn func<'a>(&'a self) -> Ref<walrus::LocalFunction> {
		Ref::map(self.module.borrow(), |m| {
			m.funcs.get(self.func).kind.unwrap_local()
		})
	}

	pub fn func_mut(&mut self) -> &mut walrus::LocalFunction {
		self.module
			.get_mut()
			.funcs
			.get_mut(self.func)
			.kind
			.unwrap_local_mut()
	}

	/// Retrieve the local position by global innovation number.
	pub fn get_instr(&self, inno: InnovNum) -> usize {
		self.markers
			.iter()
			.find_map(|&(pos, no)| (inno == no).then_some(pos))
			.unwrap()
	}

	pub fn get_inno(&self, pos: usize) -> InnovNum {
		self.markers
			.iter()
			.find_map(|&(po, no)| (pos == po).then_some(no))
			.unwrap()
	}

	/// Mark a new innovation (pushed to end)
	pub fn mark_at(&mut self, pos: usize, inno: InnovNum) {
		self.markers.push((pos, inno));
	}

	pub fn emit(&self) -> Vec<u8> {
		let mut module = self.module.borrow_mut();
		#[derive(Debug)]
		struct Fitness(f64);
		impl CustomSection for Fitness {
			fn name(&self) -> &str {
				"fitness"
			}
			fn data(&self, _ids_to_indices: &walrus::IdsToIndices) -> std::borrow::Cow<[u8]> {
				std::borrow::Cow::Owned(self.0.to_string().into_bytes())
			}
		}
		module.customs.add(Fitness(self.fitness)); // TODO debug why cant i see
		module.producers.clear(); // it should rly skip this already tho :/
		module.emit_wasm()
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

impl Clone for WasmGenome {
	fn clone(&self) -> Self {
		let mut out = WasmGenome::from_binary(&self.emit()).unwrap(); // emitted module should be valid
		out.markers.clone_from(&self.markers);
		out.fitness.clone_from(&self.fitness);
		out
	}
}

impl Debug for WasmGenome {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("WasmGenome")
			.field("func", &self.func)
			// .field("markers", &self.markers)
			.field("fitness", &self.fitness)
			.finish()
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
