use eyre::Result;
use wasm_encoder::{
	CodeSection, ExportKind, ExportSection, FunctionSection, Instruction, PrimitiveValType,
	TypeSection,
};
use wasm_encoder::{ComponentBuilder, Function, Module};
use wasmtime::component::{Component, Linker};
use wasmtime::{Engine, Store};

fn generate() -> Vec<u8> {
	let mut comp = ComponentBuilder::default();
	let mut modu = Module::new();
	let types = {
		let mut ts = TypeSection::new();
		ts.function(&[PrimitiveValType::I32], &[PrimitiveValType::Bool]);
		ts
	};
	let funcs = {
		let mut fs = FunctionSection::new();
		fs.function(0);
		fs
	};
	let expos = {
		let mut es = ExportSection::new();
		es.export("main", ExportKind::Func, 0);
		es
	};
	let codes = {
		let mut cs = CodeSection::new();
		let mut func = Function::new([]);
		func.instruction(&Instruction::LocalGet(0))
			.instruction(&Instruction::I32Eqz)
			.instruction(&Instruction::End);
		cs.function(&func);
		cs
	};
	modu.section(&types)
		.section(&funcs)
		.section(&expos)
		.section(&codes);
	comp.core_module(&modu);
	comp.finish()
}

fn main() -> Result<(), wasmtime::Error> {
	let engine = Engine::default();
	let bytes = generate();
	let comp = Component::from_binary(&engine, &bytes)?;

	let linker = Linker::new(&engine);
	let mut store = Store::new(&engine, ());
	let instance = linker.instantiate(&mut store, &comp)?;
	let main = instance.get_typed_func::<i32, bool>(&mut store, "main")?;

	let args = (10, 20, 30);
	let res = main.call(&mut store, args)?;
	println!("{args:?} -> {res}");

	Ok(())
}
