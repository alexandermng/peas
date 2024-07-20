use std::{fs::File, io::Write};

use eyre::{eyre, Result};
use peas::WasmGenome;
use walrus::{ir, ValType};
use wasmtime::{Engine, Module};

#[test]
fn empty_module() -> Result<()> {
	let g = WasmGenome::new(&[ValType::I32], &[]);
	let bytes = g.emit();
	let mut file = File::create("test_empty.wasm")?;
	file.write_all(&bytes)?;

	let engine = Engine::default();
	let module = Module::new(&engine, bytes).map_err(|e| eyre!(Box::new(e)))?; // validates

	assert!(module.exports().len() == 1, "module should have one export");
	let exp = module.exports().next().unwrap();
	assert!(exp.name() == "main", "module name should be 'main'");

	Ok(())
}

#[test]
fn basic_module() -> Result<()> {
	let mut g = WasmGenome::new(&[ValType::I32], &[ValType::I32]);
	let arg = g.func_mut().args[0];
	g.func_mut()
		.builder_mut()
		.func_body()
		.local_get(arg)
		.const_(ir::Value::I32(42))
		.binop(ir::BinaryOp::I32Add);
	let bytes = g.emit();
	let mut file = File::create("test_basic.wasm")?;
	file.write_all(&bytes)?;

	let engine = Engine::default();
	let module = Module::new(&engine, bytes).map_err(|e| eyre!(Box::new(e)))?; // validates

	assert!(module.exports().len() == 1, "module should have one export");
	let exp = module.exports().next().unwrap();
	assert!(exp.name() == "main", "module name should be 'main'");

	Ok(())
}
