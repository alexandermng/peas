use eyre::Result;
use wasmtime::{Engine, Linker, Module, Store};

fn main() -> Result<(), wasmtime::Error> {
	let engine = Engine::default();
	let module = Module::from_binary(&engine, b"\x00")?;

	let linker = Linker::new(&engine);
	let mut store = Store::new(&engine, ());
	let instance = linker.instantiate(&mut store, &module).unwrap();
	let main = instance.get_typed_func::<(i32, i32, i32), i32>(&mut store, "main")?;

	let args = (10, 20, 30);
	let res = main.call(&mut store, args)?;
	println!("{args:?} -> {res}");

	Ok(())
}
