//! (Straight up ripped from the Walrus examples... just to see that I can go back and edit a function body)
//! This example constructs a Wasm module from scratch with Walrus.
//!
//! The module we are building implements and exports the `factorial` function,
//! and imports a `env.log` function to observe incremental results.
//!
//! You can run the built Wasm module using Node.js (for example) like this:
//!
//! ```js
//! const fs = require("fs");
//!
//! async function main() {
//!   const bytes = fs.readFileSync("target/out.wasm");
//!   const env = { log: val => console.log(`logged ${val}`), };
//!   const { instance } = await WebAssembly.instantiate(
//!     bytes,
//!     {
//!       env: {
//!         log(val) {
//!           console.log(`log saw ${val}`);
//!         }
//!       }
//!     }
//!   );
//!   const result = instance.exports.factorial(5);
//!   console.log(`factorial(5) = ${result}`);
//! }
//!
//! main();
//! ```

use walrus::{ir::*, CustomSection};
use walrus::{FunctionBuilder, Module, ModuleConfig, ValType};

fn main() -> walrus::Result<()> {
	// Construct a new Walrus module.
	let mut config = ModuleConfig::new();
	config.generate_producers_section(true);
	let mut module = Module::with_config(config);

	// Import the `log` function.
	let log_type = module.types.add(&[ValType::I32], &[]);
	let (log, _) = module.add_import_func("env", "log", log_type);

	// Building this factorial implementation:
	// https://github.com/WebAssembly/testsuite/blob/7816043/fac.wast#L46-L66
	let mut factorial = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::I32]);

	// Create our parameter and our two locals.
	let n = module.locals.add(ValType::I32);
	let i = module.locals.add(ValType::I32);
	let res = module.locals.add(ValType::I32);

	factorial
		// Enter the function's body.
		.func_body()
		// (local.set $i (local.get $n))
		.local_get(n)
		.local_set(i)
		// (local.set $res (i32.const 1))
		.i32_const(1)
		.local_set(res)
		.block(None, |done| {
			let done_id = done.id();
			done.loop_(None, |loop_| {
				let loop_id = loop_.id();
				loop_
					// (call $log (local.get $res))
					.local_get(res)
					.call(log)
					// (i32.eq (local.get $i) (i32.const 0))
					.local_get(i)
					.i32_const(0)
					.binop(BinaryOp::I32Eq)
					.if_else(
						None,
						|then| {
							// (then (br $done))
							then.br(done_id);
						},
						|else_| {
							else_
								// (local.set $res (i32.mul (local.get $i) (local.get $res)))
								.local_get(i)
								.local_get(res)
								.binop(BinaryOp::I32Mul)
								.local_set(res)
								// (local.set $i (i32.sub (local.get $i) (i32.const 1))))
								.local_get(i)
								.i32_const(1)
								.binop(BinaryOp::I32Sub)
								.local_set(i);
						},
					)
					.br(loop_id);
			});
		})
		.local_get(res);

	let mut factorial = factorial.local_func(vec![n]);

	{
		// Whoops we forgot to add things... here we go!
		factorial
			.builder_mut()
			.func_body()
			.unop(UnaryOp::I32Eqz)
			.unop(UnaryOp::I32Eqz)
			.unop(UnaryOp::I32Eqz);
	}

	struct Printer;
	impl<'i> Visitor<'i> for Printer {
		fn visit_instr(&mut self, instr: &'i Instr, _instr_loc: &'i InstrLocId) {
			println!("Came across {instr:?}");
		}
	}
	impl VisitorMut for Printer {
		fn visit_instr_mut(&mut self, instr: &mut Instr, loc: &mut InstrLocId) {
			println!("Mutated {instr:?}\tat {loc:?}");
		}
	}
	println!("PRINTING IN ORDER");
	dfs_in_order(&mut Printer {}, &factorial, factorial.entry_block());
	println!("PRINTING PRE ORDER");
	let start = factorial.entry_block();
	dfs_pre_order_mut(&mut Printer {}, &mut factorial, start);

	let factorial_id = module.funcs.add_local(factorial);

	// Export the `factorial` function.
	module.exports.add("factorial", factorial_id);

	{
		let fact = module.funcs.get_mut(factorial_id).kind.unwrap_local_mut();
		fact.builder_mut().func_body().local_get(n).drop();
	}

	// Add custom!
	#[derive(Debug)]
	struct Tag(&'static str);
	impl CustomSection for Tag {
		fn name(&self) -> &str {
			"tag"
		}
		fn data(&self, ids_to_indices: &walrus::IdsToIndices) -> std::borrow::Cow<[u8]> {
			std::borrow::Cow::Borrowed(self.0.as_bytes())
		}
	}
	module.customs.add(Tag("wally"));

	// Emit the `.wasm` binary to the `target/out.wasm` file.
	module.emit_wasm_file("target/factorial.wasm")?;
	module.emit_wasm_file("target/factorial2.wasm")?;

	module.write_graphviz_dot("target/factorial.dot")?;

	Ok(())
}
