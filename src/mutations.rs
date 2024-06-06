//! Possible mutations to our program

use eyre::Result;
use rand::{
	distributions::{Distribution, Uniform},
	Rng,
};
use walrus::{
	ir::{BinaryOp, Value},
	FunctionBuilder,
};

use crate::{genetic::Mutator, Context, WasmGenome};

/// Blanket Impl for Closures
impl<T> Mutator<WasmGenome, Context<'_>> for T
where
	T: Fn(&mut Context<'_>, &mut FunctionBuilder),
{
	fn mutate(&self, ctx: &mut Context<'_>, mut indiv: WasmGenome) -> WasmGenome {
		self(ctx, indiv.func().builder_mut());
		indiv
	}
}

// Blanket Impl for Vecs of Mutators
impl<'a, T> Mutator<WasmGenome, Context<'a>> for Vec<T>
where
	T: Mutator<WasmGenome, Context<'a>>,
{
	fn mutate(&self, ctx: &mut Context<'a>, mut indiv: WasmGenome) -> WasmGenome {
		for m in self {
			indiv = m.mutate(ctx, indiv);
		}
		indiv
	}
}

pub struct NeutralAddInstr;
impl Mutator<WasmGenome, Context<'_>> for NeutralAddInstr {
	fn mutate(&self, ctx: &mut Context<'_>, mut indiv: WasmGenome) -> WasmGenome {
		let instrs = &indiv.func().instruction_mapping;
		let unif = Uniform::new(0, instrs.len());
		let chosen = instrs[unif.sample(ctx.rng)]; // chosen mutation location
		let loc = chosen.0;
		indiv
			.func()
			.builder_mut()
			.func_body()
			.const_at(loc, Value::I32(0))
			.binop(BinaryOp::I32Add);
		// TODO add dangling instr seq and then append to correct location
		indiv
	}
}
// impl Mutation for NeutralAddInstr {
// 	fn mutate(&self, genes: &[u8], rng: &mut impl Rng) -> Result<Vec<u8>> {
// 		// TODO move to algorithm mutate()
// 		let reader = BinaryReader::new(genes, 0, WasmFeatures::all());
// 		let reader = CodeSectionReader::new(reader)?;

// 		// For now, we only have one function
// 		assert!(reader.count() == 1, "we only support one function for now");

// 		let fnreaders: Vec<FunctionBody> = reader
// 			.into_iter()
// 			.collect::<Result<Vec<_>, BinaryReaderError>>()?;

// 		let funny = &fnreaders[0];
// 		let locals = funny.get_locals_reader()?;
// 		let ops = funny.get_operators_reader()?;

// 		for op in ops.into_iter() {
// 			let op = op?;
// 			print!("op is {op:?}")
// 		}

// 		todo!()
// 	}
// }

// TODO mutation sequence; mutation or; (mutation, rate)

// impl<I> Mutation for I
// where
// 	I: IntoIterator<Item = dyn Mutation>,
// {
// 	fn mutate(&self, genes: &[u8], rng: &mut impl Rng) -> Result<Vec<u8>> {
// 		todo!()
// 	}
// }
