//! Possible mutations to our program

use eyre::Result;
use rand::{
	distributions::{Bernoulli, Distribution, Uniform},
	seq::SliceRandom,
	Rng,
};
use walrus::{
	ir::{BinaryOp, Const, Instr, LocalGet, Value},
	FunctionBuilder,
};

use crate::{genetic::Mutator, Context, WasmGenome};

pub struct NeutralAddOp;
impl Mutator<WasmGenome, Context> for NeutralAddOp {
	fn mutate(&self, ctx: &mut Context, mut indiv: WasmGenome) -> WasmGenome {
		let unif = Uniform::new(1, indiv.func().size() as usize + 1);
		let loc = unif.sample(&mut ctx.rng); // chosen mutation location

		static ALLOWED_OPS: [(BinaryOp, Value); 5] = [
			// (Operation, Identity constant)
			(BinaryOp::I32Add, Value::I32(0)),  // + 0
			(BinaryOp::I32Sub, Value::I32(0)),  // - 0
			(BinaryOp::I32Mul, Value::I32(1)),  // * 1
			(BinaryOp::I32DivS, Value::I32(1)), // ÷ 1
			(BinaryOp::I32RemS, Value::I32(1)), // % 1

			                                    // ...etc
		];
		let (op, ident) = ALLOWED_OPS.choose(&mut ctx.rng).unwrap();

		log::info!(
			"Adding Operation {op:?} at {loc} (within 0..{})",
			indiv.func().size()
		);
		indiv
			.func()
			.builder_mut()
			.func_body()
			.const_at(loc, Value::I32(0))
			.binop_at(loc + 1, BinaryOp::I32Add);
		// TODO add dangling instr seq and then append to correct location
		indiv
	}
}

pub struct SwapRoot;
impl Mutator<WasmGenome, Context> for SwapRoot {
	fn mutate(&self, ctx: &mut Context, mut indiv: WasmGenome) -> WasmGenome {
		let entry = indiv.func().entry_block();
		let potentials: Vec<_> = indiv
			.func()
			.block(entry)
			.iter()
			.enumerate()
			.filter(|(_, (i, _))| i.is_const() || i.is_local_get())
			.map(|(pos, (i, _))| (pos, i))
			.collect();
		let chosen = potentials.choose(&mut ctx.rng).unwrap();
		let loc = chosen.0;

		let instr = match ctx.rng.gen_range(0f64..=1f64) {
			0.0..=0.6 => {
				let local = *indiv.func().args.choose(&mut ctx.rng).unwrap();
				Instr::LocalGet(LocalGet { local })
			}
			0.6..=1.0 => {
				let value = Value::I32(ctx.rng.gen());
				Instr::Const(Const { value })
			}
			_ => unreachable!("we don't generate outside handled range"),
		};
		indiv.func().block_mut(entry)[loc].0 = instr;
		indiv
	}
}

/***** UTILITY IMPLS *****/

// TODO mutation sequence; mutation or; (mutation, rate)

/// Blanket Impl for Closures
impl<T> Mutator<WasmGenome, Context> for T
where
	T: Fn(&mut Context, &mut FunctionBuilder),
{
	fn mutate(&self, ctx: &mut Context, mut indiv: WasmGenome) -> WasmGenome {
		self(ctx, indiv.func().builder_mut());
		indiv
	}
}

/// Sequences of mutations
pub struct SequenceMutator<'a> {
	seq: &'a [&'a dyn Mutator<WasmGenome, Context>],
}

impl<'a> SequenceMutator<'a> {
	pub fn new(seq: &'a [&'a dyn Mutator<WasmGenome, Context>]) -> Self {
		Self { seq }
	}
}

impl<'a> Mutator<WasmGenome, Context> for SequenceMutator<'a> {
	fn mutate(&self, ctx: &mut Context, mut indiv: WasmGenome) -> WasmGenome {
		for m in self.seq {
			indiv = m.mutate(ctx, indiv);
		}
		indiv
	}
}

impl<'a> From<&'a [&'a dyn Mutator<WasmGenome, Context>]> for SequenceMutator<'a> {
	fn from(value: &'a [&'a dyn Mutator<WasmGenome, Context>]) -> Self {
		Self::new(value)
	}
}

// TODO impl Debug for SequenceMutator

/// Mutation with a fixed probability rate of happening (based on the Bernoulli distribution)
#[derive(Debug)]
pub struct Rated<M>
where
	M: Mutator<WasmGenome, Context>,
{
	pub rate: f64,
	pub mutator: M,
	dist: Bernoulli, // distribution
}

impl<M> Rated<M>
where
	M: Mutator<WasmGenome, Context>,
{
	/// Create a new Rated Mutation with the given rate. `1.0` will always happen, `0.0` will never happen.
	/// Panics if `rate > 1.0 || rate < 0.0`.
	pub fn new(mutator: M, rate: f64) -> Self {
		let dist = Bernoulli::new(rate).unwrap();
		Self {
			rate,
			mutator,
			dist,
		}
	}
}

impl<M> Mutator<WasmGenome, Context> for Rated<M>
where
	M: Mutator<WasmGenome, Context>,
{
	fn mutate(&self, ctx: &mut Context, indiv: WasmGenome) -> WasmGenome {
		if self.dist.sample(&mut ctx.rng) {
			self.mutator.mutate(ctx, indiv)
		} else {
			indiv
		}
	}
}
