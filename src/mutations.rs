//! Possible mutations to our program

use eyre::Result;
use rand::{
	distributions::{Bernoulli, Distribution, Uniform},
	seq::SliceRandom,
	Rng,
};
use walrus::{
	ir::{self, dfs_in_order, BinaryOp, Const, Instr, LocalGet, Value, Visitor},
	FunctionBuilder,
};

use crate::{genetic::Mutator, Context, WasmGenome};

/// Mutation by individual genes
pub trait WasmMutator {
	/// Finds potential mutations for this genome, as a list of function-local indices.
	fn find_valids(&self, indiv: &WasmGenome) -> Vec<usize>;

	/// Mutate a gene at a particular index.
	fn mutate_gene(&self, ctx: &mut Context, indiv: &mut WasmGenome, loc: usize);

	/// Get the rate of mutation for the genome. Should be between 0..=1.
	fn rate(&self, ctx: &mut Context, indiv: &WasmGenome) -> f64;
}

impl<M: WasmMutator> Mutator<WasmGenome, Context> for M {
	fn mutate(&self, ctx: &mut Context, mut indiv: WasmGenome) -> WasmGenome {
		let entry = indiv.func().entry_block();
		let valids = self.find_valids(&indiv);
		if valids.is_empty() {
			return indiv;
		}
		let rate = self.rate(ctx, &indiv) / (valids.len() as f64); // rate per-valid-gene
		let dist = Bernoulli::new(rate).unwrap();

		for loc in valids {
			if dist.sample(&mut ctx.rng) {
				self.mutate_gene(ctx, &mut indiv, loc);
			}
		}

		indiv
	}
}

pub struct NeutralAddOp {
	rate: f64,
}
impl NeutralAddOp {
	pub fn from_rate(rate: f64) -> Self {
		Self { rate }
	}
}
impl WasmMutator for NeutralAddOp {
	fn find_valids(&self, indiv: &WasmGenome) -> Vec<usize> {
		// Finds valid instructions
		struct Cataloguer {
			idx: usize,
			cata: Vec<usize>,
		};
		impl<'instr> Visitor<'instr> for Cataloguer {
			fn visit_instr(&mut self, instr: &'instr Instr, _: &'instr walrus::InstrLocId) {
				// everything valid
				self.cata.push(self.idx);
				self.idx += 1;
			}
		}
		let mut vis = Cataloguer {
			idx: 0,
			cata: vec![],
		};
		let entry = indiv.func().entry_block();
		dfs_in_order(&mut vis, &indiv.func(), entry);
		log::debug!(
			"NeutralAddOp found ({} valid / {} total) genes",
			vis.cata.len(),
			vis.idx
		);
		vis.cata
	}

	fn rate(&self, ctx: &mut Context, _: &WasmGenome) -> f64 {
		self.rate
	}

	fn mutate_gene(&self, ctx: &mut Context, indiv: &mut WasmGenome, loc: usize) {
		static ALLOWED_OPS: [(BinaryOp, Value); 6] = [
			// (Operation, Identity constant)
			(BinaryOp::I32Add, Value::I32(0)), // + 0
			(BinaryOp::I32Sub, Value::I32(0)), // - 0
			(BinaryOp::I32Mul, Value::I32(1)), // * 1
			// (BinaryOp::I32DivS, Value::I32(1)),    // ÷ 1
			// (BinaryOp::I32RemS, Value::I32(1)),    // % 1
			(BinaryOp::I32And, Value::I32(-1i32)), // & 0xffffffff
			(BinaryOp::I32Or, Value::I32(0)),      // | 0x00000000
			(BinaryOp::I32Xor, Value::I32(0)),     // ^ 0x00000000

			                                       // ...etc
		];
		let (op, ident) = *ALLOWED_OPS.choose(&mut ctx.rng).unwrap();

		log::debug!(
			"Adding Operation {op:?} at {loc} (within 0..{})",
			indiv.func().size()
		);
		// indiv.mark_at(
		// 	// wtf is this shit
		// 	loc,
		// 	ctx.innov(indiv.get_inno(loc), Instr::Const(Const { value: ident })),
		// );
		// indiv.mark_at(
		// 	loc + 1,
		// 	ctx.innov(indiv.get_inno(loc + 1), Instr::Binop(ir::Binop { op })),
		// );
		indiv
			.func_mut()
			.builder_mut()
			.func_body()
			.const_at(loc + 1, ident)
			.binop_at(loc + 2, op);
	}
}

pub struct SwapRoot {
	rate: f64,
}
impl SwapRoot {
	pub fn from_rate(rate: f64) -> Self {
		Self { rate }
	}
}
impl WasmMutator for SwapRoot {
	fn find_valids(&self, indiv: &WasmGenome) -> Vec<usize> {
		// Finds valid instructions
		struct Cataloguer {
			idx: usize,
			cata: Vec<usize>,
		};
		impl<'instr> Visitor<'instr> for Cataloguer {
			fn visit_instr(&mut self, instr: &'instr Instr, _: &'instr walrus::InstrLocId) {
				if instr.is_const() || instr.is_local_get() || instr.is_global_get() {
					// consts or gets
					self.cata.push(self.idx);
				}
				self.idx += 1;
			}
		}
		let mut vis = Cataloguer {
			idx: 0,
			cata: vec![],
		};
		let entry = indiv.func().entry_block();
		dfs_in_order(&mut vis, &indiv.func(), entry);
		log::debug!(
			"SwapRoot found ({} valid / {} total) genes",
			vis.cata.len(),
			vis.idx
		);
		vis.cata
	}

	fn rate(&self, ctx: &mut Context, _: &WasmGenome) -> f64 {
		self.rate
	}

	fn mutate_gene(&self, ctx: &mut Context, indiv: &mut WasmGenome, loc: usize) {
		let entry = indiv.func().entry_block();
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
		log::debug!(
			"Swapping {:?} into {:?}",
			indiv.func().block(entry)[loc].0,
			instr
		);

		indiv.func_mut().block_mut(entry)[loc].0 = instr;
	}
}

/***** UTILITY IMPLS *****/

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

// /// Mutation with a fixed probability rate of happening (based on the Bernoulli distribution)
// #[derive(Debug)]
// pub struct Rated<M>
// where
// 	M: Mutator<WasmGenome, Context>,
// {
// 	pub rate: f64,
// 	pub mutator: M,
// 	dist: Bernoulli, // distribution
// }

// impl<M> Rated<M>
// where
// 	M: Mutator<WasmGenome, Context>,
// {
// 	/// Create a new Rated Mutation with the given rate. `1.0` will always happen, `0.0` will never happen.
// 	/// Panics if `rate > 1.0 || rate < 0.0`.
// 	pub fn new(mutator: M, rate: f64) -> Self {
// 		let dist = Bernoulli::new(rate).unwrap();
// 		Self {
// 			rate,
// 			mutator,
// 			dist,
// 		}
// 	}
// }

// impl<M> Mutator<WasmGenome, Context> for Rated<M>
// where
// 	M: Mutator<WasmGenome, Context>,
// {
// 	fn mutate(&self, ctx: &mut Context, indiv: WasmGenome) -> WasmGenome {
// 		if self.dist.sample(&mut ctx.rng) {
// 			self.mutator.mutate(ctx, indiv)
// 		} else {
// 			indiv
// 		}
// 	}
// }

// TODO dynamic rate
