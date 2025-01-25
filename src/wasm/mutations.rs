//! Possible mutations to our program

use std::hash::Hash;
use std::mem;

use eyre::Result;
use rand::{
	distributions::{Bernoulli, Distribution, Uniform},
	seq::SliceRandom,
	Rng,
};
use serde::{Deserialize, Serialize};
use wasm_encoder::{Encode, Instruction};
use wasmparser::names;

use crate::wasm::{Context, InnovNum, WasmGene, WasmGenome};
use crate::{genetic::Mutator, wasm::StackValType};

/// Selection of available mutations
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum WasmMutationSet {
	AddBinaryOp(AddOperation),
	ChangeRoot(ChangeRoot),
	// Sequence(Vec<WasmMutation>),
}

impl Mutator<WasmGenome, Context> for WasmMutationSet {
	fn mutate(&self, ctx: &mut Context, indiv: WasmGenome) -> WasmGenome {
		match self {
			WasmMutationSet::AddBinaryOp(add_operation) => add_operation.mutate(ctx, indiv),
			WasmMutationSet::ChangeRoot(change_root) => change_root.mutate(ctx, indiv),
		}
	}
}

impl From<AddOperation> for WasmMutationSet {
	fn from(value: AddOperation) -> Self {
		WasmMutationSet::AddBinaryOp(value)
	}
}

impl From<ChangeRoot> for WasmMutationSet {
	fn from(value: ChangeRoot) -> Self {
		WasmMutationSet::ChangeRoot(value)
	}
}

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
		let valids = self.find_valids(&indiv);
		let rate = self.rate(ctx, &indiv);
		if valids.is_empty() || !ctx.rng.gen_bool(rate) {
			return indiv;
		}
		let chosen = *valids.choose(&mut ctx.rng).unwrap();
		self.mutate_gene(ctx, &mut indiv, chosen);

		indiv
	}
}

/// Logs where a mutation happened (previous innovnum and new instruction)
#[derive(Debug, Clone)]
pub enum MutationLog {
	AddOp(InnovNum, Instruction<'static>), // AddOp added after gene @ InnovNum, with Op instruction
	ChangeRoot(InnovNum, Instruction<'static>), // ChangeRoot from gene @ InnovNum, with new instruction
	Unique(&'static str),                       // unique mutations
}

impl Hash for MutationLog {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		mem::discriminant(self).hash(state);
		let (num, instr) = match self {
			MutationLog::AddOp(n, i) => (n, i),
			MutationLog::ChangeRoot(n, i) => (n, i),
			MutationLog::Unique(s) => {
				s.hash(state);
				return;
			}
		};
		num.hash(state);
		mem::discriminant(instr).hash(state);
		match instr {
			Instruction::LocalGet(u) => u.hash(state),
			Instruction::I32Const(u) => u.hash(state),
			_ => {}
		};
	}
}

impl PartialEq for MutationLog {
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(
				Self::AddOp(lnum, Instruction::LocalGet(lu)),
				Self::AddOp(rnum, Instruction::LocalGet(ru)),
			) => lnum == rnum && lu == ru,
			(Self::AddOp(lnum, _), Self::AddOp(rnum, _)) => lnum == rnum,
			(
				Self::ChangeRoot(lnum, Instruction::LocalGet(lu)),
				Self::ChangeRoot(rnum, Instruction::LocalGet(ru)),
			) => lnum == rnum && lu == ru,
			(Self::ChangeRoot(lnum, _), Self::AddOp(rnum, _)) => lnum == rnum,
			(Self::Unique(ls), Self::Unique(rs)) => ls == rs,
			_ => false,
		}
	}
}
impl Eq for MutationLog {}

/// Adds an Operation after a random gene. Neutral.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddOperation {
	rate: f64,
}
impl AddOperation {
	pub fn from_rate(rate: f64) -> Self {
		Self { rate }
	}
}
impl WasmMutator for AddOperation {
	fn find_valids(&self, indiv: &WasmGenome) -> Vec<usize> {
		let len = indiv.len();
		log::debug!("NeutralAddOp found ({len} valid / {len} total) genes");
		(0..len).collect()
	}

	fn rate(&self, ctx: &mut Context, _: &WasmGenome) -> f64 {
		self.rate
	}

	fn mutate_gene(&self, ctx: &mut Context, indiv: &mut WasmGenome, loc: usize) {
		static I32_OPS: [(Instruction, i32); 6] = [
			(Instruction::I32Add, 0),
			(Instruction::I32Sub, 0),
			(Instruction::I32Mul, 1),
			// (Instruction::I32DivS, 1),
			// (Instruction::I32RemS, 1),
			(Instruction::I32And, 0),
			(Instruction::I32Or, 0),
			(Instruction::I32Xor, 0),
		];
		let (ty, op, ident) = match indiv[loc].ty() {
			(
				_,
				ty @ &[StackValType::U8 | StackValType::I8 | StackValType::U32 | StackValType::I32],
			) => {
				assert!(ty.len() == 1); // TODO ^ make match on 1 exact
				let (op, i) = I32_OPS.choose(&mut ctx.rng).unwrap();
				(ty[0], op.clone(), Instruction::I32Const(*i))
			} // anything that pushes an I32
			_ => unimplemented!("unrecognized gene type"),
		};

		let mlog_1 = MutationLog::AddOp(indiv[loc].marker, ident.clone());
		let mlog_2 = MutationLog::AddOp(indiv[loc].marker, op.clone());
		log::debug!("Mutated {mlog_2:?}");
		let ident = WasmGene {
			instr: ident,
			marker: ctx.innov(mlog_1),
			popty: vec![].into(),
			pushty: vec![ty].into(),
		};
		let op = WasmGene {
			instr: op,
			marker: ctx.innov(mlog_2),
			popty: vec![ty, ty].into(),
			pushty: vec![ty].into(),
		};
		indiv.insert(loc, [ident, op]);
	}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChangeRoot {
	rate: f64,
}
impl ChangeRoot {
	pub fn from_rate(rate: f64) -> Self {
		Self { rate }
	}
}
impl WasmMutator for ChangeRoot {
	fn find_valids(&self, indiv: &WasmGenome) -> Vec<usize> {
		let out: Vec<usize> = indiv
			.genes
			.iter()
			.enumerate()
			.filter_map(|(i, g)| {
				use Instruction::*;
				match g.instr {
					LocalGet(_) | I32Const(_) | F32Const(_) | I64Const(_) | F64Const(_) => Some(i), // equivalently checks popty is [] and pushty is [_; 1]
					_ => None,
				}
			})
			.collect();
		log::debug!(
			"SwapRoot found ({} valid / {} total) genes",
			out.len(),
			indiv.len()
		);
		out
	}

	fn rate(&self, ctx: &mut Context, _: &WasmGenome) -> f64 {
		self.rate
	}

	fn mutate_gene(&self, ctx: &mut Context, indiv: &mut WasmGenome, loc: usize) {
		let pushty = indiv[loc].ty().1[0];
		let i32consts: Vec<Instruction> = vec![
			// TODO consider other consts?
			Instruction::I32Const(0),
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Const(-1),
		]; // for StackValType::I32
		let i32vars: Vec<Instruction> = indiv
			.locals
			.iter()
			.enumerate()
			.filter_map(|(i, &t)| {
				if t == pushty {
					Some(Instruction::LocalGet(i as u32))
				} else {
					None
				}
			})
			.collect(); // for StackValType::I32

		let root = match pushty {
			StackValType::I32 => {
				let do_var = !i32vars.is_empty() && ctx.rng.gen_bool(0.6);
				if do_var { i32vars } else { i32consts }
					.choose(&mut ctx.rng)
					.unwrap()
					.clone()
			}
			_ => unimplemented!("unexpected stack value type"),
		};
		let mlog = MutationLog::ChangeRoot(indiv[loc].marker, root.clone());
		log::debug!("Mutated {mlog:?}");
		let root = WasmGene {
			instr: root,
			marker: ctx.innov(mlog),
			popty: vec![].into(),
			pushty: vec![pushty].into(),
		};
		indiv.replace(loc..=loc, [root]);
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
