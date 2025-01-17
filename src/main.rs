use std::fs;

use clap::Parser;
use eyre::eyre;
use peas::{
	genetic::{Configurable, GenAlg},
	params::GenAlgParams,
	prelude::*,
	problems::Sum3,
	selection::TournamentSelection,
	wasm::{
		mutations::{AddOperation, ChangeRoot, WasmMutationSet},
		WasmGenAlg, WasmGenAlgConfig,
	},
};
use rand::{thread_rng, Rng};
use wasm_encoder::Instruction;

/// Input command-line arguments
#[derive(clap::Parser, Debug)]
pub struct GenAlgParamsCLI {
	/// Config filename
	#[arg(short = 'F', long = "config")]
	pub config: Option<String>,

	/// Problem to run. Incompatible with `--config`.
	#[arg(short = 'p', long = "problem")]
	pub problem: Option<String>,

	/// Seed for algorithm run
	#[arg(short = 's', long = "seed")]
	pub seed: Option<u64>,

	/// Population size
	#[arg(long = "popsize")]
	pub pop_size: Option<usize>,

	/// Number of Generations
	#[arg(short = 'n', long = "numgens")]
	pub num_generations: Option<usize>,
}

fn main() -> eyre::Result<()> {
	pretty_env_logger::init();

	let args = GenAlgParamsCLI::parse();
	if let Some(filename) = args.config {
		// replay a run
		let contents = fs::read_to_string(filename)?;
		let config: WasmGenAlgConfig<WasmMutationSet, TournamentSelection> =
			toml::from_str(&contents)?;
		let mut ga = WasmGenAlg::<Sum3>::from_config(config);
		ga.run();
		return Ok(());
	}

	let problem = args.problem.unwrap_or_else(|| String::from("sum3"));
	let (problem, init) = match &*problem {
		"sum3" => (Sum3::new(1000, 0.1, 0.2), {
			let params = &[StackValType::I32, StackValType::I32, StackValType::I32];
			let result = &[StackValType::I32];
			let mut wg = WasmGenome::new(0, params, result);
			wg.genes
				.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
			wg
		}),
		_ => return Err(eyre!("Unknown problem")),
	};
	let muts: Vec<WasmMutationSet> = vec![
		AddOperation::from_rate(0.2).into(), // local variable
		ChangeRoot::from_rate(0.3).into(),   // consts, locals, push onto stack
	];
	let params = GenAlgParams::builder()
		.seed(args.seed.unwrap_or_else(|| thread_rng().gen()))
		.pop_size(args.pop_size.unwrap_or(100))
		.num_generations(args.num_generations.unwrap_or(100))
		.max_fitness(1.0)
		.mutators(muts)
		.mutation_rate(1.0)
		.selector(TournamentSelection::new(0.6, 3, 0.9, false)) // can do real tournament selection when selection is fixed
		.init_genome(init)
		.elitism_rate(0.05)
		.crossover_rate(0.95)
		.enable_speciation(false)
		.build();
	let results = WasmGenAlgResults::default();
	// match problem {

	// }
	// TODO fix somehow... I need to box it? ffs
	let mut ga = WasmGenAlg::new(problem, params, results);
	ga.run();

	Ok(())
}
