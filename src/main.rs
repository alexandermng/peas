use std::fs;

use clap::Parser;
use eyre::eyre;
use peas::{
	genetic::{Configurable, GenAlg},
	params::{GenAlgParams, SpeciesParams},
	prelude::*,
	problems::{Polynom, ProblemSet, Sum3, Sum4},
	selection::TournamentSelection,
	wasm::{
		mutations::{AddOperation, ChangeRoot, WasmMutationSet},
		WasmGenAlg, WasmGenAlgConfig,
	},
};
use rand::Rng;
use wasm_encoder::Instruction;

/// Input command-line arguments
#[derive(clap::Parser, Debug)]
pub struct GenAlgParamsCLI {
	/// Config filename
	#[arg(short = 'F', long = "config")]
	pub config: Option<String>,

	/// Output directory name (e.g. "trial_1234.log")
	#[arg(short, long)]
	pub outfile: Option<String>,

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
		let mut config: WasmGenAlgConfig<WasmMutationSet, TournamentSelection> =
			toml::from_str(&contents)?;
		if let Some(of) = args.outfile {
			// mem::take?
			config.output_dir = of;
		}
		// TODO pull in an init.wasm or smthn. this is jank and repeated
		config.params.init_genome = match &config.problem {
			ProblemSet::Sum3(_) => {
				let params = &[StackValType::I32, StackValType::I32, StackValType::I32];
				let result = &[StackValType::I32];
				let mut wg = WasmGenome::new(0, params, result);
				wg.genes
					.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
				wg
			}
			ProblemSet::Sum4(_) => {
				let params = &[
					StackValType::I32,
					StackValType::I32,
					StackValType::I32,
					StackValType::I32,
				];
				let result = &[StackValType::I32];
				let mut wg = WasmGenome::new(0, params, result);
				wg.genes
					.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
				wg
			}
			ProblemSet::Polynom2(_) => {
				let params = &[StackValType::I32, StackValType::I32];
				let result = &[StackValType::I32];
				let mut wg = WasmGenome::new(0, params, result);
				wg.genes
					.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
				wg
			}
		};
		let mut ga = WasmGenAlg::<Sum3>::from_config(config);
		ga.run();
		return Ok(());
	}

	let problem = args.problem.unwrap_or_else(|| String::from("sum3"));
	let (problem, init) = match &*problem {
		"sum3" => (ProblemSet::Sum3(Sum3::new(1000, 0.1, 0.2)), {
			let params = &[StackValType::I32, StackValType::I32, StackValType::I32];
			let result = &[StackValType::I32];
			let mut wg = WasmGenome::new(0, params, result);
			wg.genes
				.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
			wg
		}),
		"sum4" => (ProblemSet::Sum4(Sum4::new(1000, 0.02, 0.04, 0.08)), {
			let params = &[
				StackValType::I32,
				StackValType::I32,
				StackValType::I32,
				StackValType::I32,
			];
			let result = &[StackValType::I32];
			let mut wg = WasmGenome::new(0, params, result);
			wg.genes
				.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
			wg
		}),
		"polynom" => (ProblemSet::Polynom2(Polynom::new(1000, 0.3)), {
			let params = &[StackValType::I32, StackValType::I32];
			let result = &[StackValType::I32];
			let mut wg = WasmGenome::new(0, params, result);
			wg.genes
				.push(WasmGene::new(Instruction::I32Const(0), InnovNum(0)));
			wg
		}),
		_ => return Err(eyre!("Unknown problem")),
	};
	let muts: Vec<WasmMutationSet> = vec![
		AddOperation::from_rate(0.1).into(), // local variable
		ChangeRoot::from_rate(0.4).into(),   // consts, locals, push onto stack
	];
	let seed = args.seed.unwrap_or_else(|| rand::rng().random());
	let params = GenAlgParams::builder()
		.seed(seed)
		.pop_size(args.pop_size.unwrap_or(100))
		.num_generations(args.num_generations.unwrap_or(100))
		.max_fitness(1.0)
		.mutators(muts)
		.mutation_rate(1.0)
		.selector(TournamentSelection::new(0.6, 2, 0.9, false)) // can do real tournament selection when selection is fixed
		.speciation(SpeciesParams {
			enabled: true,
			threshold: 1.2,
			fitness_sharing: true,
		})
		.init_genome(init)
		.elitism_rate(0.04)
		.crossover_rate(0.8)
		.build();
	let mut results = DefaultWasmGenAlgResults::default();
	results.resultsfile = Some("results.json".into());
	results.datafile = Some("data.csv".into()); // TODO deduplicate much of this by just loading a default config and merging in passed args
	match problem {
		//.... hey. it works.
		ProblemSet::Sum3(p) => {
			let mut ga = WasmGenAlg::new(p, params, results);
			ga.run();
		}
		ProblemSet::Sum4(p) => {
			let mut ga = WasmGenAlg::new(p, params, results);
			ga.run();
		}
		ProblemSet::Polynom2(p) => {
			let mut ga = WasmGenAlg::new(p, params, results);
			ga.run();
		}
	}

	Ok(())
}
