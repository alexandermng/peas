use std::fs;

use clap::{Parser, Subcommand};
use eyre::eyre;
use peas::{
	experiments::{
		ablation, partial_tests,
		speciation::{self, SpeciationExperiment},
		ExperimentConfig, ExperimentRunnable,
	},
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

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Cli {
	#[command(subcommand)]
	pub experiment: ExperimentCommand,

	/// Output directory name (e.g. "trial_1234.log")
	#[arg(short, long, global = true)]
	pub outfile: Option<String>,

	/// Problem to run
	#[arg(short = 'p', long = "problem", global = true)]
	pub problem: Option<String>,

	/// Number of trial runs per configuration
	#[arg(short = 'n', long = "nruns", global = true)]
	pub num_runs: Option<usize>,

	/// Population size
	#[arg(long = "popsize", global = true)]
	pub pop_size: Option<usize>,

	/// Maximum number of generations cutoff
	#[arg(short = 'g', long = "gens", global = true)]
	pub num_generations: Option<usize>,
}

#[derive(Subcommand, Debug)]
pub enum ExperimentCommand {
	/// Run a single trial
	Single {
		/// Seed for algorithm run
		#[arg(short = 's', long = "seed")]
		seed: Option<u64>,

		/// Config filename (for replay)
		#[arg(short = 'F', long = "config")]
		config: Option<String>,
	},
	/// Run a Speciation experiment testing different compatibility thresholds. Runs a linear space of thresholds across a range,
	/// unless specific thresholds are set.
	Speciation {
		/// List of speciation thresholds to test.
		#[arg(short, long, value_delimiter = ',')]
		thresholds: Option<Vec<f64>>,
		// TODO take in ranges
	},
	/// Run a Partial Test Cases experiment, testing different rates of partial test cases
	PartialTests,
	/// Run an Ablation study
	Ablation {
		/// List of configurations to include; if set, runs only these. Options are: control, no_speciation, no_crossover, no_elitism, no_partials, no_partials_no_speciation, no_partials_no_crossover.
		#[arg(short, long, value_delimiter = ',')]
		include: Option<Vec<String>>,

		/// List of configurations to exclude; if set, overrides any --include and excludes these from being run. See --include for options
		#[arg(short, long, value_delimiter = ',')]
		exclude: Option<Vec<String>>,
	},
}

impl ExperimentCommand {
	pub fn create_experiment(
		&self,
		control: ExperimentConfig,
		problem: &str, // TODO fix this, extract from config
		num_runs_per: usize,
	) -> Box<dyn ExperimentRunnable> {
		match self {
			ExperimentCommand::Speciation { thresholds } => {
				Box::new(if let Some(thresholds) = thresholds {
					speciation::gen_selection(
						"speciation_threshold",
						control,
						num_runs_per,
						thresholds.iter().copied(),
					)
				} else {
					speciation::gen_linspace(
						"speciation_threshold",
						control,
						1.5..2.5,
						10,
						num_runs_per,
					)
				})
			}
			ExperimentCommand::PartialTests => {
				let name = format!("partial_tests_{problem}");
				Box::new(partial_tests::gen_basic(&name, control, num_runs_per))
			}
			ExperimentCommand::Ablation { include, exclude } => {
				let name = format!("ablation_{problem}");
				let mut out = ablation::gen_basic(&name, control, num_runs_per);
				if let Some(incl) = include {
					out.configurations
						.retain(|c| incl.iter().any(|l| c.label == *l)); // retain only those in incl
				}
				if let Some(excl) = exclude {
					out.configurations
						.retain(|c| !excl.iter().any(|l| c.label == *l)); // retain only those not in excl
				}
				Box::new(out)
			}
			_ => panic!("create_experiment called on Single variant"),
		}
	}
}

/// Helper to construct base params from CLI args
fn base_params(
	problem: &ProblemSet,
	cli: &Cli,
	seed: Option<u64>,
) -> GenAlgParams<WasmGenome, Context, WasmMutationSet, TournamentSelection> {
	let num_tests = 100; // TODO: put in args or ProblemParams
	let init = problem.init_genome();
	let muts: Vec<WasmMutationSet> = vec![
		AddOperation::from_rate(0.1).into(),
		ChangeRoot::from_rate(0.4).into(),
	];
	let seed = seed.unwrap_or_else(|| rand::rng().random());
	GenAlgParams::builder()
		.seed(seed)
		.pop_size(cli.pop_size.unwrap_or(100))
		.num_generations(cli.num_generations.unwrap_or(100))
		.max_fitness(1.0)
		.mutators(muts)
		.mutation_rate(1.0)
		.selector(TournamentSelection::new(0.6, 2, 0.9, true))
		.speciation(SpeciesParams {
			enabled: true,
			threshold: 2.0,
			fitness_sharing: true,
		})
		.init_genome(init)
		.elitism_rate(0.05)
		.crossover_rate(0.8)
		.build()
}

fn main() -> eyre::Result<()> {
	pretty_env_logger::init();

	let cli = Cli::parse();

	let problem_str = cli.problem.clone().unwrap_or_else(|| String::from("sum3"));
	let num_tests = 100; // TODO: put in args or ProblemParams
	let problem = match &*problem_str {
		"sum3" => ProblemSet::Sum3(Sum3::new(num_tests, 0.1, 0.2)),
		"sum4" => ProblemSet::Sum4(Sum4::new(num_tests, 0.02, 0.04, 0.08)),
		"poly2" => ProblemSet::Polynom2(Polynom::new(num_tests, 0.3)),
		_ => return Err(eyre!("Unknown problem")),
	};

	match &cli.experiment {
		ExperimentCommand::Single { seed, config } => {
			if let Some(filename) = config {
				// replay a run
				let contents = fs::read_to_string(filename)?;
				let mut config: WasmGenAlgConfig<WasmMutationSet, TournamentSelection> =
					toml::from_str(&contents)?;
				if let Some(of) = cli.outfile.clone() {
					config.output_dir = of;
				}
				config.params.init_genome = config.problem.init_genome();
				let mut ga = WasmGenAlg::<Sum3>::from_config(config);
				ga.run();
				return Ok(());
			}

			let params = base_params(&problem, &cli, *seed);
			match problem {
				ProblemSet::Sum3(p) => {
					let mut ga = WasmGenAlg::new(p, params);
					ga.run();
				}
				ProblemSet::Sum4(p) => {
					let mut ga = WasmGenAlg::new(p, params);
					ga.run();
				}
				ProblemSet::Polynom2(p) => {
					let mut ga = WasmGenAlg::new(p, params);
					ga.run();
				}
			}
		}
		other => {
			let base_params = base_params(&problem, &cli, None);
			let num_runs = cli.num_runs.unwrap_or(10);
			let control = ExperimentConfig::new("control", problem, base_params);
			let mut exper = other.create_experiment(control, &problem_str, num_runs);
			exper.run();
		}
	}

	Ok(())
}
