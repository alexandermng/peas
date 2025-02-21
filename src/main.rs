use std::fs;

use clap::{Parser, ValueEnum};
use eyre::eyre;
use peas::{
	experiments::{speciation::SpeciationExperiment, Experiment},
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

/// Input command-line arguments
#[derive(clap::Parser, Debug)]
pub struct GenAlgParamsCLI {
	/// Experiment
	#[arg(short, long, value_enum)]
	pub experiment: Option<AvailableExperiments>,

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

#[derive(ValueEnum, Clone, Debug)]
pub enum AvailableExperiments {
	SpeciationControl,
	SpeciationRange,
}

type DynExperiment = dyn Experiment<
	Genome = WasmGenome,
	Ctx = Context,
	ProblemSet = ProblemSet,
	MutationSet = WasmMutationSet,
	SelectorSet = TournamentSelection,
	GA = WasmGenAlg<Sum3, WasmMutationSet, TournamentSelection>,
>;
impl AvailableExperiments {
	pub fn get_experiment(&self) -> Box<DynExperiment> {
		match self {
			AvailableExperiments::SpeciationControl => Box::new(SpeciationExperiment::gen_control(
				"speciation_control",
				2.0,
				100,
			)),
			AvailableExperiments::SpeciationRange => Box::new(SpeciationExperiment::gen_linspace(
				"speciation_range",
				1.5..2.5,
				10,
				10,
			)),
		}
	}
}

fn main() -> eyre::Result<()> {
	pretty_env_logger::init();

	let args = GenAlgParamsCLI::parse();

	// Pre-configured Experiments
	if let Some(experiment) = args.experiment {
		let mut exper = experiment.get_experiment();
		exper.run();

		return Ok(());
	}

	if let Some(filename) = args.config {
		// replay a run
		let contents = fs::read_to_string(filename)?;
		let mut config: WasmGenAlgConfig<WasmMutationSet, TournamentSelection> =
			toml::from_str(&contents)?;
		if let Some(of) = args.outfile {
			// mem::take?
			config.output_dir = of;
		}
		// TODO be able to pull in an init.wasm or smthn.
		config.params.init_genome = config.problem.init_genome();
		let mut ga = WasmGenAlg::<Sum3>::from_config(config);
		ga.run();
		return Ok(());
	}

	let problem = args.problem.unwrap_or_else(|| String::from("sum3"));
	let num_tests = 100; // TODO put in args. or even better, put in ProblemParams
	let problem = match &*problem {
		"sum3" => ProblemSet::Sum3(Sum3::new(num_tests, 0.1, 0.2)),
		"sum4" => ProblemSet::Sum4(Sum4::new(num_tests, 0.02, 0.04, 0.08)),
		"poly2" => ProblemSet::Polynom2(Polynom::new(num_tests, 0.3)),
		_ => return Err(eyre!("Unknown problem")),
	};
	let init = problem.init_genome();
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
		.selector(TournamentSelection::new(0.6, 2, 0.9, true)) // can do real tournament selection when selection is fixed
		.speciation(SpeciesParams {
			enabled: true,
			threshold: 2.0,
			fitness_sharing: true,
		})
		.init_genome(init)
		.elitism_rate(0.05)
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
