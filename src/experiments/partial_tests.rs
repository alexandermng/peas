//! Full PartialTests Study, including all configurations of the GA with different features turned on or off.
//! Feature configuration parameters are found from other experiments.

use std::{
	collections::HashMap,
	fs::{self, File},
	iter,
	ops::Range,
	path::PathBuf,
	sync::{
		atomic::{AtomicU32, AtomicUsize, Ordering},
		RwLock,
	},
	time::Instant,
};

use polars::prelude::*;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;

use crate::{
	genetic::{Configurable, GenAlg, Genome, Results},
	params::{GenAlgParams, GenAlgParamsBuilder, ResultsParams, SpeciesParams},
	prelude::Problem,
	problems::{Polynom, ProblemSet, Sum3, Sum4},
	selection::TournamentSelection,
	wasm::{
		mutations::{AddOperation, ChangeRoot, WasmMutationSet},
		species::WasmSpecies,
		Context, WasmGenAlg, WasmGenAlgConfig, WasmGenome,
	},
	Id,
};

use super::Experiment;

#[derive(Debug, Clone)]
pub struct ExperimentConfig {
	/// Name representing the configuration parameters. Not unique across multiple trials.
	pub label: String,

	/// Problem to solve
	pub problem: ProblemSet,

	/// Input Parameters for Genetic Algorithm
	pub params: GenAlgParams<WasmGenome, Context, WasmMutationSet, TournamentSelection>,
}

/// Tests performance based on the proportion of partial test cases. The control is with it disabled.
pub struct PartialTestsExperiment {
	/// How many runs to run for each configuration
	pub num_runs_per: usize,

	/// Configurations with different parameters to test
	pub configurations: Vec<ExperimentConfig>,

	// trials: Vec<<Self as Experiment>::GA>,
	outdir: PathBuf,

	/// Collected data, if it has been run already.
	data: Option<DataFrame>,
}

/// Overall results for the entire experiment
#[derive(Debug, Default, Clone)]
pub struct PartialTestsExperimentResults {
	/// Map of trial ID to Parameter configuration
	pub params: HashMap<usize, ExperimentConfig>,

	/// Result data collated across all trials
	pub data: DataFrame,
}

/// Results for a single trial/run
#[derive(Clone)]
pub struct PartialTestsTrialResults {
	/// Unique ID of the trial
	pub trial_id: usize,

	/// Number of generations at completion
	pub num_generations: usize,

	/// Number of species, by generation
	pub num_species: Vec<u32>,
	/// Average fitness, by generation
	pub avg_fitnesses: Vec<f64>,
	/// Max fitness, by generation
	pub max_fitnesses: Vec<f64>,

	/// Total time elapsed for the run, in seconds
	pub time_taken: f64,

	/// Whether it succeeded or not
	pub success: bool,

	experiment_data: Arc<RwLock<PartialTestsExperimentResults>>,
}

impl PartialTestsExperiment {
	pub fn new(name: &str, num_runs_per: usize, configurations: Vec<ExperimentConfig>) -> Self {
		let mut outdir = PathBuf::new();
		outdir.push("data");
		outdir.push(name); // experiment name
		Self {
			num_runs_per,
			configurations,
			// trials: Vec::new(),
			outdir,
			data: None,
		}
	}

	/// Create an experiment with a range of partial test proportions, for the given problem.
	/// Currently the rates given in the problem are ignored.
	pub fn gen_basic(name: &str, problem: ProblemSet, num_runs_per: usize) -> Self {
		let base_params = {
			let mut p = PartialTestsExperiment::new("empty", 0, vec![]).base_params(); // ugly circular, but works. will get warning, but this fixes
			p.init_genome = problem.init_genome();
			p
		};
		let problemsets = match problem {
			ProblemSet::Sum3(orig) => vec![
				ProblemSet::Sum3(Sum3::new(orig.num_tests, 0.0, 0.0)),
				ProblemSet::Sum3(Sum3::new(orig.num_tests, 0.1, 0.0)),
				ProblemSet::Sum3(Sum3::new(orig.num_tests, 0.0, 0.2)),
				ProblemSet::Sum3(Sum3::new(orig.num_tests, 0.1, 0.2)),
				ProblemSet::Sum3(Sum3::new(orig.num_tests, 0.2, 0.1)),
			],
			ProblemSet::Sum4(orig) => vec![
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.0, 0.0, 0.0)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.02, 0.04, 0.08)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.08, 0.04, 0.02)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.1, 0.0, 0.0)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.0, 0.1, 0.0)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.0, 0.0, 0.1)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.04, 0.04, 0.0)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.0, 0.04, 0.04)),
				ProblemSet::Sum4(Sum4::new(orig.num_tests, 0.04, 0.0, 0.04)),
			],
			_ => vec![problem], // singleton
		};
		let configs = problemsets
			.into_iter()
			.map(|p| ExperimentConfig {
				label: match &p {
					ProblemSet::Sum3(pr) => format!(
						"sum3_{:.1}_{:.1}",
						pr.partial1_tests_rate, pr.partial2_tests_rate
					),
					ProblemSet::Sum4(pr) => format!(
						"sum4_{:.2}_{:.2}_{:.2}",
						pr.partial1_tests_rate, pr.partial2_tests_rate, pr.partial3_tests_rate,
					),
					_ => "unknown".to_owned(),
				},
				problem: p.clone(),
				params: base_params.clone(),
			})
			.collect();
		Self::new(name, num_runs_per, configs)
	}
}

impl Default for PartialTestsExperiment {
	fn default() -> Self {
		Self::gen_basic(
			"partial_tests",
			ProblemSet::Sum3(Sum3::new(100, 0.1, 0.2)),
			10,
		)
	}
}

impl Experiment for PartialTestsExperiment {
	type Genome = WasmGenome;
	type Ctx = Context;
	type ProblemSet = ProblemSet;
	type MutationSet = WasmMutationSet;
	type SelectorSet = TournamentSelection;
	type GA = WasmGenAlg<Sum3, Self::MutationSet, Self::SelectorSet>;
	// TODO ^ fix this. rn hardcoded to sum3, but also not rly used rn.

	fn run(&mut self) {
		fs::create_dir_all(&self.outdir).unwrap();
		// TODO check empty
		log::info!(
			"Beginning ablation experiment on problem {} :: {} configs x {} runs each ({} total).",
			self.configurations[0].problem, // since all configs have same problem (for now)
			self.configurations.len(),
			self.num_runs_per,
			self.configurations.len() * self.num_runs_per
		);
		let timer = Instant::now();

		let trial_count = AtomicUsize::new(0);
		let results = Arc::new(RwLock::new(PartialTestsExperimentResults::new()));
		self.configurations
			.par_iter() // Lazy
			.flat_map_iter(|c| iter::repeat_n(c, self.num_runs_per))
			.map(|cfg| {
				let id = trial_count.fetch_add(1, Ordering::Relaxed);
				results.write().unwrap().params.insert(id, cfg.clone());
				let params = cfg.params.clone();
				let results = ResultsParams::from_single(PartialTestsTrialResults::new(
					id,
					Arc::clone(&results),
				));
				let ga = WasmGenAlg::<Sum3>::from_config(
					WasmGenAlgConfig::builder()
						.problem(cfg.problem.clone())
						.params(params)
						.results(results)
						.output_dir(self.outdir.to_str().unwrap().to_owned())
						.build(),
				);
				log::info!("Beginning trial {id} :: {}, {}", cfg.problem, cfg.label);
				ga
			})
			.for_each(|mut ga| ga.run()); // Run in parallel!

		let trial_count = trial_count.load(Ordering::Relaxed);
		log::info!("Completed {trial_count} trials.");
		let results = Arc::try_unwrap(results)
			.expect("should only have one reference at this point")
			.into_inner()
			.unwrap();
		let PartialTestsExperimentResults {
			mut data,
			params: trials,
		} = results;
		data.align_chunks_par(); // due to multiple vstacks

		// TODO: configure output, generate graphs
		let datafile = self.outdir.join("data.csv");
		let mut file = File::create(datafile).unwrap();
		CsvWriter::new(&mut file)
			.include_header(true)
			.finish(&mut data)
			.unwrap();

		let fail_gens = self.base_params().num_generations as f64;
		let mut gens_per_trial = data
			.clone()
			.lazy()
			.group_by([col("trial_id")])
			.agg([
				col("label").first(),
				col("generation").len().alias("num_generations"),
				col("max_fitness").last().eq(lit(1.0)).alias("success"),
			]) // number of generations per trial
			.collect()
			.unwrap();
		let mut gens_file = File::create(self.outdir.join("generations.csv")).unwrap();
		CsvWriter::new(&mut gens_file)
			.include_header(true)
			.finish(&mut gens_per_trial)
			.unwrap();
		let gens = gens_per_trial
			.lazy()
			.group_by([col("label").sort(Default::default())])
			.agg([
				col("num_generations").mean().alias("avg_gens"),
				col("num_generations").std(0).alias("stddev_gens"),
				col("num_generations").min().alias("min_gens"),
				col("num_generations").max().alias("max_gens"),
				col("success").mean().alias("success_rate"),
			]) // final generation stats
			.collect()
			.unwrap();
		log::info!(
			"Generations:\n{gens}\nTotal Experiment Time: {:.3} secs.",
			timer.elapsed().as_secs_f64()
		);

		self.data = Some(data);
	}

	fn base_params(
		&self,
	) -> GenAlgParams<Self::Genome, Self::Ctx, Self::MutationSet, Self::SelectorSet> {
		let muts: Vec<WasmMutationSet> = vec![
			AddOperation::from_rate(0.1).into(), // local variable
			ChangeRoot::from_rate(0.4).into(),   // consts, locals, push onto stack
		];
		let init = if !self.configurations.is_empty() {
			self.configurations[0].problem.clone()
		} else {
			log::warn!("Problem not found for init genome, defaulting to Sum3.");
			ProblemSet::Sum3(Sum3::new(100, 0.0, 0.0))
		}
		.init_genome();
		let seed: u64 = rand::rng().random();
		GenAlgParams::builder()
			.seed(seed)
			.pop_size(100)
			.num_generations(299)
			.max_fitness(1.0)
			.mutators(muts)
			.mutation_rate(1.0)
			.selector(TournamentSelection::new(0.6, 2, 0.9, true))
			.speciation(SpeciesParams {
				enabled: true,
				threshold: 1.2,
				fitness_sharing: true,
			})
			.init_genome(init)
			.elitism_rate(0.05)
			.crossover_rate(0.8)
			.build()
	}

	fn outdir(&self) -> &std::path::Path {
		&self.outdir
	}
}

impl PartialTestsExperimentResults {
	pub fn new() -> Self {
		let data = df!(
			"trial_id" => Vec::<u32>::new(),
			"label" => Vec::<String>::new(),
			"generation" => Vec::<u32>::new(),
			"num_species" => Vec::<u32>::new(),
			"avg_fitness" => Vec::<f64>::new(),
			"max_fitness" => Vec::<f64>::new(),
		)
		.unwrap();
		Self {
			data,
			params: HashMap::new(),
		}
	}
	pub fn report_trial(&mut self, trial_id: u32, results: &PartialTestsTrialResults) {
		let mut trial_data = results.to_data();
		let ids = Series::new("trial_id".into(), vec![trial_id; trial_data.height()]);
		let label = self.params[&(trial_id as usize)].label.clone();
		let labels = Series::new("label".into(), vec![label.clone(); trial_data.height()]);
		trial_data.insert_column(0, ids);
		trial_data.insert_column(1, labels);

		self.data.vstack_mut_owned_unchecked(trial_data);
		log::info!(
			"Collected data from trial {trial_id} :: {label} ({} generations).",
			results.num_generations
		);
	}
}

impl PartialTestsTrialResults {
	pub fn new(
		trial_id: usize,
		experiment_data: Arc<RwLock<PartialTestsExperimentResults>>,
	) -> Self {
		Self {
			trial_id,
			num_generations: 0,
			num_species: Vec::new(),
			avg_fitnesses: Vec::new(),
			max_fitnesses: Vec::new(),
			experiment_data,
			time_taken: 0.0,
			success: false,
		}
	}

	pub fn to_data(&self) -> DataFrame {
		let generations: Vec<u32> = (0..(self.num_generations as u32)).collect();
		let num_species = if self.num_species.is_empty() {
			// speciation is disabled, so set to all 0s
			&vec![0; self.num_generations]
		} else {
			&self.num_species
		};
		DataFrame::new(vec![
			Column::new("generation".into(), generations),
			Column::new("num_species".into(), &num_species),
			Column::new("avg_fitness".into(), &self.avg_fitnesses),
			Column::new("max_fitness".into(), &self.max_fitnesses),
		])
		.unwrap()
	}
}

impl Results for PartialTestsTrialResults {
	type Genome = WasmGenome;
	type Ctx = Context;

	fn initialize(&mut self, ctx: &mut Self::Ctx) {
		ctx.start_time = Instant::now();
	}

	fn record_generation(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>]) {
		self.avg_fitnesses.push(ctx.avg_fitness);
		self.max_fitnesses.push(ctx.max_fitness);
	}

	fn record_speciation(&mut self, ctx: &mut Self::Ctx, species: &[Id<WasmSpecies>]) {
		self.num_species.push(species.len() as u32)
	}

	fn record_success(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>]) {
		self.success = true;
	}

	fn finalize(
		&mut self,
		ctx: &mut Self::Ctx,
		pop: &[Id<Self::Genome>],
		outdir: &std::path::Path,
	) {
		self.time_taken = ctx.start_time.elapsed().as_secs_f64();
		self.num_generations = ctx.generation + 1; // since 0-indexed
		let mut exper = self.experiment_data.write().unwrap_or_else(|p| {
			log::error!("Lock Poison error: {p}");
			p.into_inner()
		});
		exper.report_trial(self.trial_id as u32, self);
		log::info!(
			"Completed trial {} ({}, {:.3} secs).",
			self.trial_id,
			if self.success { "success" } else { "failure" },
			self.time_taken
		);
	}
}
