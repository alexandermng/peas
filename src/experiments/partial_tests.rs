//! Test the effect of degenerate/partial test cases used in the fitness function.

use std::{
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
	genetic::{GenAlg, Results},
	params::{GenAlgParams, SpeciesParams},
	problems::{ProblemSet, Sum3},
	selection::TournamentSelection,
	wasm::{
		mutations::{AddOperation, ChangeRoot, WasmMutationSet},
		species::WasmSpecies,
		Context, WasmGenAlg, WasmGenome,
	},
	Id,
};

use super::Experiment;

/// Tests performance based on the proportion of partial test cases. The control is with it disabled.
pub struct PartialTestsExperiment {
	/// How many runs to run for each configuration
	pub num_runs_per: usize,

	/// Configurations of the intended parameter to vary, in this case the problem itself, with a partial test case proportion.
	pub configurations: Vec<ProblemSet>,

	// trials: Vec<<Self as Experiment>::GA>,
	outdir: PathBuf,

	/// Collected data, if it has been run already.
	data: Option<DataFrame>,
}

/// Overall results for the entire experiment
#[derive(Debug, Default, Clone)]
pub struct PartialTestsExperimentResults {
	pub data: DataFrame,
}

/// Results for a single trial/run
#[derive(Clone)]
pub struct PartialTestsTrialResults {
	/// Unique ID of the trial
	pub trial_id: usize,

	/// Given proportion parameter for this trial
	pub proportion: f64,

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
	pub fn new(name: &str, num_runs_per: usize, configurations: Vec<ProblemSet>) -> Self {
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

	/// Create an experiment with two configs, with speciation enabled and disabled. `onproportion` is the total partial tests proportion
	/// to test when it's enabled. A good value is `0.9`.
	pub fn gen_control(name: &str, onproportion: f64, num_runs_per: usize) -> Self {
		let one_proportion = onproportion * (0.3 / 0.9) / 3.0; // 3 partial tests of 1
		let two_proportion = onproportion * (0.6 / 0.9) / 3.0; // 3 partial tests of 2, weighted double
		let configs = vec![
			ProblemSet::Sum3(Sum3::new(100, 0.0, 0.0)),
			ProblemSet::Sum3(Sum3::new(100, 0.0, two_proportion)),
			ProblemSet::Sum3(Sum3::new(100, one_proportion, 0.0)),
			ProblemSet::Sum3(Sum3::new(100, one_proportion, two_proportion)),
		];
		Self::new(name, num_runs_per, configs)
	}
}

impl Default for PartialTestsExperiment {
	fn default() -> Self {
		Self::gen_control("partial_tests", 0.6, 10)
	}
}

impl Experiment for PartialTestsExperiment {
	type Genome = WasmGenome;
	type Ctx = Context;
	type ProblemSet = ProblemSet;
	type MutationSet = WasmMutationSet;
	type SelectorSet = TournamentSelection;
	type GA = WasmGenAlg<Sum3, Self::MutationSet, Self::SelectorSet>;
	// TODO ^ fix this. rn hardcoded to sum3.

	fn run(&mut self) {
		fs::create_dir_all(&self.outdir).unwrap();
		// TODO check empty
		log::info!(
			"Beginning Experiment with {} configs x {} runs each ({} total).",
			self.configurations.len(),
			self.num_runs_per,
			self.configurations.len() * self.num_runs_per
		);

		let trial_count = AtomicUsize::new(0);
		let results = Arc::new(RwLock::new(PartialTestsExperimentResults::new()));
		self.configurations
			.par_iter() // Lazy
			.flat_map_iter(|c| iter::repeat_n(c, self.num_runs_per))
			.map(|c| {
				let (problem, proportion) = match &c {
					ProblemSet::Sum3(sum3) => (
						sum3.clone(),
						sum3.partial1_tests_rate * 3.0 + sum3.partial2_tests_rate * 3.0,
					),
					_ => unimplemented!(),
				};
				let params = self.base_params();
				let id = trial_count.fetch_add(1, Ordering::Relaxed);
				let results = PartialTestsTrialResults::new(id, proportion, Arc::clone(&results));
				let ga: Self::GA = WasmGenAlg::new(problem, params, results);
				log::info!("Beginning trial {id}.");
				ga
			})
			.for_each(|mut ga| ga.run()); // Run in parallel!

		let trial_count = trial_count.load(Ordering::Relaxed);
		log::info!("Completed {trial_count} trials.");
		let results = Arc::try_unwrap(results)
			.expect("should only have one reference at this point")
			.into_inner()
			.unwrap();
		let PartialTestsExperimentResults { mut data } = results;
		data.align_chunks_par(); // due to multiple vstacks

		// TODO: configure output, generate graphs
		let datafile = self.outdir.join("data.csv");
		let mut file = File::create(datafile).unwrap();
		CsvWriter::new(&mut file)
			.include_header(true)
			.finish(&mut data)
			.unwrap();

		self.data = Some(data);
	}

	fn base_params(
		&self,
	) -> GenAlgParams<Self::Genome, Self::Ctx, Self::MutationSet, Self::SelectorSet> {
		let muts: Vec<WasmMutationSet> = vec![
			AddOperation::from_rate(0.1).into(), // local variable
			ChangeRoot::from_rate(0.4).into(),   // consts, locals, push onto stack
		];
		let init = self.configurations[0].init_genome();
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
			"proportion" => Vec::<f64>::new(),
			"generation" => Vec::<u32>::new(),
			"num_species" => Vec::<u32>::new(),
			"avg_fitness" => Vec::<f64>::new(),
			"max_fitness" => Vec::<f64>::new(),
		)
		.unwrap();
		Self { data }
	}
	pub fn report_trial(
		&mut self,
		trial_id: u32,
		proportion: f64,
		results: &PartialTestsTrialResults,
	) {
		let mut trial_data = results.to_data();
		let ids = Series::new("trial_id".into(), vec![trial_id; trial_data.height()]);
		let proportions = Series::new("proportion".into(), vec![proportion; trial_data.height()]);
		trial_data.insert_column(0, ids);
		trial_data.insert_column(1, proportions);

		self.data.vstack_mut_owned_unchecked(trial_data);
		log::info!(
			"Collected data from trial {trial_id} (proportion {proportion:.3}, {} generations).",
			results.num_generations
		);
	}
}

impl PartialTestsTrialResults {
	pub fn new(
		trial_id: usize,
		proportion: f64,
		experiment_data: Arc<RwLock<PartialTestsExperimentResults>>,
	) -> Self {
		Self {
			trial_id,
			proportion,
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
		exper.report_trial(self.trial_id as u32, self.proportion, self);
		log::info!(
			"Completed trial {} ({}, {:.3} secs).",
			self.trial_id,
			if self.success { "success" } else { "failure" },
			self.time_taken
		);
	}
}
