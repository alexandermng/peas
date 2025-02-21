//! Test the effect of speciation.

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

/// Tests performance based on the compatibility threshold in speciation. The control is with it disabled.
/// Aggregates data about the number of species per generation
pub struct SpeciationExperiment {
	/// which problem to be acting on
	pub problem: ProblemSet,

	/// How many runs to run for each configuration
	pub num_runs_per: usize,

	/// Configurations of the intended parameter to vary, in this case the species paramaters (in particular, threshold).
	pub configurations: Vec<SpeciesParams>,

	// trials: Vec<<Self as Experiment>::GA>,
	outdir: PathBuf,

	/// Collected data, if it has been run already.
	data: Option<DataFrame>,
}

/// Overall results for the entire experiment
#[derive(Debug, Default, Clone)]
pub struct SpeciationExperimentResults {
	pub data: DataFrame,
}

/// Results for a single trial/run
#[derive(Clone)]
pub struct SpeciationTrialResults {
	/// Unique ID of the trial
	pub trial_id: usize,

	/// Given threshold parameter for this trial
	pub threshold: f64,

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

	experiment_data: Arc<RwLock<SpeciationExperimentResults>>,
}

impl SpeciationExperiment {
	pub fn new(
		name: &str,
		problem: ProblemSet,
		num_runs_per: usize,
		configurations: Vec<SpeciesParams>,
	) -> Self {
		let mut outdir = PathBuf::new();
		outdir.push("data");
		outdir.push(name); // experiment name
		Self {
			problem,
			num_runs_per,
			configurations,
			// trials: Vec::new(),
			outdir,
			data: None,
		}
	}

	/// Creates a speciation experiment from a linear space of the thresholds, as provided
	/// by the range and number of configs to spread along it. Does not include a control.
	pub fn gen_linspace(
		name: &str,
		range: Range<f64>,
		num_configs: usize,
		num_runs_per: usize,
	) -> Self {
		let linspace: Vec<_> = {
			let step = (range.end - range.start) / (num_configs as f64);
			let mut count = range.start - step;
			iter::from_fn(move || {
				count += step;
				if range.contains(&count) {
					Some(count)
				} else {
					None
				}
			})
			.map(|threshold| SpeciesParams {
				enabled: true,
				threshold,
				fitness_sharing: true,
			})
			.collect()
		};
		let problem = ProblemSet::Sum3(Sum3::new(100, 0.1, 0.2));
		Self::new(name, problem, num_runs_per, linspace)
	}

	/// Create an experiment with two configs, with speciation enabled and disabled. `onthreshold` is the threshold
	/// to test when it's enabled.
	pub fn gen_control(name: &str, onthreshold: f64, num_runs_per: usize) -> Self {
		let problem = ProblemSet::Sum3(Sum3::new(100, 0.1, 0.2));
		let configs = vec![
			SpeciesParams {
				enabled: false,
				threshold: 0.0,
				fitness_sharing: false,
			},
			SpeciesParams {
				enabled: true,
				threshold: onthreshold,
				fitness_sharing: true,
			},
		];
		Self::new(name, problem, num_runs_per, configs)
	}
}

impl Default for SpeciationExperiment {
	fn default() -> Self {
		Self::gen_linspace("speciation", 0.0..5.0, 10, 3)
	}
}

impl Experiment for SpeciationExperiment {
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
		let results = Arc::new(RwLock::new(SpeciationExperimentResults::new()));
		self.configurations
			.par_iter() // Lazy
			.flat_map_iter(|c| iter::repeat_n(c, self.num_runs_per))
			.map(|c| {
				let problem = match &self.problem {
					ProblemSet::Sum3(sum3) => sum3.clone(),
					_ => unimplemented!(),
				};
				let mut params = self.base_params();
				params.speciation = c.clone();
				let id = trial_count.fetch_add(1, Ordering::Relaxed);
				let results = SpeciationTrialResults::new(
					id,
					params.speciation.threshold,
					Arc::clone(&results),
				);
				let ga: Self::GA = WasmGenAlg::new(problem, params, results);
				log::info!("Beginning trial {id}.");
				ga
			})
			.for_each(|mut ga| ga.run()); // Run in parallel!

		/*
		TODO: do 1 run for each and then prune early (don't do extra trials if no fitness).
		or better, to systematically search, arrange batches around successful thresholds or smthn.
		need some way of intelligently finding.
		 */
		let trial_count = trial_count.load(Ordering::Relaxed);
		log::info!("Completed {trial_count} trials.");
		let results = Arc::try_unwrap(results)
			.expect("should only have one reference at this point")
			.into_inner()
			.unwrap();
		let SpeciationExperimentResults { mut data } = results;
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
		let init = self.problem.init_genome();
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
				enabled: false,
				threshold: 0.0,
				fitness_sharing: false,
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

impl SpeciationExperimentResults {
	pub fn new() -> Self {
		let data = df!(
			"trial_id" => Vec::<u32>::new(),
			"threshold" => Vec::<f64>::new(),
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
		threshold: f64,
		results: &SpeciationTrialResults,
	) {
		let mut trial_data = results.to_data();
		let ids = Series::new("trial_id".into(), vec![trial_id; trial_data.height()]);
		let thresholds = Series::new("threshold".into(), vec![threshold; trial_data.height()]);
		trial_data.insert_column(0, ids);
		trial_data.insert_column(1, thresholds);

		self.data.vstack_mut_owned_unchecked(trial_data);
		log::info!(
			"Collected data from trial {trial_id} (threshold {threshold:.3}, {} generations).",
			results.num_generations
		);
	}
}

impl SpeciationTrialResults {
	pub fn new(
		trial_id: usize,
		threshold: f64,
		experiment_data: Arc<RwLock<SpeciationExperimentResults>>,
	) -> Self {
		Self {
			trial_id,
			threshold,
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

impl Results for SpeciationTrialResults {
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
		exper.report_trial(self.trial_id as u32, self.threshold, self);
		log::info!(
			"Completed trial {} ({}, {:.3} secs).",
			self.trial_id,
			if self.success { "success" } else { "failure" },
			self.time_taken
		);
	}
}
