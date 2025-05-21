pub mod ablation;
pub mod partial_tests;
pub mod speciation;

use std::{
	any::Any,
	fs, mem,
	path::{Path, PathBuf},
	sync::{
		atomic::{AtomicUsize, Ordering},
		Arc, RwLock,
	},
	time::Instant,
};

use downcast_rs::Downcast;
use polars::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
	genetic::{Configurable, Results},
	params::ResultsParams,
	problems::{Polynom, ProblemSet, Sum3, Sum4},
	selection::TournamentSelection,
	wasm::{
		mutations::WasmMutationSet, species::WasmSpecies, Context, WasmGenAlg, WasmGenAlgConfig,
		WasmGenAlgParams, WasmGenome,
	},
	Id,
};

/// Trait for experiment result aggregation and output.
pub trait ExperimentResults: Default + Send + Sync + 'static {
	type TrialResults: Results + Any;

	/// Collect results from a single trial.
	fn collect(&mut self, trial: &Self::TrialResults);

	/// Output all results to disk.
	fn output(&mut self, outdir: &Path);
}

/// An experiment to run, generic over the outputted results type.
pub struct Experiment<R: ExperimentResults> {
	pub num_runs_per: usize,
	pub configurations: Vec<ExperimentConfig>,
	pub outdir: PathBuf,
	pub results: R,
}

impl<R: ExperimentResults> Experiment<R> {
	pub fn new(name: &str, num_runs_per: usize, configurations: Vec<ExperimentConfig>) -> Self {
		let mut outdir = PathBuf::new();
		outdir.push("data");
		outdir.push(name);
		Self {
			num_runs_per,
			configurations,
			outdir,
			results: R::default(),
		}
	}

	pub fn run(&mut self) {
		fs::create_dir_all(&self.outdir).unwrap();
		// TODO check empty
		log::info!(
			"Beginning experiment on problem {} :: {} configs x {} runs each ({} total).",
			self.configurations[0].problem,
			self.configurations.len(),
			self.num_runs_per,
			self.configurations.len() * self.num_runs_per
		);
		let timer = Instant::now();

		let trial_count = AtomicUsize::new(0);
		self.results = {
			let results = Arc::new(RwLock::new(mem::take(&mut self.results)));
			self.configurations
				.par_iter()
				.flat_map_iter(|c| std::iter::repeat_n(c, self.num_runs_per))
				.map(|cfg| {
					let id = trial_count.fetch_add(1, Ordering::Relaxed);
					let params = cfg.params.clone();
					let trial_results = DefaultTrialResults::new(id); // TODO fix into R::TrialResults
					let results = ResultsParams::from_single(trial_results.clone());
					let ga = WasmGenAlg::<Sum3>::from_config(
						WasmGenAlgConfig::builder()
							.problem(cfg.problem.clone())
							.params(params)
							.results(results)
							.output_dir(self.outdir.to_str().unwrap().to_owned())
							.build(),
					);
					log::info!("Beginning trial {id} :: {}, {}", cfg.problem, cfg.label);
					(ga, id)
				})
				.for_each(|(mut ga, id)| {
					let trial_results = ga.run();
					for res in trial_results {
						// FIXME: big bug, data collection not being run.
						if let Some(r) = res.as_any().downcast_ref::<R::TrialResults>() {
							results.write().unwrap().collect(r);
						}
					}
				});
			Arc::into_inner(results)
				.expect("parallel execution should have completed")
				.into_inner()
				.unwrap()
		};

		let trial_count = trial_count.load(Ordering::Relaxed);
		log::info!("Completed {trial_count} trials.");
		self.results.output(&self.outdir);
		log::info!(
			"Total Experiment Time: {:.3} secs.",
			timer.elapsed().as_secs_f64()
		);
	}
}

/// Helper trait for running an `Experiment` in a trait-object. For use as `Box<dyn ExperimentRunnable>`.
pub trait ExperimentRunnable {
	fn run(&mut self);
}

impl<R: ExperimentResults> ExperimentRunnable for Experiment<R> {
	fn run(&mut self) {
		self.run();
	}
}

pub type WasmExperimentParams = WasmGenAlgParams<WasmMutationSet, TournamentSelection>;

#[derive(Debug, Clone)]
pub struct ExperimentConfig {
	/// Name representing the configuration parameters. Not unique across multiple trials.
	pub label: String,
	/// Problem to solve
	pub problem: ProblemSet,
	/// Input Parameters for Genetic Algorithm
	pub params: WasmExperimentParams,
}

impl ExperimentConfig {
	pub fn new(label: &str, problem: ProblemSet, params: WasmExperimentParams) -> Self {
		Self {
			label: label.to_string(),
			problem,
			params,
		}
	}

	/// Clones the config with a new label.
	pub fn relabel(&self, label: impl AsRef<str>) -> Self {
		Self {
			label: label.as_ref().to_owned(), // ok i know i should just take in String but i want to simplify calling
			..self.clone()
		}
	}

	pub fn map_problem(self, f: impl FnOnce(ProblemSet) -> ProblemSet) -> Self {
		Self {
			problem: f(self.problem),
			..self
		}
	}

	pub fn map_params(self, f: impl FnOnce(WasmExperimentParams) -> WasmExperimentParams) -> Self {
		Self {
			params: f(self.params),
			..self
		}
	}

	pub fn no_speciation(self) -> Self {
		self.map_params(|mut p| {
			p.speciation.enabled = false;
			p
		})
	}
	pub fn no_crossover(self) -> Self {
		self.map_params(|mut p| {
			p.speciation.enabled = false;
			p.crossover_rate = 0.0;
			p
		})
	}
	pub fn no_elitism(self) -> Self {
		self.map_params(|mut p| {
			p.elitism_rate = 0.0;
			p
		})
	}
	pub fn no_partials(self) -> Self {
		self.map_problem(|mut p| match p {
			ProblemSet::Sum3(Sum3 { num_tests, .. }) => {
				ProblemSet::Sum3(Sum3::new(num_tests, 0.0, 0.0))
			}
			ProblemSet::Sum4(Sum4 { num_tests, .. }) => {
				ProblemSet::Sum4(Sum4::new(num_tests, 0.0, 0.0, 0.0))
			}
			ProblemSet::Polynom2(Polynom::<2> { num_tests, .. }) => {
				ProblemSet::Polynom2(Polynom::<2>::new(num_tests, 0.0))
			}
		})
	}
}

/// Default trial results (was AblationTrialResults)
#[derive(Clone)]
pub struct DefaultTrialResults {
	pub trial_id: usize,
	pub num_generations: usize,
	pub num_species: Vec<u32>,
	pub avg_fitnesses: Vec<f64>,
	pub max_fitnesses: Vec<f64>,
	pub time_taken: f64,
	pub success: bool,
	pub hof: Option<Vec<u8>>, // best genome on success
}

impl DefaultTrialResults {
	pub fn new(trial_id: usize) -> Self {
		Self {
			trial_id,
			num_generations: 0,
			num_species: Vec::new(),
			avg_fitnesses: Vec::new(),
			max_fitnesses: Vec::new(),
			time_taken: 0.0,
			success: false,
			hof: None,
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

impl Results for DefaultTrialResults {
	type Genome = WasmGenome;
	type Ctx = Context;

	fn initialize(&mut self, ctx: &mut Self::Ctx) {
		ctx.start_time = Instant::now();
	}

	fn record_generation(&mut self, ctx: &mut Self::Ctx, _pop: &[Id<Self::Genome>]) {
		self.avg_fitnesses.push(ctx.avg_fitness);
		self.max_fitnesses.push(ctx.max_fitness);
	}

	fn record_speciation(&mut self, _ctx: &mut Self::Ctx, species: &[Id<WasmSpecies>]) {
		self.num_species.push(species.len() as u32)
	}

	fn record_success(&mut self, ctx: &mut Self::Ctx, pop: &[Id<Self::Genome>]) {
		self.success = true;
		// Store the best genome in hof
		let genome = ctx.get_genome(pop[0]).emit();
		self.hof = Some(genome);
	}

	fn finalize(
		&mut self,
		ctx: &mut Self::Ctx,
		_pop: &[Id<Self::Genome>],
		_outdir: &std::path::Path,
	) {
		self.time_taken = ctx.start_time.elapsed().as_secs_f64();
		self.num_generations = ctx.generation + 1; // since 0-indexed

		// no innate output; handled by ExperimentResults
	}
}
