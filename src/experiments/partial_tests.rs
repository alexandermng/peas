//! Full PartialTests Study, including all configurations of the GA with different features turned on or off.

use std::{
	collections::HashMap,
	fs::{self, File},
	io::Write,
	path::Path,
};

use polars::prelude::*;

use crate::{
	experiments::{DefaultTrialResults, Experiment, ExperimentConfig, ExperimentResults},
	problems::{ProblemSet, Sum3, Sum4},
};

pub type PartialTestsExperiment = Experiment<PartialTestsExperimentResults>;

/// Overall results for the entire experiment
#[derive(Debug, Clone)]
pub struct PartialTestsExperimentResults {
	/// Map of trial ID to Parameter configuration
	pub params: HashMap<usize, String>,

	/// Result data collated across all trials
	pub data: DataFrame,

	/// Hall of fame, keyed by trial ID
	pub hof: HashMap<usize, Vec<u8>>,
}

/// Basic experiment suite for partial tests studies, given a base configuration which defines the problem partial test case rates.
/// This is used as the base problem and parameters for all configurations variants.
pub fn gen_basic(
	name: &str,
	base_config: ExperimentConfig,
	num_runs_per: usize,
) -> PartialTestsExperiment {
	let ExperimentConfig {
		problem, params, ..
	} = base_config;
	// TODO extract partial test case rates from base_config
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
		p => vec![p], // singleton, since unknown problem
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
			problem: p,
			params: params.clone(),
		})
		.collect();
	PartialTestsExperiment::new(name, num_runs_per, configs)
}

/// Runs the partial tests experiment with only the given configurations, designated by label.
pub fn gen_custom(
	name: &str,
	control: ExperimentConfig,
	num_runs_per: usize,
	enabled_configs: &[&str],
) -> PartialTestsExperiment {
	let mut out = gen_basic(name, control, num_runs_per);
	out.configurations.retain(|c| {
		enabled_configs
			.iter()
			.any(|label| label == &c.label.as_str())
	});
	out
}

impl Default for PartialTestsExperimentResults {
	fn default() -> Self {
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
			hof: HashMap::new(),
		}
	}
}

impl ExperimentResults for PartialTestsExperimentResults {
	type TrialResults = DefaultTrialResults;

	fn register(&mut self, trial_id: usize, label: String) {
		log::info!("Beginning trial {trial_id}/{label}");
		self.params.insert(trial_id, label);
	}

	fn collect(&mut self, trial: &Self::TrialResults) {
		let mut trial_data = trial.to_data();
		let trial_id = trial.trial_id as u32;
		let ids = Series::new("trial_id".into(), vec![trial_id; trial_data.height()]);
		let label = self.params[&trial.trial_id].clone();
		let labels = Series::new("label".into(), vec![label.clone(); trial_data.height()]);
		trial_data.insert_column(0, ids);
		trial_data.insert_column(1, labels);
		self.data.vstack_mut_owned_unchecked(trial_data);

		if let Some(ref genome) = trial.hof {
			self.hof.insert(trial.trial_id, genome.clone());
		}

		log::info!(
			"Completed trial {trial_id}/{label}\t({:.3}s, {} gens)",
			trial.time_taken,
			trial.num_generations
		);
	}

	fn output(&mut self, outdir: &Path) {
		self.data.align_chunks_par();

		let Self {
			data,
			params: trials,
			hof,
		} = self;
		data.align_chunks_par();

		let datafile = outdir.join("data.csv");
		let mut file = File::create(datafile).unwrap();
		CsvWriter::new(&mut file)
			.include_header(true)
			.finish(data)
			.unwrap();

		let mut gens_per_trial = data
			.clone()
			.lazy()
			.group_by([col("trial_id")])
			.agg([
				col("label").first(),
				col("generation").len().alias("num_generations"),
				col("max_fitness").last().eq(lit(1.0)).alias("success"),
			])
			.collect()
			.unwrap();
		let mut gens_file = File::create(outdir.join("generations.csv")).unwrap();
		CsvWriter::new(&mut gens_file)
			.include_header(true)
			.finish(&mut gens_per_trial)
			.unwrap();
		let gens_stats = gens_per_trial
			.clone()
			.lazy()
			.group_by([col("label").sort(Default::default())])
			.agg([
				col("num_generations").mean().alias("avg_gens"),
				col("num_generations").std(0).alias("stddev_gens"),
				col("num_generations").min().alias("min_gens"),
				col("num_generations").max().alias("max_gens"),
				col("success").mean().alias("success_rate"),
			])
			.collect()
			.unwrap();
		log::info!("Overview:\n{gens_stats}");

		let gens_map: HashMap<_, _> = (|| -> PolarsResult<_> {
			let trial_id = gens_per_trial["trial_id"].u32()?.into_no_null_iter();
			let num_gens = gens_per_trial["num_generations"].u32()?.into_no_null_iter();
			Ok(trial_id
				.zip(num_gens)
				.map(|(a, b)| (a as usize, b))
				.collect())
		})()
		.unwrap();
		let hofdir = outdir.join("hof");
		fs::create_dir_all(&hofdir).unwrap();
		for (trial_id, genome) in hof {
			let label = trials[&trial_id].as_str();
			let eration = gens_map[&trial_id];
			let path = hofdir.join(format!("{label}_gen{eration}_id{trial_id}.wasm"));
			let mut file = File::create(path).unwrap();
			file.write_all(&genome).unwrap();
		}
	}
}
