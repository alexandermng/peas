//! Speciation threshold sweep experiment.

use std::{
	collections::HashMap,
	fs::{self, File},
	io::Write,
	ops::Range,
	path::Path,
};

use crate::{
	experiments::{DefaultTrialResults, Experiment, ExperimentConfig, ExperimentResults},
	params::SpeciesParams,
};
use polars::prelude::*;

pub type SpeciationExperiment = Experiment<SpeciationExperimentResults>;

/// Generates an experiment based on a selection of speciation thresholds.
pub fn gen_selection(
	name: &str,
	base_config: ExperimentConfig,
	num_runs_per: usize,
	thresholds: impl Iterator<Item = f64>,
) -> SpeciationExperiment {
	let configs: Vec<_> = thresholds
		.map(|threshold| {
			base_config
				.relabel(format!("threshold_{:.1}", threshold)) // TODO figure out num decimal places
				.map_params(|mut p| {
					p.speciation = SpeciesParams {
						enabled: true,
						threshold,
						fitness_sharing: true,
					};
					p
				})
		})
		.collect();
	SpeciationExperiment::new(name, num_runs_per, configs)
}

/// Generates a sweep of speciation thresholds, using the given base config for all other params.
pub fn gen_linspace(
	name: &str,
	base_config: ExperimentConfig,
	range: Range<f64>,
	num_configs: usize,
	num_runs_per: usize,
) -> SpeciationExperiment {
	let step = (range.end - range.start) / (num_configs as f64);
	gen_selection(
		name,
		base_config,
		num_runs_per,
		(0..num_configs).map(|i| range.start + step * (i as f64)), // linspace
	)
}

#[derive(Debug, Clone)]
pub struct SpeciationExperimentResults {
	pub params: HashMap<usize, String>,
	pub data: DataFrame,
	pub hof: HashMap<usize, Vec<u8>>,
}

impl Default for SpeciationExperimentResults {
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

impl ExperimentResults for SpeciationExperimentResults {
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

		let gens_map: HashMap<_, _> = (|| -> polars::prelude::PolarsResult<_> {
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
