//! Full Ablation Study, including all configurations of the GA with different features turned on or off.
//! Feature configuration parameters are found from other experiments.

use std::{
	collections::HashMap,
	fs::{self, File},
	io::Write,
	path::Path,
};

use polars::prelude::*;

use super::{DefaultTrialResults, Experiment, ExperimentConfig, ExperimentResults};

pub type AblationExperiment = Experiment<AblationExperimentResults>;

/// Overall results for the entire experiment
#[derive(Debug, Clone)]
pub struct AblationExperimentResults {
	/// Map of trial ID to config label
	pub params: HashMap<usize, String>,

	/// Result data collated across all trials
	pub data: DataFrame,

	/// Hall of fame, keyed by trial ID
	pub hof: HashMap<usize, Vec<u8>>,
}

/// Basic experiment suite for ablation studies, given a control configuration. This is used as the base problem and parameters for all configurations variants.
pub fn gen_basic(name: &str, control: ExperimentConfig, num_runs_per: usize) -> AblationExperiment {
	let configs = vec![
		control.relabel("control"), // NOTE: original label ignored
		control.relabel("no_speciation").no_speciation(),
		control.relabel("no_crossover").no_crossover(),
		control.relabel("no_elitism").no_elitism(),
		control.relabel("no_partials").no_partials(),
		control
			.relabel("no_partials_no_speciation")
			.no_partials()
			.no_speciation(),
		control
			.relabel("no_partials_no_crossover")
			.no_partials()
			.no_crossover(),
	];
	AblationExperiment::new(name, num_runs_per, configs)
}

/// Runs the ablation experiment with only the given configurations, designated by label.
/// Labels are: control, no_speciation, no_crossover, no_elitism, no_partials, no_partials_no_speciation, no_partials_no_crossover.
pub fn gen_custom(
	name: &str,
	control: ExperimentConfig,
	num_runs_per: usize,
	enabled_configs: &[&str],
) -> AblationExperiment {
	let mut out = gen_basic(name, control, num_runs_per);
	out.configurations.retain(|c| {
		enabled_configs
			.iter()
			.any(|label| label == &c.label.as_str())
	});
	out
}

impl Default for AblationExperimentResults {
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

impl ExperimentResults for AblationExperimentResults {
	type TrialResults = DefaultTrialResults;

	fn register(&mut self, trial_id: usize, label: String) {
		log::info!("Beginning trial {trial_id}/{label}");
		self.params.insert(trial_id, label);
	}

	fn collect(&mut self, trial: &Self::TrialResults) {
		// Add trial data to dataframe
		let mut trial_data = trial.to_data();
		let trial_id = trial.trial_id as u32;
		let ids = Series::new("trial_id".into(), vec![trial_id; trial_data.height()]);
		let label = self.params[&trial.trial_id].clone();
		let labels = Series::new("label".into(), vec![label.clone(); trial_data.height()]);
		trial_data.insert_column(0, ids);
		trial_data.insert_column(1, labels);
		#[cfg(debug_assertions)]
		{
			let trial_height = trial_data.height();
			let trial_width = trial_data.width();
			let pre_height = self.data.height();
			let pre_width = self.data.width();
			assert_eq!(
				trial_width, pre_width,
				"Vstacked data has different number of columns"
			);
			self.data.vstack_mut_owned_unchecked(trial_data);
			let post_height = self.data.height();
			let post_width = self.data.width();
			log::debug!("Vstacked {label}({trial_id}) data {trial_height}x{trial_width}\t({pre_height}x{pre_width} -> {post_height}x{post_width})");
		}
		#[cfg(not(debug_assertions))]
		{
			self.data.vstack_mut_owned_unchecked(trial_data);
		}

		// Add hof genome if present
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
		let Self {
			data,
			params: trials,
			hof,
		} = self;
		let mut data = data.clone();
		data.align_chunks_par(); // due to multiple vstacks

		// TODO: configure output, generate graphs
		let datafile = outdir.join("data.csv");
		let mut file = File::create(datafile).unwrap();
		CsvWriter::new(&mut file)
			.include_header(true)
			.finish(&mut data)
			.unwrap();

		// let fail_gens = self.base_params().num_generations as f64;
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
			]) // final generation stats
			.collect()
			.unwrap();
		log::info!("Overview:\n{gens_stats}");

		// write out hall of fame
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

		// TODO manifest file giving overall experiment info (ie. problem, time taken, date run, etc.)
	}
}
