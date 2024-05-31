use peas::{genetic::GenAlg, WasmGA, WasmGenome};
use rand::{thread_rng, Rng};

fn main() {
	println!("Hello, world!");

	let seed: u64 = thread_rng().gen();

	let mut wa = WasmGA::new(seed, 100, 50, 1).init_with(Default::default);
	wa.epoch();
}
