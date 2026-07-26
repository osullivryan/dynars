//! End-to-end binout in Rust: create arbitrary time-history curves, read them
//! back, and edit an existing file.
//!
//!     cargo run --release --example binout_demo

use dynars::results::{Binout, BinoutEditor, Data, ReadResult};

fn main() {
    let path = std::env::temp_dir().join("dynars_demo_binout");
    let p = path.to_str().unwrap();

    // --- 1. CREATE arbitrary curves --------------------------------------
    // A binout "curve" is a value per state (dNNNNNN dirs) + a sibling `time`.
    let mut e = BinoutEditor::new();
    let times: Vec<f64> = (0..12).map(|i| i as f64 * 0.1).collect();
    for (i, &t) in times.iter().enumerate() {
        let d = format!("d{:06}", i + 1);
        e.set(&["mycurve", &d, "time"], Data::F64(vec![t])).unwrap();
        e.set(
            &["mycurve", &d, "energy"],
            Data::F32(vec![(6.0 * t).sin() as f32]),
        )
        .unwrap();
    }
    e.set(
        &["mycurve", "metadata", "title"],
        Data::Str("custom curve".into()),
    )
    .unwrap();
    e.write(&path).unwrap();
    println!("wrote {}", path.display());

    // --- 2. READ it back -------------------------------------------------
    let b = Binout::new(p).unwrap();
    println!("top-level: {:?}", b.read(&[]).unwrap().keys());

    // Gather the curve across states (dNNNNNN dirs, sorted).
    let mut states: Vec<String> = b
        .read(&["mycurve"])
        .unwrap()
        .keys()
        .into_iter()
        .filter(|k| k.starts_with('d'))
        .collect();
    states.sort();
    let curve: Vec<f64> = states
        .iter()
        .map(|s| b.read_f64(&["mycurve", s.as_str(), "energy"]).unwrap()[0])
        .collect();
    println!("energy curve ({} points): {:?}", curve.len(), curve);

    // Read many channels in parallel (lock-free).
    let paths: Vec<Vec<&str>> = states
        .iter()
        .map(|s| vec!["mycurve", s.as_str(), "energy"])
        .collect();
    let results = b.read_many(&paths);
    let first = results[0].as_ref().unwrap();
    if let ReadResult::F32(v) = first {
        println!(
            "read_many: {} channels, first value {}",
            results.len(),
            v[0]
        );
    }

    // --- 3. EDIT an existing binout --------------------------------------
    let mut ed = BinoutEditor::open(p).unwrap();
    ed.set(&["mycurve", "d000001", "energy"], Data::F32(vec![999.0]))
        .unwrap();
    ed.write(&path).unwrap();
    let b2 = Binout::new(p).unwrap();
    println!(
        "after edit, d000001/energy = {:?}",
        b2.read_f64(&["mycurve", "d000001", "energy"]).unwrap()
    );

    std::fs::remove_file(&path).ok();
    println!("OK");
}
