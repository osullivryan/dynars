//! Benchmark the HIC hot loop: scalar vs SIMD (`hic_batch`) vs SIMD+rayon, on a
//! real binout. Run: `cargo run --release --example hic_bench -- <binout>`
use dynars::results::{injury, signal, Binout};
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/ryanosullivan/RustroverProjects/lassoBinout/src/binout".into());

    // ── read: assemble (nt × nn) from the per-timestep nodout dirs ──────────
    let t = Instant::now();
    let b = Binout::new(&path).expect("open binout");
    let mut dirs: Vec<String> = b
        .read(&["nodout"])
        .unwrap()
        .keys()
        .into_iter()
        .filter(|k| k.starts_with('d'))
        .collect();
    dirs.sort();
    let nt = dirs.len();
    let (mut xa, mut ya, mut za, mut time) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for d in &dirs {
        xa.extend(b.read_f64(&["nodout", d, "x_acceleration"]).unwrap());
        ya.extend(b.read_f64(&["nodout", d, "y_acceleration"]).unwrap());
        za.extend(b.read_f64(&["nodout", d, "z_acceleration"]).unwrap());
        time.push(b.read_f64(&["nodout", d, "time"]).unwrap()[0]);
    }
    let read = t.elapsed().as_secs_f64() * 1e3;
    let nc = xa.len() / nt;
    let dt = (time[nt - 1] - time[0]) / (nt as f64 - 1.0);

    // ── filtered resultant (nt × nc), one-time, scalar (not the thing timed) ─
    let mut filt = vec![0.0f64; nt * nc];
    let mut res = vec![0.0f64; nt];
    for j in 0..nc {
        for i in 0..nt {
            let k = i * nc + j;
            res[i] = (xa[k] * xa[k] + ya[k] * ya[k] + za[k] * za[k]).sqrt() / 9810.0;
        }
        let f = signal::cfc(&res, 180.0, dt);
        for i in 0..nt {
            filt[i * nc + j] = f[i];
        }
    }

    // ── HIC36: scalar per-channel (serial) ──────────────────────────────────
    let t = Instant::now();
    let hics_scalar: Vec<f64> = (0..nc)
        .map(|j| {
            let col: Vec<f64> = (0..nt).map(|i| filt[i * nc + j]).collect();
            injury::hic36(&col, dt)
        })
        .collect();
    let scalar = t.elapsed().as_secs_f64() * 1e3;

    // ── HIC36: SIMD across channels (hic_batch), serial ─────────────────────
    let t = Instant::now();
    let hics_simd = injury::hic_batch(&filt, nt, nc, dt, 0.036);
    let simd = t.elapsed().as_secs_f64() * 1e3;

    // ── HIC36: SIMD + rayon (both axes) ─────────────────────────────────────
    let stripe = (nc / (4 * rayon::current_num_threads()).max(1)).max(1) * 4;
    let t = Instant::now();
    let hics_both: Vec<f64> = (0..nc)
        .step_by(stripe)
        .collect::<Vec<_>>()
        .par_iter()
        .flat_map(|&j0| {
            let j1 = (j0 + stripe).min(nc);
            let sub_nc = j1 - j0;
            let mut sub = vec![0.0f64; nt * sub_nc];
            for i in 0..nt {
                sub[i * sub_nc..(i + 1) * sub_nc]
                    .copy_from_slice(&filt[i * nc + j0..i * nc + j1]);
            }
            injury::hic_batch(&sub, nt, sub_nc, dt, 0.036)
        })
        .collect();
    let both = t.elapsed().as_secs_f64() * 1e3;

    // correctness: all three agree
    let close = |a: &[f64], b: &[f64]| {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-9 * x.abs().max(1.0))
    };
    assert!(close(&hics_scalar, &hics_simd) && close(&hics_scalar, &hics_both));

    println!("nodes={nc} steps={nt} dt={dt:.3e}   read={read:.1} ms   threads={}",
        rayon::current_num_threads());
    println!("HIC36 over all {nc} channels:");
    println!("  scalar (powf)      : {scalar:8.2} ms   1.0x");
    println!("  SIMD (hic_batch)   : {simd:8.2} ms   {:.1}x", scalar / simd);
    println!("  SIMD + rayon       : {both:8.2} ms   {:.1}x", scalar / both);
    println!("  max HIC = {:.6}", hics_scalar.iter().cloned().fold(0.0, f64::max));
}
