//! Single-dummy HIC speed: one node's CFC-filtered head resultant, deferred-√
//! HIC vs the classic `powf` form. Read via the clean `read_states` API.
//! Run: `cargo run --release --example hic_bench -- <binout>`
use dynars::results::{injury, signal, Binout};
use std::hint::black_box;
use std::time::Instant;

/// Baseline: the textbook `max (t2−t1)·avg^2.5` with a scalar `powf` per window.
fn hic_powf(a: &[f64], dt: f64, window: f64) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut vel = vec![0.0; n];
    for i in 1..n {
        vel[i] = vel[i - 1] + 0.5 * (a[i] + a[i - 1]) * dt;
    }
    let w = ((window / dt).round() as usize).clamp(1, n - 1);
    let mut h = 0.0f64;
    for i in 0..n - 1 {
        for j in i + 1..=(i + w).min(n - 1) {
            let td = (j - i) as f64 * dt;
            let avg = (vel[j] - vel[i]) / td;
            if avg > 0.0 {
                h = h.max(td * avg.powf(2.5));
            }
        }
    }
    h
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/ryanosullivan/RustroverProjects/lassoBinout/src/binout".into());

    // ── read one dummy's channels via the clean state-aggregation API ───────
    let b = Binout::new(&path).expect("open binout");
    let xa = b.read_states("nodout", "x_acceleration").unwrap();
    let ya = b.read_states("nodout", "y_acceleration").unwrap();
    let za = b.read_states("nodout", "z_acceleration").unwrap();
    let nt = xa.n_steps;
    let dt = (xa.time[nt - 1] - xa.time[0]) / (nt as f64 - 1.0);

    // one dummy = one node's CFC-filtered head resultant
    let node = 0;
    let (x, y, z) = (xa.column(node), ya.column(node), za.column(node));
    let res: Vec<f64> = (0..nt)
        .map(|i| (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt() / 9810.0)
        .collect();
    let filt = signal::cfc(&res, 180.0, dt);

    // correctness: deferred-√ HIC equals the powf definition
    let (h_new, h_ref) = (injury::hic36(&filt, dt), hic_powf(&filt, dt, 0.036));
    assert!((h_new - h_ref).abs() <= 1e-9 * h_ref.max(1.0), "{h_new} vs {h_ref}");

    // ── time single-channel HIC (loop to get measurable ns/call) ────────────
    let reps = 20_000;
    let f = black_box(&filt);
    let t = Instant::now();
    let mut s = 0.0;
    for _ in 0..reps {
        s += hic_powf(black_box(f), dt, 0.036);
    }
    let powf_ns = t.elapsed().as_secs_f64() * 1e9 / reps as f64;

    let t = Instant::now();
    let mut s2 = 0.0;
    for _ in 0..reps {
        s2 += injury::hic36(black_box(f), dt);
    }
    let new_ns = t.elapsed().as_secs_f64() * 1e9 / reps as f64;
    black_box((s, s2));

    println!("single dummy: 1 channel, {nt} steps, dt={dt:.3e}");
    println!("  HIC36 powf        : {powf_ns:8.0} ns/call   1.0x");
    println!("  HIC36 deferred-√  : {new_ns:8.0} ns/call   {:.1}x", powf_ns / new_ns);
    println!("  HIC = {h_new:.6}");
}
