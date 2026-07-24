use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

static FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generate a large test file hierarchy for benchmarking.
///
/// Each file gets a globally unique name to avoid collisions in the DashMap
/// cycle detection. Files are all in the same flat directory for simplicity.
pub fn generate_test_files(depth: usize, breadth: usize, nodes_per_file: usize, output_dir: &str) {
    let out = Path::new(output_dir);
    // Clean previous run
    let _ = fs::remove_dir_all(out);
    fs::create_dir_all(out).expect("Failed to create output directory");

    FILE_COUNTER.store(0, Ordering::SeqCst);

    generate_file(out, "root.k", 0, depth, breadth, nodes_per_file);

    let total_files = FILE_COUNTER.load(Ordering::SeqCst);
    println!("Generated {} files in {}", total_files, output_dir);
}

fn generate_file(
    out_dir: &Path,
    filename: &str,
    current_depth: usize,
    max_depth: usize,
    breadth: usize,
    nodes_per_file: usize,
) {
    FILE_COUNTER.fetch_add(1, Ordering::Relaxed);

    let filepath = out_dir.join(filename);
    let file = fs::File::create(&filepath).expect("Failed to create file");
    let mut w = BufWriter::with_capacity(1024 * 1024, file);

    // Write header
    writeln!(w, "$# Generated test file: {}", filename).unwrap();
    writeln!(w, "$# Depth: {}, Breadth: {}", current_depth, breadth).unwrap();
    writeln!(w, "*KEYWORD").unwrap();

    // Write *NODE data (realistic 80-char fixed-width format)
    writeln!(w, "*NODE").unwrap();
    let base_id = FILE_COUNTER.load(Ordering::Relaxed) * 10_000_000;
    for i in 0..nodes_per_file {
        let nid = base_id + i + 1;
        let x = (i as f64) * 1.5;
        let y = (i as f64) * 2.5;
        let z = (i as f64) * 0.1;
        // 8 + 16 + 16 + 16 + 8 + 8 = 72 chars + 8 padding = 80
        write!(
            w,
            "{:>8}{:>16.6}{:>16.6}{:>16.6}{:>8}{:>8}        ",
            nid, x, y, z, 0, 0
        )
        .unwrap();
        writeln!(w).unwrap();
    }

    // If not at max depth, add includes to unique child files
    if current_depth < max_depth {
        for i in 0..breadth {
            // Unique child name using a counter
            let child_id = FILE_COUNTER.load(Ordering::Relaxed) + i * 1000 + current_depth * 100;
            let child_name = format!("d{}_b{}_{}.k", current_depth + 1, i, child_id);

            match i % 4 {
                0 => {
                    writeln!(w, "*INCLUDE").unwrap();
                    writeln!(w, "{}", child_name).unwrap();
                }
                1 => {
                    writeln!(w, "*INCLUDE_TRANSFORM").unwrap();
                    writeln!(w, "{}", child_name).unwrap();
                    // Transform data cards (our parser grabs filename, skips the rest
                    // because the next *KEYWORD or *INCLUDE terminates the block)
                }
                2 => {
                    writeln!(w, "*include").unwrap(); // lowercase test
                    writeln!(w, "{}", child_name).unwrap();
                }
                3 => {
                    writeln!(w, "$# Comment before include").unwrap();
                    writeln!(w, "*INCLUDE").unwrap();
                    writeln!(w, "$# Another comment").unwrap();
                    writeln!(w, "{}", child_name).unwrap();
                }
                _ => unreachable!(),
            }

            // Flush before recursion so file counter advances
            w.flush().unwrap();

            generate_file(out_dir, &child_name, current_depth + 1, max_depth, breadth, nodes_per_file);
        }
    }

    writeln!(w, "*END").unwrap();
    w.flush().unwrap();
}
