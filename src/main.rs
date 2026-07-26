use std::path::{Path, PathBuf};
use std::time::Instant;

use dynars::{include, testgen};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "dynars",
    about = "High-performance LS-DYNA keyword file include tree parser"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate test LS-DYNA keyword files for benchmarking
    Generate {
        /// Include nesting depth
        #[arg(long, default_value_t = 6)]
        depth: usize,

        /// Number of includes per file at each level
        #[arg(long, default_value_t = 4)]
        breadth: usize,

        /// Number of *NODE lines per file
        #[arg(long, default_value_t = 100_000)]
        nodes: usize,

        /// Output directory
        #[arg(long, default_value = "test_output")]
        output: String,
    },

    /// Parse an LS-DYNA keyword file and build the include tree
    Parse {
        /// Path to the root keyword file
        file: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Generate {
            depth,
            breadth,
            nodes,
            output,
        } => {
            cmd_generate(depth, breadth, nodes, &output);
        }
        Command::Parse { file } => {
            cmd_parse(&file);
        }
    }
}

fn cmd_generate(depth: usize, breadth: usize, nodes: usize, output: &str) {
    println!("Generating test files...");
    println!(
        "  depth={}, breadth={}, nodes_per_file={}, output={}",
        depth, breadth, nodes, output
    );

    let start = Instant::now();
    testgen::generate_test_files(depth, breadth, nodes, output);
    let elapsed = start.elapsed();
    println!("Generation completed in {:.3}s", elapsed.as_secs_f64());
}

fn cmd_parse(file_path: &Path) {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("Parsing: {}", file_path.display());
    println!("Threads: {}", threads);
    println!();

    let start = Instant::now();
    let tree = include::build_include_tree(file_path);
    let elapsed = start.elapsed();

    match tree {
        Ok(root) => {
            let total_files = root.total_files();
            let total_bytes = root.total_bytes();

            println!("=== Include Tree ===");
            if total_files <= 200 {
                root.print_tree(0);
                println!();
            } else {
                println!("(Tree too large to print — {} files)", total_files);
                println!();
            }

            println!("=== Performance ===");
            println!("Total files:  {}", total_files);
            println!(
                "Total bytes:  {} ({:.2} MB)",
                total_bytes,
                total_bytes as f64 / 1_048_576.0
            );
            println!(
                "Parse time:   {:.6}s ({:.3}ms)",
                elapsed.as_secs_f64(),
                elapsed.as_secs_f64() * 1000.0
            );

            if elapsed.as_secs_f64() > 0.0 {
                let mb_per_sec = (total_bytes as f64 / 1_048_576.0) / elapsed.as_secs_f64();
                println!("Throughput:   {:.1} MB/s", mb_per_sec);
                println!(
                    "              {:.0} files/s",
                    total_files as f64 / elapsed.as_secs_f64()
                );
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
