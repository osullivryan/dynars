//! Convert LS-DYNA results into Apache Arrow `RecordBatch`es.
//!
//! This is the **seam** in front of the Parquet / Iceberg sinks. It lives in a
//! separate crate so the fast core never takes an arrow-rs dependency. The
//! shape mirrors `python/dynars/iceberg.py`: one long/tidy table per binout
//! branch,
//!
//! ```text
//! run_id | time | id | <var1> | <var2> | ...
//! ```
//!
//! where `id` is the per-entity index within a state (interface / part / …),
//! and scalar-per-state variables broadcast across entities.
//!
//! Each field carries a stable Iceberg field-id in its Arrow metadata
//! (`PARQUET:field_id`) so the emitted data is adoptable into an Iceberg table
//! without a rewrite.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use dynars::results::Binout;

/// One binout branch rendered as an Arrow table.
pub struct BranchTable {
    pub branch: String,
    pub batch: RecordBatch,
}

/// The Iceberg field-id metadata key Parquet readers/writers understand.
const FIELD_ID_KEY: &str = "PARQUET:field_id";

fn field(name: &str, ty: DataType, id: i32) -> Field {
    let mut md = HashMap::new();
    md.insert(FIELD_ID_KEY.to_string(), id.to_string());
    Field::new(name, ty, false).with_metadata(md)
}

fn is_state_dir(name: &str) -> bool {
    name.starts_with('d') && name.len() > 1 && name[1..].bytes().all(|b| b.is_ascii_digit())
}

/// Read one variable across every state dir and stack it into `(rows = T, cols = k)`
/// row-major. Returns `None` if the per-state widths are ragged.
fn stack_states(
    b: &Binout,
    branch: &str,
    states: &[String],
    var: &str,
) -> Option<(usize, Vec<f64>)> {
    let paths: Vec<Vec<&str>> = states
        .iter()
        .map(|s| vec![branch, s.as_str(), var])
        .collect();
    let rows: Vec<Vec<f64>> = b
        .read_many(&paths)
        .into_iter()
        .map(|r| r.map(|rr| rr.to_f64_vec()).unwrap_or_default())
        .collect();
    let k = rows.first().map(|r| r.len()).unwrap_or(0);
    if k == 0 || rows.iter().any(|r| r.len() != k) {
        return None;
    }
    let mut flat = Vec::with_capacity(rows.len() * k);
    for r in &rows {
        flat.extend_from_slice(r);
    }
    Some((k, flat))
}

/// Build the long-form table for a single branch, or `None` if it has no usable
/// time-varying variables.
fn branch_table(b: &Binout, branch: &str, run_id: &str) -> Option<RecordBatch> {
    let kids = b.channels(&[branch]).ok()?;
    let mut states: Vec<String> = kids.into_iter().filter(|k| is_state_dir(k)).collect();
    states.sort();
    if states.is_empty() {
        return None;
    }
    let t = states.len();

    let var_names: Vec<String> = b
        .channels(&[branch, &states[0]])
        .ok()?
        .into_iter()
        .filter(|v| v != "time")
        .collect();

    // Stack every variable; keep those that form a rectangular (T, k) block.
    let mut series: Vec<(String, usize, Vec<f64>)> = Vec::new();
    for v in &var_names {
        if let Some((k, flat)) = stack_states(b, branch, &states, v) {
            if flat.len() == t * k {
                series.push((v.clone(), k, flat));
            }
        }
    }
    if series.is_empty() {
        return None;
    }

    let n_ent = series.iter().map(|(_, k, _)| *k).max().unwrap();
    let n_rows = t * n_ent;

    // Shared axes.
    let time_vec: Vec<f64> = match stack_states(b, branch, &states, "time") {
        Some((_, flat)) => flat.into_iter().step_by(1).take(t).collect(),
        None => (0..t).map(|i| i as f64).collect(),
    };
    let time_col: Vec<f64> = time_vec
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, n_ent))
        .collect();
    let id_col: Vec<i64> = (0..t).flat_map(|_| 0..n_ent as i64).collect();
    let run_col: Vec<&str> = vec![run_id; n_rows];

    let mut fields: Vec<Field> = vec![
        field("run_id", DataType::Utf8, 1),
        field("time", DataType::Float64, 2),
        field("id", DataType::Int64, 3),
    ];
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(run_col)),
        Arc::new(Float64Array::from(time_col)),
        Arc::new(Int64Array::from(id_col)),
    ];

    let mut next_id = 4;
    for (name, k, flat) in &series {
        let col: Vec<f64> = if *k == n_ent {
            flat.clone()
        } else if *k == 1 {
            // Broadcast a scalar-per-state across all entities.
            flat.iter()
                .flat_map(|&x| std::iter::repeat_n(x, n_ent))
                .collect()
        } else {
            continue; // ragged relative to n_ent — skip rather than misalign
        };
        fields.push(field(&safe(name), DataType::Float64, next_id));
        cols.push(Arc::new(Float64Array::from(col)));
        next_id += 1;
    }

    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).ok()
}

/// Column-name-safe (Iceberg/SQL dislike spaces and `+`).
fn safe(name: &str) -> String {
    name.replace(' ', "_").replace('+', "_plus_")
}

/// Convert a whole binout into one Arrow table per branch, tagged with `run_id`.
pub fn binout_tables(
    path: &str,
    run_id: &str,
) -> Result<Vec<BranchTable>, dynars::results::LsdaError> {
    let b = Binout::new(path)?;
    let branches = b.read(&[])?.keys();
    let mut out = Vec::new();
    for branch in branches {
        if let Some(batch) = branch_table(&b, &branch, run_id) {
            out.push(BranchTable { branch, batch });
        }
    }
    Ok(out)
}
