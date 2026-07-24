use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crossbeam::queue::SegQueue;
use dashmap::{DashMap, DashSet};

use crate::keyword::{IncludeKind, IncludeNode};
use crate::parser::parse_file_from_path;

struct WorkItem {
    id: usize,
    path: PathBuf,
    kind: Option<IncludeKind>,
}

struct ParsedEntry {
    path: PathBuf,
    byte_count: usize,
    kind: Option<IncludeKind>,
    child_ids: Vec<usize>,
}

pub fn build_include_tree(root_path: &Path) -> Result<IncludeNode, String> {
    let root_path = std::fs::canonicalize(root_path)
        .map_err(|e| format!("Cannot resolve root path {}: {}", root_path.display(), e))?;

    let queue: SegQueue<WorkItem> = SegQueue::new();
    let results: DashMap<usize, ParsedEntry> = DashMap::new();
    let visited: DashSet<PathBuf> = DashSet::new();
    let include_paths: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    let next_id = AtomicUsize::new(1);
    let in_flight = AtomicUsize::new(1);

    let root_id: usize = 0;
    visited.insert(root_path.clone());
    queue.push(WorkItem { id: root_id, path: root_path, kind: None });

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    crossbeam::scope(|s| {
        for _ in 0..num_threads {
            s.spawn(|_| {
                loop {
                    match queue.pop() {
                        Some(item) => {
                            let current_include_paths = include_paths.lock().unwrap().clone();
                            let result = parse_file_from_path(&item.path, &current_include_paths);

                            for inc in &result.includes {
                                match inc.kind {
                                    IncludeKind::IncludePath => {
                                        include_paths.lock().unwrap().push(inc.resolved_path.clone());
                                    }
                                    IncludeKind::IncludePathRelative => {
                                        let parent = item.path.parent().unwrap_or(Path::new("."));
                                        include_paths.lock().unwrap().push(parent.join(&inc.raw_path));
                                    }
                                    _ => {}
                                }
                            }

                            let mut child_ids = Vec::new();
                            for inc in &result.includes {
                                if matches!(inc.kind, IncludeKind::IncludePath | IncludeKind::IncludePathRelative) {
                                    continue;
                                }
                                if !visited.insert(inc.resolved_path.clone()) {
                                    continue;
                                }
                                let child_id = next_id.fetch_add(1, Ordering::Relaxed);
                                child_ids.push(child_id);
                                in_flight.fetch_add(1, Ordering::AcqRel);
                                queue.push(WorkItem {
                                    id: child_id,
                                    path: inc.resolved_path.clone(),
                                    kind: Some(inc.kind.clone()),
                                });
                            }

                            results.insert(item.id, ParsedEntry {
                                path: item.path,
                                byte_count: result.byte_count,
                                kind: item.kind,
                                child_ids,
                            });

                            if in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
                                // Last item done — workers will exit.
                            }
                        }
                        None => {
                            if in_flight.load(Ordering::Acquire) == 0 {
                                break;
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
            });
        }
    }).unwrap();

    Ok(build_tree_from_results(root_id, &results))
}

fn build_tree_from_results(id: usize, results: &DashMap<usize, ParsedEntry>) -> IncludeNode {
    match results.get(&id) {
        Some(entry) => {
            let children: Vec<IncludeNode> = entry
                .child_ids
                .iter()
                .map(|&child_id| build_tree_from_results(child_id, results))
                .collect();
            IncludeNode {
                path: entry.path.clone(),
                byte_count: entry.byte_count,
                kind: entry.kind.clone(),
                children,
            }
        }
        None => IncludeNode {
            path: PathBuf::from("<missing>"),
            byte_count: 0,
            kind: None,
            children: Vec::new(),
        },
    }
}
