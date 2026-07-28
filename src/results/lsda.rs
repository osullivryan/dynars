use super::LsdaError;
use super::diskfile::Diskfile;
use super::symbol::{SymMeta, SymNode};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::SeekFrom;

const BEGINSYMBOLTABLE: u8 = 5;

/// Open a binout file family for reading: the globbed base file(s) plus their
/// `name%NNN` continuation files, memory-mapped and sorted so a file's index is
/// stable (it's what leaf metadata refers back to).
pub(crate) fn open_read_family(base_files: &[String]) -> Result<Vec<Diskfile>, LsdaError> {
    let mut names: HashSet<String> = HashSet::new();
    for file in base_files {
        names.insert(file.clone());
        for i in 1..1000 {
            let cont = format!("{}%{:03}", file, i);
            if std::path::Path::new(&cont).exists() {
                names.insert(cont);
            } else {
                break;
            }
        }
    }
    let mut name_list: Vec<String> = names.into_iter().collect();
    name_list.sort();
    name_list.iter().map(|f| Diskfile::new(f, "r")).collect()
}

/// One node while the symbol tree is being built: a flat index arena, so
/// navigation is plain integer indexing — no `Arc`, no `Mutex`, no per-node
/// locking. Directories have `type_ == 0` and populated `children`; leaves carry
/// their dataset's location. Converted to the immutable [`SymNode`] in one pass
/// at the end ([`build_symnode`]), which replaces the old build-then-freeze that
/// materialised the tree twice.
///
/// Children are a `HashMap` for O(1) lookup while parsing (branch directories can
/// hold hundreds of per-state children); [`build_symnode`] sorts them once into
/// the `BTreeMap` reads want.
struct RNode {
    type_: u8,
    offset: u64,
    length: u64,
    file_index: usize,
    name_len: usize,
    parent: usize,
    children: HashMap<Vec<u8>, usize>,
}

impl RNode {
    fn dir(parent: usize) -> Self {
        RNode {
            type_: 0,
            offset: 0,
            length: 0,
            file_index: 0,
            name_len: 0,
            parent,
            children: HashMap::new(),
        }
    }
}

/// Parse the LSDA symbol table of `files` straight into a lock-free [`SymNode`].
///
/// The symbol table is a chain of records: command 2 sets the current directory
/// (a path, created on demand), command 4 declares a leaf dataset in it. Later
/// records — and later files in the family — override earlier metadata for the
/// same path, so channels written across continuation files resolve to their
/// latest location.
pub(crate) fn build_read_tree(files: &mut [Diskfile]) -> Result<SymNode, LsdaError> {
    let mut arena: Vec<RNode> = vec![RNode::dir(0)]; // root at 0, its own parent
    let mut cwd = 0usize;
    for fi in 0..files.len() {
        read_symbol_table(&mut arena, &mut cwd, files, fi)?;
    }
    Ok(build_symnode(&arena, 0))
}

/// Navigate to `path` (absolute or relative to `cwd`), creating directory nodes
/// that don't exist yet — the read-mode symbol table always declares its dirs.
fn cd(arena: &mut Vec<RNode>, cwd: &mut usize, path: &str) {
    let path = if path.ends_with('/') && path.len() > 1 {
        &path[..path.len() - 1]
    } else {
        path
    };
    if path == "/" {
        *cwd = 0;
        return;
    }
    let rest = match path.strip_prefix('/') {
        Some(rest) => {
            *cwd = 0;
            rest
        }
        None => path,
    };
    for part in rest.split('/').filter(|s| !s.is_empty()) {
        if part == ".." {
            *cwd = arena[*cwd].parent;
            continue;
        }
        let key = part.as_bytes();
        match arena[*cwd].children.get(key).copied() {
            Some(ci) => {
                if arena[ci].type_ == 0 {
                    *cwd = ci; // descend into the directory
                } else {
                    break; // a leaf shadows the path — stop, matching the old walker
                }
            }
            None => {
                let idx = arena.len();
                arena.push(RNode::dir(*cwd));
                arena[*cwd].children.insert(key.to_vec(), idx);
                *cwd = idx;
            }
        }
    }
}

/// Declare (or update) a leaf dataset named `name` in directory `cwd`.
fn add_entry(arena: &mut Vec<RNode>, cwd: usize, entry: RawEntry, fi: usize) {
    let RawEntry {
        name,
        type_,
        offset,
        length,
    } = entry;
    let name_len = name.len();
    match arena[cwd].children.get(&name).copied() {
        Some(idx) => {
            let n = &mut arena[idx];
            n.type_ = type_;
            n.offset = offset;
            n.length = length;
            n.file_index = fi;
            n.name_len = name_len;
        }
        None => {
            let idx = arena.len();
            arena.push(RNode {
                type_,
                offset,
                length,
                file_index: fi,
                name_len,
                parent: cwd,
                children: HashMap::new(),
            });
            arena[cwd].children.insert(name, idx);
        }
    }
}

fn read_symbol_table(
    arena: &mut Vec<RNode>,
    cwd: &mut usize,
    files: &mut [Diskfile],
    fi: usize,
) -> Result<(), LsdaError> {
    let command_size = files[fi].command_size;
    let length_size = files[fi].length_size;
    files[fi].at_eof = false;

    // The file starts with a fixed "write-offset" record at position 8 (right
    // after the 8-byte header): [length][command=7][pointer]. We read the
    // POINTER with read_offset(), so position past the length+command header.
    let ptr_start = 8u64 + length_size as u64 + command_size as u64;
    files[fi].seek(SeekFrom::Start(ptr_start))?;

    loop {
        files[fi].last_offset = files[fi].tell()?;
        let offset = files[fi].read_offset()?;
        if offset == 0 {
            return Ok(());
        }
        files[fi].seek(SeekFrom::Start(offset))?;
        let (_, cmd) = files[fi].read_command()?;
        if cmd != BEGINSYMBOLTABLE {
            return Ok(());
        }

        loop {
            let (clen, cmd) = files[fi].read_command()?;
            let data_len = clen
                .checked_sub(command_size as u64 + length_size as u64)
                .ok_or_else(|| {
                    LsdaError::Conversion(format!(
                        "corrupt binout symbol table: record length {clen} is smaller \
                         than its {}-byte header",
                        command_size as u64 + length_size as u64
                    ))
                })?;
            match cmd {
                2 => {
                    let path_bytes = files[fi].read_slice(data_len as usize)?;
                    let path = String::from_utf8_lossy(path_bytes);
                    cd(arena, cwd, &path);
                }
                4 => {
                    // Copy the small fixed sizes out first so the record read can
                    // borrow the mapping (zero-copy) without an overlapping borrow.
                    let f = &files[fi];
                    let (comp1, type_size, offset_size, length_size, le) = (
                        f.comp1,
                        f.type_size,
                        f.offset_size,
                        f.length_size,
                        f.is_little_endian,
                    );
                    let data = files[fi].read_slice(data_len as usize)?;
                    let entry = parse_entry(data, comp1, type_size, offset_size, length_size, le)?;
                    add_entry(arena, *cwd, entry, fi);
                }
                _ => break,
            }
        }
    }
}

/// Convert the finished arena into the immutable, lock-free [`SymNode`] reads
/// traverse. `children` is already a `BTreeMap`, so directory listings come out
/// sorted with no extra work.
fn build_symnode(arena: &[RNode], idx: usize) -> SymNode {
    let n = &arena[idx];
    if n.type_ == 0 {
        let mut m = BTreeMap::new();
        for (name, &ci) in &n.children {
            m.insert(name.clone(), build_symnode(arena, ci));
        }
        SymNode::Dir(m)
    } else {
        SymNode::Leaf(SymMeta {
            type_: n.type_,
            offset: n.offset,
            length: n.length,
            file_index: n.file_index,
            name_len: n.name_len,
        })
    }
}

/// Decoded fields of one symbol-table entry record (command 4).
struct RawEntry {
    name: Vec<u8>,
    type_: u8,
    offset: u64,
    length: u64,
}

/// Parse one entry-record body: `[name][type][offset][length]`, where `comp1`
/// is the fixed size of the trailing fields. Truncated or corrupt records
/// (as produced by interrupted LS-DYNA runs) are an error, never a panic.
fn parse_entry(
    data: &[u8],
    comp1: usize,
    type_size: u8,
    offset_size: u8,
    length_size: u8,
    le: bool,
) -> Result<RawEntry, LsdaError> {
    let corrupt = |what: &str| {
        LsdaError::Conversion(format!(
            "corrupt binout symbol-table entry: {what} ({}-byte record)",
            data.len()
        ))
    };
    let n = data
        .len()
        .checked_sub(comp1)
        .ok_or_else(|| corrupt("record shorter than its fixed fields"))?;
    let name = data[..n].to_vec();
    let type_ = *data.get(n).ok_or_else(|| corrupt("missing type byte"))?;
    let off_start = n + type_size as usize;
    let len_start = off_start + offset_size as usize;
    let offset = data
        .get(off_start..)
        .and_then(|d| read_int(d, offset_size, le))
        .ok_or_else(|| corrupt("truncated offset field"))?;
    let length = data
        .get(len_start..)
        .and_then(|d| read_int(d, length_size, le))
        .ok_or_else(|| corrupt("truncated length field"))?;
    Ok(RawEntry {
        name,
        type_,
        offset,
        length,
    })
}

/// Read a `size`-byte integer from the head of `data`; `None` when the buffer
/// is too short or the size isn't one this format uses.
fn read_int(data: &[u8], size: u8, le: bool) -> Option<u64> {
    use byteorder::{BigEndian, ByteOrder, LittleEndian};
    if data.len() < size as usize {
        return None;
    }
    Some(match size {
        1 => data[0] as u64,
        2 => {
            if le {
                LittleEndian::read_u16(data) as u64
            } else {
                BigEndian::read_u16(data) as u64
            }
        }
        4 => {
            if le {
                LittleEndian::read_u32(data) as u64
            } else {
                BigEndian::read_u32(data) as u64
            }
        }
        8 => {
            if le {
                LittleEndian::read_u64(data)
            } else {
                BigEndian::read_u64(data)
            }
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A well-formed entry: name "x", type 1, 8-byte offset + 8-byte length
    // (comp1 = type + offset + length = 17).
    fn good_entry() -> Vec<u8> {
        let mut d = b"x".to_vec();
        d.push(1); // type
        d.extend_from_slice(&42u64.to_le_bytes());
        d.extend_from_slice(&7u64.to_le_bytes());
        d
    }

    #[test]
    fn parse_entry_reads_well_formed_record() {
        let e = parse_entry(&good_entry(), 17, 1, 8, 8, true).unwrap();
        assert_eq!(e.name, b"x");
        assert_eq!(e.type_, 1);
        assert_eq!(e.offset, 42);
        assert_eq!(e.length, 7);
    }

    #[test]
    fn parse_entry_rejects_truncated_records_without_panicking() {
        // Record shorter than the fixed fields: previously a usize underflow.
        assert!(parse_entry(b"ab", 17, 1, 8, 8, true).is_err());
        // Empty record.
        assert!(parse_entry(&[], 17, 1, 8, 8, true).is_err());
        // Truncated tail: previously an out-of-bounds slice panic.
        let mut d = good_entry();
        d.truncate(d.len() - 4);
        assert!(parse_entry(&d, 13, 1, 8, 8, true).is_err());
        // Nonsense field size never panics either.
        assert!(read_int(&[0; 8], 3, true).is_none());
        assert!(read_int(&[0; 2], 4, true).is_none());
    }
}
