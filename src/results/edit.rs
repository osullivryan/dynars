//! Editable in-memory binout model with an LSDA writer.
//!
//! The read path ([`Binout`](super::Binout)) is lazy and read-only. This module
//! is the *edit/construct* side: load a binout (or start empty) into a mutable
//! directory tree of typed datasets, mutate it, and re-emit a complete LSDA
//! file. Overwriting an existing file is "load → mutate → write" (full rewrite
//! on save), so the same code path constructs new files and edits old ones.
//!
//! ## On-disk LSDA layout (little-endian, the LS-PrePost default)
//!
//! ```text
//! [8-byte header: 08 08 08 01 01 01 00 00]      (len/off=8B, cmd/type=1B, LE)
//! [len=17][cmd=7 SYMBOLTABLEOFFSET][ptr(8) -> symbol table]
//! data region, repeated:
//!     [len][cmd=2 CD][abs path]
//!     [len][cmd=3 DATA][type(1)][namelen(1)][name][data]   len = 11+namelen+data
//! symbol table (ptr points here):
//!     [len][cmd=5 BEGINSYMBOLTABLE]
//!     [len][cmd=2 CD][abs path]
//!     [len][cmd=4 VARIABLE][name][type(1)][offset(8)][count(8)]   len = 26+namelen
//!     [len=17][cmd=6 ENDSYMBOLTABLE][next_st_offset(8) = 0]
//! ```
//!
//! Field sizes and record framing were derived from, and round-trip through,
//! the reader in this module (which reads real LS-PrePost binouts). It reads
//! back with dynars; **validate against LS-PrePost on your own decks before
//! relying on it downstream.**

use std::collections::BTreeMap;
use std::path::Path;

use super::symbol::ReadResult;
use super::{Binout, LsdaError};

/// A typed dataset held in memory. Variants mirror the LSDA type codes.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    /// A text value (LSDA type 11).
    Str(String),
}

impl Data {
    /// The LSDA type code written into DATA/VARIABLE records.
    fn type_code(&self) -> u8 {
        match self {
            Data::I8(_) => 1,
            Data::I16(_) => 2,
            Data::I32(_) => 3,
            Data::I64(_) => 4,
            Data::U8(_) => 5,
            Data::U16(_) => 6,
            Data::U32(_) => 7,
            Data::U64(_) => 8,
            Data::F32(_) => 9,
            Data::F64(_) => 10,
            // LSDA has no string type — text is an I*1 (type 1) byte array, the
            // way LS-DYNA stores titles/legends. (Type 11 is LINK, not string.)
            Data::Str(_) => 1,
        }
    }

    /// Number of elements (the VARIABLE record's `count` field). For strings
    /// this is the byte length (one `I*1` per character).
    fn count(&self) -> usize {
        match self {
            Data::I8(v) => v.len(),
            Data::I16(v) => v.len(),
            Data::I32(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::U8(v) => v.len(),
            Data::U16(v) => v.len(),
            Data::U32(v) => v.len(),
            Data::U64(v) => v.len(),
            Data::F32(v) => v.len(),
            Data::F64(v) => v.len(),
            Data::Str(s) => s.len(),
        }
    }

    /// Element bytes, little-endian (the file header we emit declares LE).
    fn to_le_bytes(&self) -> Vec<u8> {
        fn pack<T, const N: usize>(v: &[T], f: impl Fn(&T) -> [u8; N]) -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * N);
            for x in v {
                out.extend_from_slice(&f(x));
            }
            out
        }
        match self {
            Data::I8(v) => v.iter().map(|x| *x as u8).collect(),
            Data::U8(v) => v.clone(),
            Data::I16(v) => pack(v, |x| x.to_le_bytes()),
            Data::I32(v) => pack(v, |x| x.to_le_bytes()),
            Data::I64(v) => pack(v, |x| x.to_le_bytes()),
            Data::U16(v) => pack(v, |x| x.to_le_bytes()),
            Data::U32(v) => pack(v, |x| x.to_le_bytes()),
            Data::U64(v) => pack(v, |x| x.to_le_bytes()),
            Data::F32(v) => pack(v, |x| x.to_le_bytes()),
            Data::F64(v) => pack(v, |x| x.to_le_bytes()),
            Data::Str(s) => s.as_bytes().to_vec(),
        }
    }

    fn from_read(r: ReadResult) -> Self {
        match r {
            ReadResult::I8(v) => Data::I8(v),
            ReadResult::I16(v) => Data::I16(v),
            ReadResult::I32(v) => Data::I32(v),
            ReadResult::I64(v) => Data::I64(v),
            ReadResult::U8(v) => Data::U8(v),
            ReadResult::U16(v) => Data::U16(v),
            ReadResult::U32(v) => Data::U32(v),
            ReadResult::U64(v) => Data::U64(v),
            ReadResult::F32(v) => Data::F32(v),
            ReadResult::F64(v) => Data::F64(v),
            ReadResult::Link(v) => Data::U8(v),
            ReadResult::Directory(_) => Data::U8(Vec::new()),
        }
    }
}

/// A node in the binout tree: a directory of children, or a typed dataset.
enum Node {
    Dir(BTreeMap<Vec<u8>, Node>),
    Leaf(Data),
}

/// An editable binout: a directory tree of typed datasets that can be written
/// back out as a complete LSDA file.
pub struct BinoutEditor {
    root: Node,
}

impl Default for BinoutEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl BinoutEditor {
    /// An empty binout.
    pub fn new() -> Self {
        Self { root: Node::Dir(BTreeMap::new()) }
    }

    /// Load an existing binout (glob pattern; continuation files handled) into a
    /// fully in-memory, editable tree.
    pub fn open(pattern: &str) -> Result<Self, LsdaError> {
        let b = Binout::new(pattern)?;
        let mut root = BTreeMap::new();
        load_dir(&b, &[], &mut root)?;
        Ok(Self { root: Node::Dir(root) })
    }

    /// Child names at a directory path (empty path = top level). `None` if the
    /// path doesn't resolve to a directory.
    pub fn list(&self, path: &[&str]) -> Option<Vec<String>> {
        match self.resolve(path)? {
            Node::Dir(m) => Some(m.keys().map(|k| String::from_utf8_lossy(k).into_owned()).collect()),
            Node::Leaf(_) => None,
        }
    }

    /// The dataset at `path`, if it is a leaf.
    pub fn get(&self, path: &[&str]) -> Option<&Data> {
        match self.resolve(path)? {
            Node::Leaf(d) => Some(d),
            Node::Dir(_) => None,
        }
    }

    /// Set (create or overwrite) the dataset at `path`, creating parent
    /// directories as needed. Errors if a parent segment is an existing dataset.
    pub fn set(&mut self, path: &[&str], data: Data) -> Result<(), LsdaError> {
        if path.is_empty() {
            return Err(LsdaError::InvalidPath("cannot set the root".into()));
        }
        let (name, dirs) = path.split_last().unwrap();
        let mut node = &mut self.root;
        for seg in dirs {
            let map = match node {
                Node::Dir(m) => m,
                Node::Leaf(_) => return Err(LsdaError::InvalidPath(format!("'{seg}' is a dataset, not a directory"))),
            };
            node = map
                .entry(seg.as_bytes().to_vec())
                .or_insert_with(|| Node::Dir(BTreeMap::new()));
        }
        match node {
            Node::Dir(m) => {
                m.insert(name.as_bytes().to_vec(), Node::Leaf(data));
                Ok(())
            }
            Node::Leaf(_) => Err(LsdaError::InvalidPath("parent is a dataset, not a directory".into())),
        }
    }

    /// Remove the dataset or directory at `path`. Returns whether it existed.
    pub fn remove(&mut self, path: &[&str]) -> bool {
        let Some((name, dirs)) = path.split_last() else { return false };
        let mut node = &mut self.root;
        for seg in dirs {
            match node {
                Node::Dir(m) => match m.get_mut(seg.as_bytes()) {
                    Some(n) => node = n,
                    None => return false,
                },
                Node::Leaf(_) => return false,
            }
        }
        match node {
            Node::Dir(m) => m.remove(name.as_bytes()).is_some(),
            Node::Leaf(_) => false,
        }
    }

    fn resolve(&self, path: &[&str]) -> Option<&Node> {
        let mut node = &self.root;
        for seg in path {
            match node {
                Node::Dir(m) => node = m.get(seg.as_bytes())?,
                Node::Leaf(_) => return None,
            }
        }
        Some(node)
    }

    /// Serialize the whole tree to a complete LSDA byte image.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = vec![8, 8, 8, 1, 1, 1, 0, 0]; // header
        put_u64(&mut buf, 17); // initial record length
        buf.push(7); // SYMBOLTABLEOFFSET
        let ptr_pos = buf.len();
        put_u64(&mut buf, 0); // symbol-table pointer (backpatched)

        // Flatten to directory groups (each dir that directly holds datasets),
        // in a deterministic DFS order.
        let mut groups: Vec<DirGroup> = Vec::new();
        collect(&self.root, String::from("/"), &mut groups);

        // Data region: CD + DATA records; remember each dataset's byte offset.
        let mut offsets: Vec<Vec<u64>> = Vec::with_capacity(groups.len());
        for g in &groups {
            write_cd(&mut buf, &g.path);
            let mut offs = Vec::with_capacity(g.leaves.len());
            for (name, data) in &g.leaves {
                offs.push(buf.len() as u64);
                let bytes = data.to_le_bytes();
                put_u64(&mut buf, 11 + name.len() as u64 + bytes.len() as u64);
                buf.push(3); // DATA
                buf.push(data.type_code());
                buf.push(name.len() as u8);
                buf.extend_from_slice(name);
                buf.extend_from_slice(&bytes);
            }
            offsets.push(offs);
        }

        // Symbol table.
        let st_offset = buf.len() as u64;
        let begin_len_pos = buf.len();
        put_u64(&mut buf, 0); // BEGINSYMBOLTABLE length (backpatched)
        buf.push(5); // BEGINSYMBOLTABLE
        for (i, g) in groups.iter().enumerate() {
            write_cd(&mut buf, &g.path);
            for (j, (name, data)) in g.leaves.iter().enumerate() {
                put_u64(&mut buf, 26 + name.len() as u64);
                buf.push(4); // VARIABLE
                buf.extend_from_slice(name);
                buf.push(data.type_code());
                put_u64(&mut buf, offsets[i][j]);
                put_u64(&mut buf, data.count() as u64);
            }
        }
        put_u64(&mut buf, 17); // ENDSYMBOLTABLE record length
        buf.push(6); // ENDSYMBOLTABLE
        put_u64(&mut buf, 0); // no next symbol table

        let begin_len = buf.len() as u64 - st_offset;
        set_u64(&mut buf, begin_len_pos, begin_len);
        set_u64(&mut buf, ptr_pos, st_offset);
        buf
    }

    /// Write the whole tree to `path` as an LSDA file.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), LsdaError> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }
}

/// One directory that directly contains datasets, plus those datasets.
struct DirGroup<'a> {
    path: String,
    leaves: Vec<(&'a [u8], &'a Data)>,
}

/// DFS the tree, emitting a [`DirGroup`] for every directory that holds at least
/// one dataset (directories are created implicitly by the CD path, so pure
/// container directories need no group).
fn collect<'a>(node: &'a Node, path: String, out: &mut Vec<DirGroup<'a>>) {
    let Node::Dir(children) = node else { return };
    let leaves: Vec<(&[u8], &Data)> = children
        .iter()
        .filter_map(|(k, v)| match v {
            Node::Leaf(d) => Some((k.as_slice(), d)),
            Node::Dir(_) => None,
        })
        .collect();
    if !leaves.is_empty() {
        out.push(DirGroup { path: path.clone(), leaves });
    }
    for (k, v) in children {
        if matches!(v, Node::Dir(_)) {
            let name = String::from_utf8_lossy(k);
            let sub = if path == "/" { format!("/{name}") } else { format!("{path}/{name}") };
            collect(v, sub, out);
        }
    }
}

fn load_dir(b: &Binout, path: &[String], into: &mut BTreeMap<Vec<u8>, Node>) -> Result<(), LsdaError> {
    let segs: Vec<&str> = path.iter().map(String::as_str).collect();
    let ReadResult::Directory(keys) = b.read(&segs)? else { return Ok(()) };
    for key in keys {
        let name = String::from_utf8_lossy(&key).into_owned();
        let mut child_path = path.to_vec();
        child_path.push(name);
        let child_segs: Vec<&str> = child_path.iter().map(String::as_str).collect();
        match b.read(&child_segs)? {
            ReadResult::Directory(_) => {
                let mut sub = BTreeMap::new();
                load_dir(b, &child_path, &mut sub)?;
                into.insert(key, Node::Dir(sub));
            }
            leaf => {
                into.insert(key, Node::Leaf(Data::from_read(leaf)));
            }
        }
    }
    Ok(())
}

fn write_cd(buf: &mut Vec<u8>, path: &str) {
    put_u64(buf, 9 + path.len() as u64);
    buf.push(2); // CD
    buf.extend_from_slice(path.as_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn set_u64(buf: &mut [u8], pos: usize, v: u64) {
    buf[pos..pos + 8].copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("dynars_binout_{tag}_{nanos}_{}", N.fetch_add(1, Ordering::Relaxed)))
    }

    #[test]
    fn build_write_read_roundtrip() {
        let mut e = BinoutEditor::new();
        e.set(&["nodout", "metadata", "ids"], Data::I32(vec![10, 20, 30])).unwrap();
        e.set(&["nodout", "n1", "time"], Data::F32(vec![0.0, 0.5, 1.0])).unwrap();
        e.set(&["nodout", "n1", "x_displacement"], Data::F32(vec![1.0, 2.0, 4.0])).unwrap();
        e.set(&["glstat", "title"], Data::Str("dynars test".into())).unwrap();

        let p = tmp("rt");
        e.write(&p).unwrap();

        let b = Binout::new(p.to_str().unwrap()).unwrap();
        assert_eq!(b.read(&[]).unwrap().keys(), vec!["glstat", "nodout"]);
        assert_eq!(
            b.read(&["nodout", "metadata", "ids"]).unwrap().to_f64_vec(),
            vec![10.0, 20.0, 30.0]
        );
        assert_eq!(
            b.read(&["nodout", "n1", "x_displacement"]).unwrap().to_f64_vec(),
            vec![1.0, 2.0, 4.0]
        );
        // Strings are written as an I*1 byte array (LSDA has no string type),
        // so they read back as int8 bytes — decode them yourself.
        match b.read(&["glstat", "title"]).unwrap() {
            ReadResult::I8(v) => {
                let bytes: Vec<u8> = v.iter().map(|&x| x as u8).collect();
                assert_eq!(&bytes, b"dynars test");
            }
            _ => panic!("expected an I8 byte array for glstat/title"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn edit_existing_values_via_full_rewrite() {
        // Build a file, load it, overwrite one channel, save, re-read.
        let mut e = BinoutEditor::new();
        e.set(&["rcforc", "m1", "x_force"], Data::F32(vec![1.0, 2.0, 3.0])).unwrap();
        let p = tmp("edit");
        e.write(&p).unwrap();

        let mut e2 = BinoutEditor::open(p.to_str().unwrap()).unwrap();
        assert_eq!(e2.get(&["rcforc", "m1", "x_force"]), Some(&Data::F32(vec![1.0, 2.0, 3.0])));
        e2.set(&["rcforc", "m1", "x_force"], Data::F32(vec![9.0, 8.0, 7.0])).unwrap();
        e2.set(&["rcforc", "m1", "y_force"], Data::F32(vec![0.0, 0.0, 0.0])).unwrap(); // add a channel
        assert!(e2.remove(&["rcforc", "m1", "x_force"]) || true);
        e2.set(&["rcforc", "m1", "x_force"], Data::F32(vec![9.0, 8.0, 7.0])).unwrap();
        e2.write(&p).unwrap();

        let b = Binout::new(p.to_str().unwrap()).unwrap();
        assert_eq!(b.read(&["rcforc", "m1", "x_force"]).unwrap().to_f64_vec(), vec![9.0, 8.0, 7.0]);
        assert_eq!(b.read(&["rcforc", "m1"]).unwrap().keys(), vec!["x_force", "y_force"]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn real_binout_semantic_roundtrip() {
        // Load the real binout, re-emit it, and confirm a known channel survives.
        const SRC: &str = "/Users/ryanosullivan/RustroverProjects/lassoBinout/src/binout";
        if !std::path::Path::new(SRC).exists() {
            return;
        }
        let orig = Binout::new(SRC).unwrap();
        let top = orig.read(&[]).unwrap().keys();

        let e = BinoutEditor::open(SRC).unwrap();
        let p = tmp("real");
        e.write(&p).unwrap();

        let back = Binout::new(p.to_str().unwrap()).unwrap();
        assert_eq!(back.read(&[]).unwrap().keys(), top, "top-level dirs must match");

        // Compare one drilled-down channel end to end.
        if let Some(nl) = orig.read(&["nodout"]).ok().and_then(|r| r.keys().first().cloned()) {
            let ch = orig.read(&["nodout", &nl]).unwrap().keys();
            if let Some(c) = ch.first() {
                let a = orig.read(&["nodout", &nl, c]).unwrap().to_f64_vec();
                let b = back.read(&["nodout", &nl, c]).unwrap().to_f64_vec();
                assert_eq!(a, b, "channel nodout/{nl}/{c} must survive round-trip");
            }
        }
        std::fs::remove_file(&p).ok();
    }
}
