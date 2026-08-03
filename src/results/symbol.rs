use super::LsdaError;
use super::diskfile::Diskfile;
use std::collections::BTreeMap;

/// An immutable, lock-free binout tree, built straight from the symbol table by
/// [`build_read_tree`](super::lsda::build_read_tree). Reads (including concurrent
/// ones) traverse it without any locking.
pub enum SymNode {
    Dir(BTreeMap<Vec<u8>, SymNode>),
    Leaf(SymMeta),
}

/// Location + type of one leaf dataset in the mapped file(s).
pub struct SymMeta {
    pub type_: u8,
    pub offset: u64,
    pub length: u64,
    pub file_index: usize,
    pub name_len: usize,
}

impl SymNode {
    /// Child node by name (only meaningful for directories).
    pub fn child(&self, seg: &[u8]) -> Option<&SymNode> {
        match self {
            SymNode::Dir(m) => m.get(seg),
            SymNode::Leaf(_) => None,
        }
    }

    /// Decode only the elements at `cols` of a leaf dataset into `out`, as f64,
    /// straight from the memory map — the targeted counterpart to [`read`](Self::read).
    ///
    /// Reads one entity's history (or a few) without materializing the whole
    /// per-state record: only the bytes of the requested elements are touched,
    /// so on a wide record the OS faults in ~one page per column instead of the
    /// entire array. Out-of-range columns yield `0.0` (a short/ragged record).
    pub fn read_cols_f64(
        &self,
        files: &[Diskfile],
        cols: &[usize],
        out: &mut Vec<f64>,
    ) -> Result<(), LsdaError> {
        match self {
            SymNode::Dir(_) => Err(LsdaError::Conversion(
                "expected a dataset, got a directory".into(),
            )),
            SymNode::Leaf(meta) => {
                let file = files
                    .get(meta.file_index)
                    .ok_or_else(|| LsdaError::SymbolNotFound("file index out of bounds".into()))?;
                let count = meta.length as usize;
                let elem = type_elem_size(meta.type_);
                let base = (meta.offset + file.comp2 as u64 + meta.name_len as u64) as usize;
                let bytes = file.bytes();
                for &j in cols {
                    let v = if j < count {
                        let at = base + j * elem;
                        bytes
                            .get(at..at + elem)
                            .map(|b| decode_one_f64(b, meta.type_, file.is_little_endian))
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    out.push(v);
                }
                Ok(())
            }
        }
    }

    /// Read this node: a directory yields its (sorted) child names; a leaf reads
    /// its data straight from the memory map (no syscalls, no locks).
    pub fn read(&self, files: &[Diskfile]) -> Result<ReadResult, LsdaError> {
        match self {
            SymNode::Dir(m) => Ok(ReadResult::Directory(m.keys().cloned().collect())),
            SymNode::Leaf(meta) => {
                let file = files
                    .get(meta.file_index)
                    .ok_or_else(|| LsdaError::SymbolNotFound("file index out of bounds".into()))?;
                let count = meta.length as usize;
                if count == 0 {
                    return Ok(empty_result(meta.type_));
                }
                let elem_size = type_elem_size(meta.type_);
                // Data starts at: offset + record header (comp2 bytes) + name bytes.
                let byte_offset = (meta.offset + file.comp2 as u64 + meta.name_len as u64) as usize;
                let byte_count = count * elem_size;
                let buf = file
                    .bytes()
                    .get(byte_offset..byte_offset + byte_count)
                    .ok_or_else(|| LsdaError::Conversion("symbol data past end of file".into()))?;
                read_typed(buf, meta.type_, count, file.is_little_endian)
            }
        }
    }
}

/// Result of reading a symbol from a binout file.
pub enum ReadResult {
    Directory(Vec<Vec<u8>>),
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
    /// A symbolic link (LSDA type 11), returned as its raw bytes. Rare.
    Link(Vec<u8>),
}

impl ReadResult {
    /// Flatten to f64 for use as a scalar result.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            ReadResult::I8(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::I16(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::I32(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::I64(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U8(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U16(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U32(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U64(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::F32(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::F64(v) => v.clone(),
            ReadResult::Directory(_) | ReadResult::Link(_) => vec![],
        }
    }

    pub fn keys(&self) -> Vec<String> {
        match self {
            ReadResult::Directory(keys) => keys
                .iter()
                .map(|k| String::from_utf8_lossy(k).into_owned())
                .collect(),
            _ => vec![],
        }
    }
}

fn type_elem_size(t: u8) -> usize {
    match t {
        1 | 5 | 11 => 1,
        2 | 6 => 2,
        3 | 7 | 9 => 4,
        4 | 8 | 10 => 8,
        _ => 1,
    }
}

fn empty_result(t: u8) -> ReadResult {
    match t {
        1 => ReadResult::I8(vec![]),
        2 => ReadResult::I16(vec![]),
        3 => ReadResult::I32(vec![]),
        4 => ReadResult::I64(vec![]),
        5 => ReadResult::U8(vec![]),
        6 => ReadResult::U16(vec![]),
        7 => ReadResult::U32(vec![]),
        8 => ReadResult::U64(vec![]),
        9 => ReadResult::F32(vec![]),
        10 => ReadResult::F64(vec![]),
        _ => ReadResult::U8(vec![]),
    }
}

/// Decode `count` little/big-endian scalars from `buf` into a `Vec<T>`. The
/// `chunks_exact` + `from_*_bytes` form vectorizes well and, on a
/// little-endian host reading LE data, is effectively a bulk copy.
macro_rules! decode {
    ($buf:expr, $count:expr, $le:expr, $ty:ty, $n:expr) => {{
        let mut v = Vec::with_capacity($count);
        for chunk in $buf.chunks_exact($n).take($count) {
            let bytes: [u8; $n] = chunk.try_into().unwrap();
            v.push(if $le {
                <$ty>::from_le_bytes(bytes)
            } else {
                <$ty>::from_be_bytes(bytes)
            });
        }
        v
    }};
}

/// Decode a single LSDA scalar (the type codes of [`read_typed`]) to f64. `buf`
/// must be at least the element's byte width; a short buffer yields `0.0`.
fn decode_one_f64(buf: &[u8], type_: u8, le: bool) -> f64 {
    macro_rules! d {
        ($ty:ty, $n:expr) => {{
            match buf.get(..$n).and_then(|s| <[u8; $n]>::try_from(s).ok()) {
                Some(b) => (if le {
                    <$ty>::from_le_bytes(b)
                } else {
                    <$ty>::from_be_bytes(b)
                }) as f64,
                None => 0.0,
            }
        }};
    }
    match type_ {
        1 => buf.first().map_or(0.0, |&b| b as i8 as f64),
        2 => d!(i16, 2),
        3 => d!(i32, 4),
        4 => d!(i64, 8),
        5 => buf.first().map_or(0.0, |&b| b as f64),
        6 => d!(u16, 2),
        7 => d!(u32, 4),
        8 => d!(u64, 8),
        9 => d!(f32, 4),
        10 => d!(f64, 8),
        _ => 0.0,
    }
}

fn read_typed(buf: &[u8], type_: u8, count: usize, le: bool) -> Result<ReadResult, LsdaError> {
    Ok(match type_ {
        1 => ReadResult::I8(
            buf[..count.min(buf.len())]
                .iter()
                .map(|&b| b as i8)
                .collect(),
        ),
        2 => ReadResult::I16(decode!(buf, count, le, i16, 2)),
        3 => ReadResult::I32(decode!(buf, count, le, i32, 4)),
        4 => ReadResult::I64(decode!(buf, count, le, i64, 8)),
        5 => ReadResult::U8(buf[..count.min(buf.len())].to_vec()),
        6 => ReadResult::U16(decode!(buf, count, le, u16, 2)),
        7 => ReadResult::U32(decode!(buf, count, le, u32, 4)),
        8 => ReadResult::U64(decode!(buf, count, le, u64, 8)),
        9 => ReadResult::F32(decode!(buf, count, le, f32, 4)),
        10 => ReadResult::F64(decode!(buf, count, le, f64, 8)),
        11 => ReadResult::Link(buf.to_vec()), // LSDA type 11 = LINK, not a string
        _ => ReadResult::U8(buf.to_vec()),
    })
}
