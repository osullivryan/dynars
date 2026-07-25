use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use super::diskfile::Diskfile;
use super::LsdaError;

#[derive(Clone)]
pub struct Symbol {
    pub name: Vec<u8>,
    pub type_: u8,
    pub offset: u64,
    pub length: u64,
    pub file_index: Option<usize>,
    pub children: HashMap<Vec<u8>, Arc<Mutex<Symbol>>>,
    pub parent: Option<Arc<Mutex<Symbol>>>,
}

impl Symbol {
    pub fn new(name: Vec<u8>) -> Self {
        Self {
            name, type_: 0, offset: 0, length: 0,
            file_index: None,
            children: HashMap::new(),
            parent: None,
        }
    }

    pub fn add_child(&mut self, name: Vec<u8>, child: Arc<Mutex<Symbol>>) {
        self.children.insert(name, child);
        self.length = self.children.len() as u64;
    }

    /// Read this symbol's data directly from the memory-mapped file (no syscalls).
    pub fn lread(&self, files: &[Diskfile], start: usize, end: Option<usize>) -> Result<ReadResult, LsdaError> {
        if self.type_ == 0 {
            let mut keys: Vec<Vec<u8>> = self.children.keys().cloned().collect();
            keys.sort();
            return Ok(ReadResult::Directory(keys));
        }

        let file_index = self.file_index.ok_or_else(|| {
            LsdaError::SymbolNotFound("no file associated with symbol".into())
        })?;

        let file = files.get(file_index).ok_or_else(|| {
            LsdaError::SymbolNotFound("file index out of bounds".into())
        })?;

        let total = self.length as usize;
        let end = end.unwrap_or(total).min(total);
        let start = start.min(total);

        if start >= end {
            return Ok(empty_result(self.type_));
        }

        let elem_size = type_elem_size(self.type_);
        // Data starts at: symbol offset + record header (comp2 bytes) + name bytes
        let data_start = self.offset + file.comp2 as u64 + self.name.len() as u64;
        let byte_offset = (data_start + (start as u64 * elem_size as u64)) as usize;
        let byte_count = (end - start) * elem_size;

        let buf = file
            .bytes()
            .get(byte_offset..byte_offset + byte_count)
            .ok_or_else(|| LsdaError::Conversion("symbol data past end of file".into()))?;

        read_typed(buf, self.type_, end - start, file.is_little_endian)
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
    String(String),
}

impl ReadResult {
    /// Flatten to f64 for use as a scalar result.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            ReadResult::I8(v)  => v.iter().map(|x| *x as f64).collect(),
            ReadResult::I16(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::I32(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::I64(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U8(v)  => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U16(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U32(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::U64(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::F32(v) => v.iter().map(|x| *x as f64).collect(),
            ReadResult::F64(v) => v.clone(),
            ReadResult::Directory(_) | ReadResult::String(_) => vec![],
        }
    }

    pub fn keys(&self) -> Vec<String> {
        match self {
            ReadResult::Directory(keys) => keys.iter()
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
            v.push(if $le { <$ty>::from_le_bytes(bytes) } else { <$ty>::from_be_bytes(bytes) });
        }
        v
    }};
}

fn read_typed(buf: &[u8], type_: u8, count: usize, le: bool) -> Result<ReadResult, LsdaError> {
    Ok(match type_ {
        1 => ReadResult::I8(buf[..count.min(buf.len())].iter().map(|&b| b as i8).collect()),
        2 => ReadResult::I16(decode!(buf, count, le, i16, 2)),
        3 => ReadResult::I32(decode!(buf, count, le, i32, 4)),
        4 => ReadResult::I64(decode!(buf, count, le, i64, 8)),
        5 => ReadResult::U8(buf[..count.min(buf.len())].to_vec()),
        6 => ReadResult::U16(decode!(buf, count, le, u16, 2)),
        7 => ReadResult::U32(decode!(buf, count, le, u32, 4)),
        8 => ReadResult::U64(decode!(buf, count, le, u64, 8)),
        9 => ReadResult::F32(decode!(buf, count, le, f32, 4)),
        10 => ReadResult::F64(decode!(buf, count, le, f64, 8)),
        11 => ReadResult::String(String::from_utf8_lossy(buf).into_owned()),
        _ => ReadResult::U8(buf.to_vec()),
    })
}
