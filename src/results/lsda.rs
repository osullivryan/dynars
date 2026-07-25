use std::collections::HashSet;
use std::io::SeekFrom;
use std::sync::{Arc, Mutex};
use super::diskfile::Diskfile;
use super::symbol::Symbol;
use super::LsdaError;

const BEGINSYMBOLTABLE: u8 = 5;

pub struct Lsda {
    pub files: Vec<Diskfile>,
    pub root: Arc<Mutex<Symbol>>,
    pub cwd: Arc<Mutex<Symbol>>,
    #[allow(dead_code)]
    pub mode: String,
}

impl Lsda {
    pub fn new(files: Vec<String>, mode: &str) -> Result<Self, LsdaError> {
        let mut disk_files = Vec::new();

        if mode.starts_with('r') {
            let mut names: HashSet<String> = HashSet::new();
            for file in &files {
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
            for f in &name_list {
                disk_files.push(Diskfile::new(f, mode)?);
            }
        } else {
            if files.len() > 1 {
                return Err(LsdaError::InvalidPath("only one file in write mode".into()));
            }
            disk_files.push(Diskfile::new(&files[0], mode)?);
        }

        let root = Arc::new(Mutex::new(Symbol::new(b"/".to_vec())));
        let cwd = Arc::clone(&root);

        let mut lsda = Self { files: disk_files, root: Arc::clone(&root), cwd, mode: mode.to_string() };

        if mode.starts_with('r') {
            for i in 0..lsda.files.len() {
                lsda.read_symbol_table(i)?;
            }
        }

        Ok(lsda)
    }

    fn cd_internal(&mut self, path: &str, create: bool) -> Result<(), LsdaError> {
        let path = if path.ends_with('/') && path.len() > 1 { &path[..path.len()-1] } else { path };
        if path == "/" { self.cwd = Arc::clone(&self.root); return Ok(()); }

        let (abs, parts_str) = if path.starts_with('/') {
            self.cwd = Arc::clone(&self.root);
            (true, path[1..].to_string())
        } else {
            (false, path.to_string())
        };
        let _ = abs;

        for part in parts_str.split('/').filter(|s| !s.is_empty()) {
            if part == ".." {
                let parent = { self.cwd.lock().unwrap().parent.clone() };
                if let Some(p) = parent { self.cwd = p; }
                continue;
            }
            let has_child = { self.cwd.lock().unwrap().children.contains_key(part.as_bytes()) };
            if has_child {
                let child = { self.cwd.lock().unwrap().children.get(part.as_bytes()).map(Arc::clone) };
                if let Some(c) = child {
                    let is_dir = c.lock().unwrap().type_ == 0;
                    if is_dir { self.cwd = c; } else { break; }
                }
            } else if create {
                let new_sym = Arc::new(Mutex::new(Symbol::new(part.as_bytes().to_vec())));
                new_sym.lock().unwrap().parent = Some(Arc::clone(&self.cwd));
                self.cwd.lock().unwrap().add_child(part.as_bytes().to_vec(), Arc::clone(&new_sym));
                self.cwd = new_sym;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn read_symbol_table(&mut self, fi: usize) -> Result<(), LsdaError> {
        let command_size = self.files[fi].command_size;
        let length_size  = self.files[fi].length_size;
        self.files[fi].at_eof = false;

        // The file starts with a fixed "write-offset" record at position 8 (right after the
        // 8-byte file header). Its layout is:
        //   [total_length (length_size)][command=7 (command_size)][pointer (offset_size)]
        // The outer loop reads the POINTER data with read_offset(), so we must position
        // the file at the start of the pointer field, skipping the length+command header.
        let ptr_start = 8u64 + length_size as u64 + command_size as u64;
        self.files[fi].seek(SeekFrom::Start(ptr_start))?;

        loop {
            self.files[fi].last_offset = self.files[fi].tell()?;
            let offset = self.files[fi].read_offset()?;
            if offset == 0 { return Ok(()); }
            self.files[fi].seek(SeekFrom::Start(offset))?;
            let (_, cmd) = self.files[fi].read_command()?;
            if cmd != BEGINSYMBOLTABLE { return Ok(()); }

            loop {
                let (clen, cmd) = self.files[fi].read_command()?;
                let data_len = clen.checked_sub(command_size as u64 + length_size as u64)
                    .ok_or_else(|| LsdaError::Conversion(format!(
                        "corrupt binout symbol table: record length {clen} is smaller \
                         than its {}-byte header", command_size as u64 + length_size as u64)))?;
                match cmd {
                    2 => {
                        let path_bytes = self.files[fi].read_bytes(data_len as usize)?;
                        let path = String::from_utf8_lossy(&path_bytes).to_string();
                        self.cd_internal(&path, true)?;
                    }
                    4 => { self.read_entry(fi, data_len as usize)?; }
                    _ => break,
                }
            }
        }
    }

    fn read_entry(&mut self, fi: usize, reclen: usize) -> Result<(), LsdaError> {
        let data = self.files[fi].read_bytes(reclen)?;
        let f = &self.files[fi];
        let RawEntry { name, type_, offset, length } =
            parse_entry(&data, f.comp1, f.type_size, f.offset_size, f.length_size,
                        f.is_little_endian)?;

        let sym = {
            let mut cwd = self.cwd.lock().unwrap();
            if let Some(existing) = cwd.children.get(&name) {
                Arc::clone(existing)
            } else {
                let s = Arc::new(Mutex::new(Symbol::new(name.clone())));
                s.lock().unwrap().parent = Some(Arc::clone(&self.cwd));
                cwd.add_child(name.clone(), Arc::clone(&s));
                s
            }
        };
        let mut s = sym.lock().unwrap();
        s.type_ = type_;
        s.offset = offset;
        s.length = length;
        s.file_index = Some(fi);
        Ok(())
    }
}

/// Decoded fields of one symbol-table entry record (command 4).
struct RawEntry {
    name:   Vec<u8>,
    type_:  u8,
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
    let corrupt = |what: &str| LsdaError::Conversion(format!(
        "corrupt binout symbol-table entry: {what} ({}-byte record)", data.len()));
    let n = data.len().checked_sub(comp1)
        .ok_or_else(|| corrupt("record shorter than its fixed fields"))?;
    let name = data[..n].to_vec();
    let type_ = *data.get(n).ok_or_else(|| corrupt("missing type byte"))?;
    let off_start = n + type_size as usize;
    let len_start = off_start + offset_size as usize;
    let offset = data.get(off_start..).and_then(|d| read_int(d, offset_size, le))
        .ok_or_else(|| corrupt("truncated offset field"))?;
    let length = data.get(len_start..).and_then(|d| read_int(d, length_size, le))
        .ok_or_else(|| corrupt("truncated length field"))?;
    Ok(RawEntry { name, type_, offset, length })
}

/// Read a `size`-byte integer from the head of `data`; `None` when the buffer
/// is too short or the size isn't one this format uses.
fn read_int(data: &[u8], size: u8, le: bool) -> Option<u64> {
    use byteorder::{ByteOrder, LittleEndian, BigEndian};
    if data.len() < size as usize { return None; }
    Some(match size {
        1 => data[0] as u64,
        2 => if le { LittleEndian::read_u16(data) as u64 } else { BigEndian::read_u16(data) as u64 },
        4 => if le { LittleEndian::read_u32(data) as u64 } else { BigEndian::read_u32(data) as u64 },
        8 => if le { LittleEndian::read_u64(data) } else { BigEndian::read_u64(data) },
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
