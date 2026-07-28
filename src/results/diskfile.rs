use std::fs::File;
use std::io::SeekFrom;
use std::path::Path;

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use memmap2::Mmap;

use super::LsdaError;

/// A memory-mapped LSDA file opened for reading. The whole file is mapped once;
/// the symbol-table walk and every data read work directly off the mapping — no
/// per-read `open`/`seek`/`read` syscalls.
pub struct Diskfile {
    map: Mmap,
    pos: usize,
    pub at_eof: bool,
    pub length_size: u8,
    pub offset_size: u8,
    pub command_size: u8,
    pub type_size: u8,
    pub is_little_endian: bool,
    pub comp1: usize,
    pub comp2: usize,
    pub last_offset: u64,
}

impl Diskfile {
    pub fn new(name: &str, mode: &str) -> Result<Self, LsdaError> {
        if !mode.starts_with('r') {
            return Err(LsdaError::InvalidPath("Diskfile is read-only".into()));
        }
        let file = File::open(Path::new(name))?;
        // SAFETY: the file is opened read-only and not mutated elsewhere for the
        // lifetime of this mapping.
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 8 {
            return Err(LsdaError::Conversion(
                "LSDA file shorter than its 8-byte header".into(),
            ));
        }
        let header = &map[..8];
        let length_size = header[1];
        let offset_size = header[2];
        let command_size = header[3];
        let type_size = header[4];
        let is_little_endian = header[5] == 1;
        // Data starts right after the header (or where header[0] points).
        let pos = if header[0] > 8 { header[0] as usize } else { 8 };

        let comp1 = type_size as usize + offset_size as usize + length_size as usize;
        let comp2 = length_size as usize + command_size as usize + type_size as usize + 1;

        Ok(Self {
            map,
            pos,
            at_eof: false,
            length_size,
            offset_size,
            command_size,
            type_size,
            is_little_endian,
            comp1,
            comp2,
            last_offset: 0,
        })
    }

    /// The whole mapped file, for direct (zero-copy) data reads.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn read_command(&mut self) -> Result<(u64, u8), LsdaError> {
        let length = self.read_value(self.length_size)?;
        let command = self.read_u8()?;
        Ok((length, command))
    }

    pub fn read_offset(&mut self) -> Result<u64, LsdaError> {
        self.read_value(self.offset_size)
    }

    /// Read `len` bytes as a borrowed slice into the mapping — no copy. Used by
    /// the symbol-table walk, which parses each record in place and only
    /// allocates the (small) names it keeps.
    pub fn read_slice(&mut self, len: usize) -> Result<&[u8], LsdaError> {
        let start = self.pos;
        let end = start
            .checked_add(len)
            .filter(|e| *e <= self.map.len())
            .ok_or_else(|| LsdaError::Conversion("read past end of LSDA file".into()))?;
        self.pos = end;
        Ok(&self.map[start..end])
    }

    pub fn tell(&mut self) -> Result<u64, LsdaError> {
        Ok(self.pos as u64)
    }

    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64, LsdaError> {
        let new = match pos {
            SeekFrom::Start(o) => o as i64,
            SeekFrom::Current(o) => self.pos as i64 + o,
            SeekFrom::End(o) => self.map.len() as i64 + o,
        };
        if new < 0 || new as usize > self.map.len() {
            return Err(LsdaError::Conversion(
                "seek out of range in LSDA file".into(),
            ));
        }
        self.pos = new as usize;
        Ok(self.pos as u64)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, LsdaError> {
        let b = *self
            .map
            .get(self.pos)
            .ok_or_else(|| LsdaError::Conversion("read past end of LSDA file".into()))?;
        self.pos += 1;
        Ok(b)
    }

    fn read_value(&mut self, size: u8) -> Result<u64, LsdaError> {
        let s = size as usize;
        let end = self.pos.checked_add(s).filter(|e| *e <= self.map.len());
        let end = end.ok_or_else(|| LsdaError::Conversion("read past end of LSDA file".into()))?;
        let b = &self.map[self.pos..end];
        let le = self.is_little_endian;
        let v = match size {
            1 => b[0] as u64,
            2 => {
                if le {
                    LittleEndian::read_u16(b) as u64
                } else {
                    BigEndian::read_u16(b) as u64
                }
            }
            4 => {
                if le {
                    LittleEndian::read_u32(b) as u64
                } else {
                    BigEndian::read_u32(b) as u64
                }
            }
            8 => {
                if le {
                    LittleEndian::read_u64(b)
                } else {
                    BigEndian::read_u64(b)
                }
            }
            _ => return Err(LsdaError::InvalidDataTypeSize),
        };
        self.pos = end;
        Ok(v)
    }
}
