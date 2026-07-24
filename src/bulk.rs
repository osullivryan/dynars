//! Phase 3: columnar bulk parsers for the high-volume keywords.
//!
//! `*NODE` and `*ELEMENT_*` account for essentially all of a large deck's
//! bytes. Rather than materialize millions of heap structs, these parsers
//! produce struct-of-arrays (parallel `Vec`s) that map directly onto numpy
//! arrays on the Python side. The generic tokenizer (Phase 2) still works on
//! these blocks, but this is the path you want for anything large.

use rayon::prelude::*;

use crate::keyword::{Block, CardFormat, ParsedFile};
use crate::parser::Field;

/// Fixed-format `*NODE` column widths: I8 id, three E16 coords, I8 tc, I8 rc.
/// Node coordinates are 16 wide, so the generic 8-column splitter cannot read
/// them — node cards need these explicit widths.
const NODE_WIDTHS_FIXED: [usize; 6] = [8, 16, 16, 16, 8, 8];
/// Long-format `*NODE` widths (all fields doubled to 20).
const NODE_WIDTHS_LONG: [usize; 6] = [20, 20, 20, 20, 20, 20];

/// Minimum bytes per parallel chunk. Below this, chunking overhead outweighs
/// the win, so a block stays a single chunk.
const MIN_CHUNK: usize = 256 * 1024;

/// Split a block body into line-aligned chunks for parallel parsing. Each
/// chunk contains only whole lines; concatenating the chunks' output preserves
/// file order.
fn line_chunks(body: &[u8], max_chunks: usize) -> Vec<&[u8]> {
    if body.is_empty() {
        return Vec::new();
    }
    let n = (body.len() / MIN_CHUNK).clamp(1, max_chunks.max(1));
    if n <= 1 {
        return vec![body];
    }
    let mut bounds = Vec::with_capacity(n + 1);
    bounds.push(0usize);
    for i in 1..n {
        let target = body.len() * i / n;
        let cut = match memchr::memchr(b'\n', &body[target..]) {
            Some(off) => target + off + 1, // start of the next line
            None => body.len(),
        };
        if cut > *bounds.last().unwrap() && cut < body.len() {
            bounds.push(cut);
        }
    }
    bounds.push(body.len());
    bounds.windows(2).map(|w| &body[w[0]..w[1]]).collect()
}

/// True for comment (`$`) and blank lines, which carry no card data.
#[inline]
fn is_skippable(line: &[u8]) -> bool {
    let indent = line.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
    line.is_empty()
        || line.get(indent) == Some(&b'$')
        || strip_eol(&line[indent..]).is_empty()
}

/// Collect line-aligned chunks (with their format) across every block whose
/// name matches `keyword`. Handles both the one-huge-block and
/// many-small-blocks cases: a single 5M-node block fans out into many chunks,
/// while many small blocks each contribute one.
fn collect_chunks<'a>(parsed: &'a ParsedFile, keyword: &str) -> Vec<(&'a [u8], CardFormat)> {
    let max_chunks = rayon::current_num_threads() * 4;
    let mut chunks = Vec::new();
    for block in &parsed.blocks {
        if !parsed.keyword_name(block).eq_ignore_ascii_case(keyword) {
            continue;
        }
        for c in line_chunks(parsed.body(block), max_chunks) {
            chunks.push((c, block.format));
        }
    }
    chunks
}

/// Split one `*NODE` data line into its up-to-6 fields, honouring format:
/// comma-delimited for free format, otherwise the node-specific column widths.
fn node_columns(line: &[u8], format: CardFormat) -> Vec<Field<'_>> {
    let line = strip_eol(line);
    if format == CardFormat::Free || memchr::memchr(b',', line).is_some() {
        return line.split(|&c| c == b',').map(|raw| Field { raw }).collect();
    }
    let widths = if format == CardFormat::Long {
        &NODE_WIDTHS_LONG
    } else {
        &NODE_WIDTHS_FIXED
    };
    let mut fields = Vec::with_capacity(6);
    let mut i = 0;
    for &w in widths {
        if i >= line.len() {
            break;
        }
        let end = (i + w).min(line.len());
        fields.push(Field { raw: &line[i..end] });
        i = end;
    }
    fields
}

#[inline]
fn strip_eol(s: &[u8]) -> &[u8] {
    let mut end = s.len();
    while end > 0 && matches!(s[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    &s[..end]
}

/// Iterate the data lines of a `*NODE` block, skipping `$`-comment and blank
/// lines, yielding node-column fields per line.
fn node_lines<'a>(
    parsed: &'a ParsedFile,
    block: &Block,
) -> impl Iterator<Item = Vec<Field<'a>>> {
    let body = parsed.body(block);
    let format = block.format;
    body.split(|&c| c == b'\n').filter_map(move |line| {
        let indent = line.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
        if line.is_empty()
            || line.get(indent) == Some(&b'$')
            || strip_eol(&line[indent..]).is_empty()
        {
            return None;
        }
        Some(node_columns(line, format))
    })
}

/// `*NODE` data as parallel arrays. `coords` is row-major `N x 3`.
#[derive(Debug, Default)]
pub struct NodeArrays {
    pub ids: Vec<i64>,
    pub coords: Vec<f64>,
}

impl NodeArrays {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Element connectivity as parallel arrays. `nodes` is row-major
/// `N x nodes_per_elem` (4 for shells, 8 for solids).
#[derive(Debug, Default)]
pub struct ElementArrays {
    pub eids: Vec<i64>,
    pub pids: Vec<i64>,
    pub nodes: Vec<i64>,
    pub nodes_per_elem: usize,
}

impl ElementArrays {
    pub fn len(&self) -> usize {
        self.eids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.eids.is_empty()
    }
}

/// Parse a single `*NODE` data line into (id, x, y, z) without allocating,
/// honouring free vs fixed/long column widths. Returns `None` for lines with
/// no readable id.
#[inline]
fn parse_node_line(line: &[u8], format: CardFormat) -> Option<(i64, f64, f64, f64)> {
    let line = strip_eol(line);
    if format == CardFormat::Free || memchr::memchr(b',', line).is_some() {
        let mut it = line.split(|&c| c == b',');
        let id = Field { raw: it.next()? }.as_i64()?;
        let x = it.next().map_or(0.0, |r| Field { raw: r }.as_f64().unwrap_or(0.0));
        let y = it.next().map_or(0.0, |r| Field { raw: r }.as_f64().unwrap_or(0.0));
        let z = it.next().map_or(0.0, |r| Field { raw: r }.as_f64().unwrap_or(0.0));
        return Some((id, x, y, z));
    }
    let w = if format == CardFormat::Long { &NODE_WIDTHS_LONG } else { &NODE_WIDTHS_FIXED };
    let id = Field { raw: col(line, w, 0) }.as_i64()?;
    let x = Field { raw: col(line, w, 1) }.as_f64().unwrap_or(0.0);
    let y = Field { raw: col(line, w, 2) }.as_f64().unwrap_or(0.0);
    let z = Field { raw: col(line, w, 3) }.as_f64().unwrap_or(0.0);
    Some((id, x, y, z))
}

/// The k-th fixed-width column of a line given cumulative field widths.
#[inline]
fn col<'a>(line: &'a [u8], widths: &[usize], k: usize) -> &'a [u8] {
    let start: usize = widths[..k].iter().sum();
    if start >= line.len() {
        return &[];
    }
    let end = (start + widths[k]).min(line.len());
    &line[start..end]
}

/// Collect all `*NODE` blocks into parallel id/coordinate arrays, parsing
/// chunks across all cores.
///
/// Single pass per chunk into local vecs, then concatenated. A two-pass
/// variant (count, allocate once, write into disjoint slices) was measured
/// *slower* — the counting scan over the data costs more than the concat copy
/// it removes.
pub fn parse_nodes(parsed: &ParsedFile) -> NodeArrays {
    let chunks = collect_chunks(parsed, "NODE");
    let partials: Vec<(Vec<i64>, Vec<f64>)> = chunks
        .par_iter()
        .map(|(chunk, format)| {
            let mut ids = Vec::new();
            let mut coords = Vec::new();
            for line in chunk.split(|&c| c == b'\n') {
                if is_skippable(line) {
                    continue;
                }
                if let Some((id, x, y, z)) = parse_node_line(line, *format) {
                    ids.push(id);
                    coords.push(x);
                    coords.push(y);
                    coords.push(z);
                }
            }
            (ids, coords)
        })
        .collect();

    let total: usize = partials.iter().map(|(ids, _)| ids.len()).sum();
    let mut out = NodeArrays {
        ids: Vec::with_capacity(total),
        coords: Vec::with_capacity(total * 3),
    };
    for (ids, coords) in partials {
        out.ids.extend(ids);
        out.coords.extend(coords);
    }
    out
}

/// Collect `*ELEMENT_SHELL` blocks (eid, pid, 4 corner nodes).
pub fn parse_element_shell(parsed: &ParsedFile) -> ElementArrays {
    parse_elements(parsed, "ELEMENT_SHELL", 4)
}

/// Collect `*ELEMENT_SOLID` blocks (eid, pid, 8 corner nodes).
pub fn parse_element_solid(parsed: &ParsedFile) -> ElementArrays {
    parse_elements(parsed, "ELEMENT_SOLID", 8)
}

/// Parse a single element data line: eid, pid, and `npe` node ids appended to
/// `nodes`. Element fields are integer columns (I8 fixed / I16 long), or
/// comma-separated in free format. Returns `None` (pushing nothing) for lines
/// with no readable eid.
#[inline]
fn parse_element_line(
    line: &[u8],
    format: CardFormat,
    npe: usize,
    nodes: &mut Vec<i64>,
) -> Option<(i64, i64)> {
    let line = strip_eol(line);
    // eid + pid + up to 8 node columns.
    let mut buf: [&[u8]; 10] = [&[]; 10];
    let cap = (2 + npe).min(buf.len());
    let mut n = 0;
    if format == CardFormat::Free || memchr::memchr(b',', line).is_some() {
        for raw in line.split(|&c| c == b',') {
            if n >= cap {
                break;
            }
            buf[n] = raw;
            n += 1;
        }
    } else {
        let w = if format == CardFormat::Long { 16 } else { 8 };
        let mut i = 0;
        while i < line.len() && n < cap {
            let end = (i + w).min(line.len());
            buf[n] = &line[i..end];
            n += 1;
            i = end;
        }
    }

    let eid = Field { raw: buf[0] }.as_i64()?;
    let pid = Field { raw: buf[1] }.as_i64().unwrap_or(0);
    for k in 0..npe {
        nodes.push(Field { raw: buf[2 + k] }.as_i64().unwrap_or(0));
    }
    Some((eid, pid))
}

fn parse_elements(parsed: &ParsedFile, keyword: &str, nodes_per_elem: usize) -> ElementArrays {
    let chunks = collect_chunks(parsed, keyword);
    let partials: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = chunks
        .par_iter()
        .map(|(chunk, format)| {
            let mut eids = Vec::new();
            let mut pids = Vec::new();
            let mut nodes = Vec::new();
            for line in chunk.split(|&c| c == b'\n') {
                if is_skippable(line) {
                    continue;
                }
                if let Some((eid, pid)) =
                    parse_element_line(line, *format, nodes_per_elem, &mut nodes)
                {
                    eids.push(eid);
                    pids.push(pid);
                }
            }
            (eids, pids, nodes)
        })
        .collect();

    let total: usize = partials.iter().map(|(e, _, _)| e.len()).sum();
    let mut out = ElementArrays {
        eids: Vec::with_capacity(total),
        pids: Vec::with_capacity(total),
        nodes: Vec::with_capacity(total * nodes_per_elem),
        nodes_per_elem,
    };
    for (eids, pids, nodes) in partials {
        out.eids.extend(eids);
        out.pids.extend(pids);
        out.nodes.extend(nodes);
    }
    out
}

/// Rewrite every `*NODE` block from a new coordinate array (row-major `N x 3`),
/// preserving each node's id and its translational/rotational constraint
/// columns. Marks the affected blocks dirty so `to_bytes()` emits the update
/// while leaving all other blocks byte-for-byte untouched.
///
/// Editing a block reformats it to canonical fixed-width columns; any comments
/// interleaved in the original node data are not preserved (this is the
/// documented cost of editing a block, versus verbatim pass-through of
/// untouched ones).
pub fn update_node_coords(parsed: &mut ParsedFile, new_coords: &[f64]) -> Result<(), String> {
    let node_block_indices: Vec<usize> = parsed
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| parsed.keyword_name(b).eq_ignore_ascii_case("NODE"))
        .map(|(i, _)| i)
        .collect();

    let total: usize = node_block_indices
        .iter()
        .map(|&i| node_lines(parsed, &parsed.blocks[i]).count())
        .sum();

    if new_coords.len() != total * 3 {
        return Err(format!(
            "coordinate array length {} does not match {} nodes (expected {})",
            new_coords.len(),
            total,
            total * 3
        ));
    }

    let mut gi = 0usize;
    for bi in node_block_indices {
        let block = parsed.blocks[bi].clone();
        let mut out = Vec::new();
        out.extend_from_slice(parsed.trivia(&block));
        out.extend_from_slice(parsed.name_line(&block));
        for card in node_lines(parsed, &block) {
            let id = card.first().and_then(|f| f.as_i64()).unwrap_or(0);
            let tc = card.get(4).and_then(|f| f.as_i64()).unwrap_or(0);
            let rc = card.get(5).and_then(|f| f.as_i64()).unwrap_or(0);
            let x = new_coords[gi * 3];
            let y = new_coords[gi * 3 + 1];
            let z = new_coords[gi * 3 + 2];
            gi += 1;
            out.extend_from_slice(fmt_node_line(id, x, y, z, tc, rc).as_bytes());
        }
        parsed.set_block_bytes(bi, out);
    }
    Ok(())
}

/// Format a node card in fixed-width columns: I8 id, three F16.9 coords, I8
/// tc, I8 rc.
fn fmt_node_line(id: i64, x: f64, y: f64, z: f64, tc: i64, rc: i64) -> String {
    format!(
        "{:>8}{:16.9}{:16.9}{:16.9}{:>8}{:>8}\n",
        id, x, y, z, tc, rc
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword::ParsedFile;
    use crate::parser::split_blocks;

    fn parsed(src: &[u8]) -> ParsedFile {
        ParsedFile::new("deck.k".into(), src.to_vec(), split_blocks(src))
    }

    #[test]
    fn parses_nodes_free_and_fixed() {
        // Free format nodes across two *NODE blocks.
        let src = b"*KEYWORD\n\
                    *NODE\n\
                    1,0.0,0.0,0.0\n\
                    2,1.0,2.0,3.0\n\
                    *NODE\n\
                    3,4.0,5.0,6.0\n\
                    *END\n";
        let p = parsed(src);
        let n = parse_nodes(&p);
        assert_eq!(n.ids, vec![1, 2, 3]);
        assert_eq!(n.coords, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(n.len(), 3);
    }

    #[test]
    fn parses_fixed_width_nodes() {
        // Exactly the layout testgen emits: I8, three F16.6, I8, I8, padding.
        let line1 = format!("{:>8}{:>16.6}{:>16.6}{:>16.6}{:>8}{:>8}        ", 1, 1.5, 2.5, 0.1, 0, 0);
        let line2 = format!("{:>8}{:>16.6}{:>16.6}{:>16.6}{:>8}{:>8}        ", 42, -3.0, 4.0, 5.0, 0, 0);
        let src = format!("*NODE\n{}\n{}\n*END\n", line1, line2);
        let p = parsed(src.as_bytes());
        let n = parse_nodes(&p);
        assert_eq!(n.ids, vec![1, 42]);
        assert_eq!(&n.coords[0..3], &[1.5, 2.5, 0.1]);
        assert_eq!(&n.coords[3..6], &[-3.0, 4.0, 5.0]);
    }

    #[test]
    fn parses_shell_and_solid_connectivity() {
        let src = b"*ELEMENT_SHELL\n\
                    1,10,1,2,3,4\n\
                    2,10,5,6,7,8\n\
                    *ELEMENT_SOLID\n\
                    100,20,1,2,3,4,5,6,7,8\n";
        let p = parsed(src);

        let shell = parse_element_shell(&p);
        assert_eq!(shell.nodes_per_elem, 4);
        assert_eq!(shell.eids, vec![1, 2]);
        assert_eq!(shell.pids, vec![10, 10]);
        assert_eq!(shell.nodes, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let solid = parse_element_solid(&p);
        assert_eq!(solid.nodes_per_elem, 8);
        assert_eq!(solid.eids, vec![100]);
        assert_eq!(solid.nodes, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn coord_edit_round_trips_and_leaves_other_blocks_verbatim() {
        let src = b"*KEYWORD\n\
                    *NODE\n\
                    1,0.0,0.0,0.0\n\
                    2,1.0,1.0,1.0\n\
                    *ELEMENT_SHELL\n\
                    1,10,1,2,1,1\n\
                    *END\n";
        let mut p = parsed(src);

        // Shift every node's x by +100.
        let mut coords = parse_nodes(&p).coords;
        for i in (0..coords.len()).step_by(3) {
            coords[i] += 100.0;
        }
        update_node_coords(&mut p, &coords).unwrap();
        assert!(p.is_dirty());

        // Re-parse the rewritten bytes; coordinates must reflect the edit.
        let rewritten = p.to_bytes();
        let p2 = parsed(&rewritten);
        let n2 = parse_nodes(&p2);
        assert_eq!(n2.ids, vec![1, 2]);
        assert_eq!(&n2.coords[0..3], &[100.0, 0.0, 0.0]);
        assert_eq!(&n2.coords[3..6], &[101.0, 1.0, 1.0]);

        // The untouched *ELEMENT_SHELL / *END blocks survive verbatim.
        let text = String::from_utf8(rewritten).unwrap();
        assert!(text.contains("*ELEMENT_SHELL\n1,10,1,2,1,1\n"));
        assert!(text.contains("*END\n"));
        assert!(text.starts_with("*KEYWORD\n"));
    }

    #[test]
    fn coord_edit_rejects_wrong_length() {
        let src = b"*NODE\n1,0.0,0.0,0.0\n";
        let mut p = parsed(src);
        assert!(update_node_coords(&mut p, &[1.0, 2.0]).is_err());
    }
}
