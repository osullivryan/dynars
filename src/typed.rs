//! Phase 4: example typed keyword structs.
//!
//! These sit on top of the generic reader and add strongly-typed access for a
//! few high-value keywords. They are intentionally a small sample — the same
//! pattern (match the keyword name, read cards, pull typed fields) extends to
//! any keyword you care to add, while everything else remains reachable
//! through the generic `Keyword` model.

use crate::keyword::ParsedFile;
use crate::parser::split_fields;

/// A `*PART` definition: a free-text title followed by a data card.
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    pub title: String,
    pub pid: i64,
    pub secid: i64,
    pub mid: i64,
}

/// A `*MAT_ELASTIC` definition.
#[derive(Debug, Clone, PartialEq)]
pub struct MatElastic {
    pub mid: i64,
    pub ro: f64,
    pub e: f64,
    pub pr: f64,
}

/// Non-comment, non-blank data lines of a block, as raw byte slices.
fn data_lines<'a>(parsed: &'a ParsedFile, block: &crate::keyword::Block) -> Vec<&'a [u8]> {
    parsed
        .body(block)
        .split(|&c| c == b'\n')
        .filter(|line| {
            let indent = line.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
            !(line.is_empty()
                || line.get(indent) == Some(&b'$')
                || line[indent..].iter().all(|&c| matches!(c, b' ' | b'\t' | b'\r')))
        })
        .collect()
}

/// Parse all `*PART` blocks. A block may define several parts as repeated
/// (title, data-card) pairs.
pub fn parse_parts(parsed: &ParsedFile) -> Vec<Part> {
    let mut parts = Vec::new();
    for block in &parsed.blocks {
        if !parsed.keyword_name(block).eq_ignore_ascii_case("PART") {
            continue;
        }
        let lines = data_lines(parsed, block);
        for pair in lines.chunks(2) {
            let [title_line, data_line] = pair else { continue };
            let title = String::from_utf8_lossy(strip_eol(title_line)).trim().to_string();
            let fields = split_fields(data_line, block.format);
            parts.push(Part {
                title,
                pid: fields.first().and_then(|f| f.as_i64()).unwrap_or(0),
                secid: fields.get(1).and_then(|f| f.as_i64()).unwrap_or(0),
                mid: fields.get(2).and_then(|f| f.as_i64()).unwrap_or(0),
            });
        }
    }
    parts
}

/// Parse all `*MAT_ELASTIC` blocks (one material per card).
pub fn parse_mat_elastic(parsed: &ParsedFile) -> Vec<MatElastic> {
    let mut mats = Vec::new();
    for block in &parsed.blocks {
        if !parsed.keyword_name(block).eq_ignore_ascii_case("MAT_ELASTIC") {
            continue;
        }
        for card in parsed.cards(block) {
            mats.push(MatElastic {
                mid: card.first().and_then(|f| f.as_i64()).unwrap_or(0),
                ro: card.get(1).and_then(|f| f.as_f64()).unwrap_or(0.0),
                e: card.get(2).and_then(|f| f.as_f64()).unwrap_or(0.0),
                pr: card.get(3).and_then(|f| f.as_f64()).unwrap_or(0.0),
            });
        }
    }
    mats
}

#[inline]
fn strip_eol(s: &[u8]) -> &[u8] {
    let mut end = s.len();
    while end > 0 && matches!(s[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    &s[..end]
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
    fn parses_parts_with_titles() {
        let src = b"*PART\n\
                    steel bracket\n\
                    1,2,3\n\
                    $ comment between\n\
                    aluminium panel\n\
                    10,20,30\n";
        let p = parsed(src);
        let parts = parse_parts(&p);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], Part { title: "steel bracket".into(), pid: 1, secid: 2, mid: 3 });
        assert_eq!(parts[1], Part { title: "aluminium panel".into(), pid: 10, secid: 20, mid: 30 });
    }

    #[test]
    fn parses_mat_elastic() {
        let src = b"*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n";
        let p = parsed(src);
        let mats = parse_mat_elastic(&p);
        assert_eq!(mats.len(), 1);
        assert_eq!(mats[0].mid, 1);
        assert_eq!(mats[0].e, 210000.0);
        assert_eq!(mats[0].pr, 0.3);
    }
}
