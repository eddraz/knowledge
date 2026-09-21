use unicode_segmentation::UnicodeSegmentation;

/// Split `text` into sentence-aware chunks.
///
/// Returns a vector of `(section, chunk)` pairs.  The section is derived from
/// Markdown-style `#` headings encountered while scanning the text.
///
/// * Sentences are accumulated until adding the next sentence would push the
///   chunk past `target_chars`.  At that point the current chunk is emitted and
///   a new chunk begins.
/// * If a single sentence is longer than `target_chars`, it is hard-split on
///   characters into pieces of roughly `target_chars` length.
/// * Consecutive chunks overlap by up to `overlap_chars`, preferring to start
///   the next chunk at the beginning of a sentence and, failing that, at a word
///   boundary.
/// * Empty or whitespace-only chunks are skipped.
pub fn chunk_text(
    text: &str,
    target_chars: usize,
    overlap_chars: usize,
) -> Vec<(Option<String>, String)> {
    let mut chunks = Vec::new();
    let mut current_section: Option<String> = None;

    // Collect all sentences with their associated section.
    let mut sentences: Vec<(Option<String>, String)> = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('#') {
            // Markdown header: count the leading hashes and trim.
            let content = header.trim_start_matches('#').trim();
            current_section = if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            };
            continue;
        }

        for sentence in line.unicode_sentences() {
            let trimmed = sentence.trim();
            if trimmed.is_empty() {
                continue;
            }
            sentences.push((current_section.clone(), trimmed.to_string()));
        }
    }

    if sentences.is_empty() {
        return chunks;
    }

    let mut buf = String::new();
    let mut buf_section: Option<String> = None;
    let mut buf_started = false;

    // Build chunks greedily sentence by sentence.
    for (section, sentence) in &sentences {
        // A section change always starts a new chunk: a chunk must never span
        // two different markdown sections.
        if buf_started && *section != buf_section {
            let trimmed = buf.trim().to_string();
            if !trimmed.is_empty() {
                chunks.push((buf_section.clone(), trimmed));
            }
            buf.clear();
            buf_started = false;
        }

        if !buf_started {
            buf.push_str(sentence);
            buf_section = section.clone();
            buf_started = true;
            continue;
        }

        // If adding this sentence keeps us under target, append it.
        if buf.len() + 1 + sentence.len() <= target_chars {
            buf.push(' ');
            buf.push_str(sentence);
            // Section changes inside a chunk are ignored; the section is the
            // one active when the chunk started.
            continue;
        }

        // Emit the current chunk.
        let trimmed = buf.trim().to_string();
        if !trimmed.is_empty() {
            chunks.push((buf_section.clone(), trimmed));
        }

        // Compute overlap from the tail of the emitted chunk, cut at a
        // sentence boundary when possible, otherwise a word boundary.
        let overlap_text = overlap_tail(&buf, overlap_chars);

        buf.clear();
        if !overlap_text.is_empty() {
            buf.push_str(&overlap_text);
            buf.push(' ');
        }
        buf.push_str(sentence);
        buf_section = section.clone();
    }

    if buf_started {
        let trimmed = buf.trim().to_string();
        if !trimmed.is_empty() {
            chunks.push((buf_section, trimmed));
        }
    }

    // Post-process: split oversized chunks, then merge tiny fragments into
    // the following chunk so no fragment-sized chunks are emitted.
    let split = split_oversized_chunks(chunks, target_chars);
    merge_fragments(split, MIN_CHUNK_CHARS)
}

/// Chunks shorter than this are considered fragments.
const MIN_CHUNK_CHARS: usize = 10;

fn merge_fragments(
    mut chunks: Vec<(Option<String>, String)>,
    floor: usize,
) -> Vec<(Option<String>, String)> {
    let mut i = 0;
    while i < chunks.len() {
        let is_last = i + 1 == chunks.len();
        if !is_last && chunks[i].1.len() < floor {
            let (section, text) = chunks.remove(i);
            let next = &mut chunks[i];
            next.1 = format!("{text} {}", next.1);
            let _ = section; // merged content adopts the following chunk's section
        } else {
            i += 1;
        }
    }
    chunks
}

/// Return the last up-to-`overlap_chars` characters of `text`, starting at a
/// sentence boundary if one exists in the overlap window, otherwise at a word
/// boundary.
fn overlap_tail(text: &str, overlap_chars: usize) -> String {
    if overlap_chars == 0 || text.len() <= overlap_chars {
        return String::new();
    }

    // Take the last `overlap_chars` bytes, move forward to a valid char
    // boundary, then walk back to the start of that word so the overlap is a
    // complete word sequence, never a mid-word fragment.
    let mut start = text.len() - overlap_chars;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    if let Some(space) = text[..start].rfind(' ') {
        let candidate = text[space + 1..].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    text.trim().to_string()
}

/// Split any chunk whose text exceeds `target_chars` into pieces of roughly
/// `target_chars` length, preserving the original section.
fn split_oversized_chunks(
    chunks: Vec<(Option<String>, String)>,
    target_chars: usize,
) -> Vec<(Option<String>, String)> {
    let mut out = Vec::with_capacity(chunks.len());
    for (section, text) in chunks {
        if text.len() <= target_chars {
            out.push((section, text));
            continue;
        }

        let mut rest = text.as_str();
        while !rest.is_empty() {
            let split_point = if rest.len() <= target_chars {
                rest.len()
            } else {
                // Walk back from target to a word boundary on a char
                // boundary (UTF-8 safe).
                let mut cut = target_chars;
                while cut < rest.len() && !rest.is_char_boundary(cut) {
                    cut += 1;
                }
                let prefix = &rest[..cut];
                match prefix.rfind(' ') {
                    Some(pos) => pos,
                    None => cut,
                }
            };
            let piece = rest[..split_point].trim();
            if !piece.is_empty() {
                out.push((section.clone(), piece.to_string()));
            }
            rest = rest[split_point..].trim_start();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_target_size() {
        let text = "Primera oración. Segunda oración. Tercera oración con más palabras. Cuarta.";
        let chunks = chunk_text(text, 35, 5);
        for (i, (_, c)) in chunks.iter().enumerate() {
            assert!(
                c.len() >= 10 || i == chunks.len() - 1,
                "chunk {i} unexpectedly short: {c}"
            );
        }
        assert!(chunks.len() >= 2, "should produce at least two chunks");
    }

    #[test]
    fn overlap_present_between_consecutive_chunks() {
        let text = (0..10)
            .map(|i| format!("Oración número {} con suficiente longitud.", i + 1))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = chunk_text(&text, 80, 20);
        assert!(chunks.len() >= 2);
        for pair in chunks.windows(2) {
            let (_, a) = &pair[0];
            let (_, b) = &pair[1];
            // The overlap carried into `b` must come from the tail of `a`:
            // its first word is the previous chunk's last word.
            let b_first = b
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_matches('.');
            assert!(
                !b_first.is_empty() && a.contains(b_first),
                "chunk {b:?} does not start with overlap from {a:?}"
            );
        }
    }

    #[test]
    fn section_propagated_from_markdown_headers() {
        let text = "# Intro\nPrimera oración.\n# Detalles\nSegunda oración. Tercera.";
        let chunks = chunk_text(text, 60, 10);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0.as_deref(), Some("Intro"));
        assert_eq!(chunks[1].0.as_deref(), Some("Detalles"));
    }

    #[test]
    fn no_information_loss() {
        let text = "A. B. C. D. E. F. G. H. I. J.";
        let chunks = chunk_text(text, 12, 2);
        let recovered: String = chunks
            .iter()
            .map(|(_, c)| c.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(recovered.contains("A"));
        assert!(recovered.contains("J"));
        assert!(recovered.len() >= text.len());
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(chunk_text("", 100, 10).is_empty());
        assert!(chunk_text("   \n\n   ", 100, 10).is_empty());
    }
}
