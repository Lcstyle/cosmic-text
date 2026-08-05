// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rope-backed line storage for a [`Buffer`](crate::Buffer).
//!
//! `RopeStore` is the data and read surface of the rope arm of the buffer's
//! line storage: text lives in a rope, per-line metadata is stored sparsely,
//! shaping and layout results live in an LRU cache, and [`BufferLine`]s are
//! materialized on demand into a bounded LRU so only lines actually visited
//! carry the full per-line cost.

#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::borrow::Cow;

use core::num::NonZeroUsize;
use lru::LruCache;

use crate::{Attrs, BufferLine, LineCache, LineEnding, RopeText, Shaping, SparseMetadata};

/// Capacity of the materialized-line LRU (~4 windows of shaped lines).
const MATERIALIZED_CAPACITY: usize = 2048;

/// Rope-backed line storage for a [`Buffer`](crate::Buffer).
///
/// Composes the rope machinery: [`RopeText`] for the text itself,
/// [`SparseMetadata`] for per-line attributes that differ from defaults, a
/// [`LineCache`] for shaping/layout results, and an LRU of materialized
/// [`BufferLine`]s. Reads are cheap anywhere in the file; mutation goes
/// through [`RopeStore::thaw`], which drains the store into the classic
/// full line vector.
#[derive(Debug)]
pub struct RopeStore {
    text: RopeText,
    metadata: SparseMetadata,
    pub(crate) cache: LineCache,
    materialized: LruCache<usize, BufferLine>,
}

impl Clone for RopeStore {
    /// Clones share the rope and metadata cheaply; the shape/layout and
    /// materialized-line caches start cold.
    fn clone(&self) -> Self {
        Self {
            text: self.text.clone(),
            metadata: self.metadata.clone(),
            cache: LineCache::default(),
            materialized: LruCache::new(
                NonZeroUsize::new(MATERIALIZED_CAPACITY).expect("capacity must be > 0"),
            ),
        }
    }
}

impl RopeStore {
    fn with_text(text: RopeText, attrs: &Attrs, shaping: Shaping) -> Self {
        let mut metadata = SparseMetadata::new();
        metadata.set_default_attrs(attrs);
        metadata.set_default_shaping(shaping);
        Self {
            text,
            metadata,
            cache: LineCache::default(),
            materialized: LruCache::new(
                NonZeroUsize::new(MATERIALIZED_CAPACITY).expect("capacity must be > 0"),
            ),
        }
    }

    /// Create a store from a string.
    pub fn from_text(text: &str, attrs: &Attrs, shaping: Shaping) -> Self {
        Self::with_text(RopeText::from_str(text), attrs, shaping)
    }

    /// Create a store by streaming from a reader (e.g. a file), without
    /// buffering the whole content in a `String` first.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or the data is not valid UTF-8,
    /// like `read_to_string`.
    #[cfg(feature = "std")]
    pub fn from_reader<R: std::io::Read>(
        reader: R,
        attrs: &Attrs,
        shaping: Shaping,
    ) -> std::io::Result<Self> {
        Ok(Self::with_text(
            RopeText::from_reader(reader)?,
            attrs,
            shaping,
        ))
    }

    /// Number of lines in the store. Never less than 1: an empty text is one
    /// empty line, matching `Buffer` semantics.
    pub fn line_count(&self) -> usize {
        self.text.line_count().max(1)
    }

    /// Text of line `i` without its line ending. Cheap for any line in the
    /// file; does not touch the materialized-line cache.
    pub fn line_text(&self, i: usize) -> Option<Cow<'_, str>> {
        self.text.line_text(i)
    }

    /// Line ending for line `i`: the sparse-metadata override if one is set,
    /// otherwise detected from the rope text itself.
    fn line_ending(&mut self, i: usize) -> LineEnding {
        match self.metadata.line_ending(i) {
            LineEnding::None => self.text.line_ending(i),
            ending => ending,
        }
    }

    fn build_line(&mut self, i: usize) -> BufferLine {
        let text = self.text.line_text(i).unwrap_or_default().into_owned();
        let ending = self.line_ending(i);
        let attrs_list = self.metadata.attrs_list(i);
        let shaping = self.metadata.shaping(i);
        let mut line = BufferLine::new(text, ending, attrs_list, shaping);
        if let Some(align) = self.metadata.align(i) {
            line.set_align(Some(align));
        }
        line
    }

    /// Materialize line `i` into the LRU and return it. `&mut` because it
    /// caches; `Buffer`'s read paths that need `&BufferLine` route through
    /// the shaped window kept warm by `shape_until_scroll`.
    pub fn materialize(&mut self, i: usize) -> Option<&BufferLine> {
        if i >= self.line_count() {
            return None;
        }
        if !self.materialized.contains(&i) {
            let line = self.build_line(i);
            self.materialized.put(i, line);
        }
        self.materialized.get(&i).map(|l| &*l)
    }

    /// Return line `i` only if it is already materialized — a cache hit with
    /// no fault. Cold lines return `None`; use [`RopeStore::line_text`] for
    /// text reads that must work anywhere in the file.
    pub fn materialized(&self, i: usize) -> Option<&BufferLine> {
        self.materialized.peek(&i)
    }

    /// Drain everything into a full line vector (the thaw path).
    ///
    /// Linear in the size of the file; the one-time cost of making a
    /// rope-backed buffer editable.
    pub fn thaw(mut self) -> Vec<BufferLine> {
        let count = self.line_count();
        let mut lines = Vec::with_capacity(count);
        for i in 0..count {
            let line = self.build_line(i);
            lines.push(line);
        }
        lines
    }

    /// Total text length in bytes.
    pub fn text_len_bytes(&self) -> usize {
        self.text.len_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(text: &str) -> RopeStore {
        RopeStore::from_text(text, &Attrs::new(), Shaping::Advanced)
    }

    /// Rebuild the original text from materialized lines: text + ending per line.
    fn reconstruct(store: &mut RopeStore) -> String {
        let mut out = String::new();
        for i in 0..store.line_count() {
            let line = store.materialize(i).expect("line index in bounds");
            out.push_str(line.text());
            out.push_str(line.ending().as_str());
        }
        out
    }

    #[test]
    fn materialize_round_trip_crlf() {
        let input = "one\r\ntwo\r\nthree";
        let mut s = store(input);
        assert_eq!(s.line_count(), 3);
        {
            let l0 = s.materialize(0).expect("line 0");
            assert_eq!(l0.text(), "one");
            assert_eq!(l0.ending(), LineEnding::CrLf);
        }
        {
            let l2 = s.materialize(2).expect("line 2");
            assert_eq!(l2.text(), "three");
            assert_eq!(l2.ending(), LineEnding::None);
        }
        assert_eq!(reconstruct(&mut s), input);
    }

    #[test]
    fn materialize_round_trip_trailing_newline() {
        let input = "alpha\nbeta\n";
        let mut s = store(input);
        // Ropey counts the empty line after the final newline.
        assert_eq!(s.line_count(), 3);
        assert_eq!(s.materialize(2).map(|l| l.text()), Some(""));
        assert_eq!(reconstruct(&mut s), input);
    }

    #[test]
    fn materialize_round_trip_no_trailing_newline() {
        let input = "alpha\nbeta";
        let mut s = store(input);
        assert_eq!(s.line_count(), 2);
        {
            let l1 = s.materialize(1).expect("line 1");
            assert_eq!(l1.text(), "beta");
            assert_eq!(l1.ending(), LineEnding::None);
        }
        assert_eq!(reconstruct(&mut s), input);
    }

    #[test]
    fn materialize_round_trip_empty() {
        let mut s = store("");
        assert_eq!(s.line_count(), 1);
        {
            let l0 = s.materialize(0).expect("line 0");
            assert_eq!(l0.text(), "");
            assert_eq!(l0.ending(), LineEnding::None);
        }
        assert_eq!(reconstruct(&mut s), "");
    }

    #[test]
    fn materialize_round_trip_multibyte() {
        let input = "héllo wörld\n日本語のテキスト\nэмодзи 😀🦀\n";
        let mut s = store(input);
        assert_eq!(s.materialize(1).map(|l| l.text()), Some("日本語のテキスト"));
        assert_eq!(reconstruct(&mut s), input);
    }

    #[test]
    fn materialize_out_of_bounds() {
        let mut s = store("a\nb");
        assert!(s.materialize(2).is_none());
    }

    #[test]
    fn materialized_is_cache_hit_only() {
        let mut s = store("a\nb\nc");
        assert!(s.materialized(1).is_none());
        assert!(s.materialize(1).is_some());
        assert_eq!(s.materialized(1).map(|l| l.text()), Some("b"));
        // peek must not fault line 2 in
        assert!(s.materialized(2).is_none());
    }

    #[test]
    fn thaw_matches_input() {
        for input in [
            "one\r\ntwo\r\nthree",
            "alpha\nbeta\n",
            "no newline",
            "",
            "日本語\nmixed 😀\r\nend",
        ] {
            let lines = store(input).thaw();
            let mut out = String::new();
            for line in &lines {
                out.push_str(line.text());
                out.push_str(line.ending().as_str());
            }
            assert_eq!(out, input, "thaw must reconstruct the input exactly");
        }
    }

    #[test]
    fn thaw_matches_materialized_lines() {
        let input = "a\nbb\r\nccc\n";
        let mut s = store(input);
        let materialized: Vec<(String, LineEnding)> = (0..s.line_count())
            .map(|i| {
                let line = s.materialize(i).expect("line in bounds");
                (line.text().to_string(), line.ending())
            })
            .collect();
        let thawed = store(input).thaw();
        assert_eq!(thawed.len(), materialized.len());
        for (line, (text, ending)) in thawed.iter().zip(&materialized) {
            assert_eq!(line.text(), text);
            assert_eq!(line.ending(), *ending);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_reader_matches_from_text() {
        let input = "line one\nline two\r\nliné three 🦀\nlast";
        let mut a = store(input);
        let mut b = RopeStore::from_reader(
            std::io::Cursor::new(input.as_bytes()),
            &Attrs::new(),
            Shaping::Advanced,
        )
        .expect("in-memory read cannot fail");
        assert_eq!(b.line_count(), a.line_count());
        assert_eq!(b.text_len_bytes(), a.text_len_bytes());
        assert_eq!(b.text_len_bytes(), input.len());
        for i in 0..a.line_count() {
            assert_eq!(b.line_text(i), a.line_text(i));
        }
        assert_eq!(reconstruct(&mut b), reconstruct(&mut a));
    }

    #[test]
    fn lru_eviction_does_not_change_results() {
        // 3000 lines against a 2048-line cache: early lines get evicted and
        // must re-materialize identically.
        let input = (0..3000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut s = store(&input);
        assert_eq!(s.line_count(), 3000);
        for i in 0..3000 {
            let line = s.materialize(i).expect("line in bounds");
            assert_eq!(line.text(), format!("line {i}"));
        }
        // Sequential materialization retains only the most recent 2048 lines.
        assert!(s.materialized(0).is_none());
        let line0 = s.materialize(0).expect("re-materialize after eviction");
        assert_eq!(line0.text(), "line 0");
        assert_eq!(line0.ending(), LineEnding::Lf);
    }
}
