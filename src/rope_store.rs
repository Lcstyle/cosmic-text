// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rope-backed line storage for a [`Buffer`](crate::Buffer).
//!
//! `RopeStore` is the data surface of the rope arm of the buffer's line
//! storage: text lives in a rope, per-line metadata is stored sparsely,
//! shaping and layout results live in an LRU cache, and [`BufferLine`]s are
//! materialized on demand into a bounded LRU so only lines actually visited
//! carry the full per-line cost. Plain-text edits are native: they splice
//! the rope via byte offsets ([`RopeStore::insert_text`] /
//! [`RopeStore::delete_text`]) and shift or invalidate the caches and
//! metadata around the edit point — no thaw, no whole-file work.

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
/// [`BufferLine`]s. Reads are cheap anywhere in the file. Plain-text edits
/// are native: [`RopeStore::insert_text`] and [`RopeStore::delete_text`]
/// splice the rope in place and shift or invalidate metadata and caches
/// around the edit. Only rich-text (per-span attrs) mutation still drains
/// the store into the classic full line vector via [`RopeStore::thaw`].
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

    /// Line ending of line `i` without materializing and without caching:
    /// the sparse-metadata override if set, else detected from the rope
    /// bytes. `&self` and cold-capable — the read exact serialization needs.
    pub(crate) fn ending(&self, i: usize) -> LineEnding {
        match self.metadata.line_ending(i) {
            LineEnding::None => self.text.detect_line_ending(i),
            ending => ending,
        }
    }

    /// Map an absolute byte offset to its (line, byte-index-within-line)
    /// position in the current text.
    fn map_byte(&self, byte: usize) -> (usize, usize) {
        let line = self.text.byte_to_line(byte);
        (line, byte - self.text.line_to_byte(line))
    }

    /// Insert `data` at byte position `index` of line `line`. Splices the
    /// rope and shifts/invalidates metadata, the shape/layout caches and the
    /// materialized-line LRU. Returns the post-insert byte-true (line, index)
    /// positions of the edit boundaries: `(start, end)` where `start` maps
    /// `start_byte` and `end` maps `start_byte + data.len()` in the
    /// post-insert text. `start` equals the input position except when a
    /// CR|LF adjacency merge re-segments the boundary (the rope fuses an
    /// adjacent `\r` and `\n` into one CRLF break).
    ///
    /// Preconditions (unchecked, same loudness contract as the full arm's
    /// `String::split_off`): `line < self.line_count()`, `index` is a byte
    /// offset from the line start on a char boundary; `index` may point into
    /// or past the line's ending bytes (replay of byte-true cursors) —
    /// `line_to_byte(line) + index` is well-defined regardless.
    pub(crate) fn insert_text(
        &mut self,
        line: usize,
        index: usize,
        data: &str,
    ) -> ((usize, usize), (usize, usize)) {
        let start_byte = self.text.line_to_byte(line) + index;
        let old_line_count = self.line_count();
        self.text.insert(start_byte, data);
        let new_line_count = self.line_count();
        let delta = new_line_count as isize - old_line_count as isize;

        // Both boundaries mapped in the post-insert text (byte-true rule).
        let start = self.map_byte(start_byte);
        let end = self.map_byte(start_byte + data.len());
        // In a CR|LF merge the preceding line absorbed the boundary; its
        // cached state went stale even though the edit "happened" on `line`.
        let mapped = start.0;

        // RopeText::insert self-invalidates only `line`'s cached ending; a
        // merge leaves e.g. a cached Cr on the absorbing line while the
        // bytes now say CrLf.
        if mapped != line {
            let ending = self.text.detect_line_ending(mapped);
            self.text.set_line_ending(mapped, ending);
        }

        // Sparse metadata: entries above the edit shift with their lines.
        // The head line's ending override cannot survive in place (on a
        // split the tail line carries the original ending; on a fusion the
        // merged ending belongs to neither input line — bytes are truth).
        // Attrs-list overrides deliberately stay on `line`.
        let ending_override = self.metadata.line_ending(line);
        if delta > 0 {
            self.metadata.shift_lines(line + 1, delta);
        }
        self.metadata.set_line_ending(line, LineEnding::None);
        // Migrate the override only for a clean intra-line split
        // (mapped == line): there the tail line really does carry the
        // original terminator bytes. When the terminator itself was cloven
        // (insert between the \r and \n of a CRLF: mapped == line + 1) or
        // fused (CR|LF merge: mapped < line), NO surviving line carries the
        // original ending — bytes are truth, migrating would corrupt
        // serialization.
        if delta > 0 && mapped == line && ending_override != LineEnding::None {
            self.metadata
                .set_line_ending(line + delta as usize, ending_override);
        }
        if mapped != line {
            self.metadata.set_line_ending(mapped, LineEnding::None);
        }

        // Shape/layout caches: entries above the edit keep valid content at
        // shifted indices; new lines are cold and fault in on demand.
        if delta > 0 {
            self.cache.shift_lines(line + 1, delta);
        }
        self.cache.invalidate_line(line);
        if mapped != line {
            self.cache.invalidate_line(mapped);
        }

        // Materialized-line LRU: materialize() short-circuits on a present
        // entry, so stale lines must be popped, never left to age out.
        self.materialized.pop(&line);
        if delta > 0 {
            self.shift_materialized(line + 1, delta);
        }
        if mapped != line {
            self.materialized.pop(&mapped);
        }

        (start, end)
    }

    /// Delete `[(start_line, start_index), (end_line, end_index))`. Returns
    /// `(start, removed)`: `start` is the post-delete byte-true (line, index)
    /// of the join point (equals the input start except across a CR|LF
    /// adjacency merge), `removed` is the removed text exactly (endings
    /// included), captured before the splice — byte-identical to the full
    /// arm's `ChangeItem` text. Same precondition contract as
    /// [`RopeStore::insert_text`]; requires start <= end. An empty range
    /// returns `((start_line, start_index), String::new())` without
    /// mutating.
    pub(crate) fn delete_text(
        &mut self,
        start_line: usize,
        start_index: usize,
        end_line: usize,
        end_index: usize,
    ) -> ((usize, usize), String) {
        if start_line == end_line && start_index == end_index {
            return ((start_line, start_index), String::new());
        }
        let start_byte = self.text.line_to_byte(start_line) + start_index;
        let end_byte = self.text.line_to_byte(end_line) + end_index;
        let removed = self.text.slice(start_byte, end_byte).into_owned();
        let old_line_count = self.line_count();
        self.text.delete(start_byte, end_byte);
        let new_line_count = self.line_count();
        let removed_count = old_line_count - new_line_count;

        let start = self.map_byte(start_byte);
        let mapped = start.0;

        // RopeText::delete self-invalidates only the start line; a CR|LF
        // fusion staled the absorbing line's cached ending too.
        if mapped != start_line {
            let ending = self.text.detect_line_ending(mapped);
            self.text.set_line_ending(mapped, ending);
        }

        if removed_count > 0 {
            let end_override = self.metadata.line_ending(end_line);
            self.metadata.remove_lines(mapped + 1, removed_count);
            if mapped == start_line {
                // The merged line takes the end line's ending (the full-arm
                // rule) — including becoming unset when the end line had no
                // override, which clears a now-stale start-line override.
                self.metadata.set_line_ending(start_line, end_override);
            } else {
                // A fused ending belongs to neither input line; clear so
                // detection from the bytes wins.
                self.metadata.set_line_ending(mapped, LineEnding::None);
            }
        }

        // Deliberately invalidate instead of shifting: a shift would alias
        // entries from the deleted range onto surviving lines. Bounded by
        // the LRU capacities; only lines scrolling into view re-shape.
        let invalidate_from = start_line.min(mapped);
        if removed_count > 0 {
            self.cache.invalidate_from(invalidate_from);
            self.invalidate_materialized_from(invalidate_from);
        } else {
            self.cache.invalidate_line(invalidate_from);
            self.materialized.pop(&invalidate_from);
        }

        (start, removed)
    }

    /// Shift materialized-line LRU keys at or after `start` by `delta`.
    /// Two-phase (pop all affected, then re-insert) so shifted entries never
    /// collide with not-yet-shifted ones.
    fn shift_materialized(&mut self, start: usize, delta: isize) {
        if delta == 0 {
            return;
        }
        let keys: Vec<usize> = self
            .materialized
            .iter()
            .filter_map(|(k, _)| (*k >= start).then_some(*k))
            .collect();
        let mut moved: Vec<(usize, BufferLine)> = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(line) = self.materialized.pop(&key) {
                let new_key = if delta > 0 {
                    key.checked_add(delta as usize)
                } else {
                    key.checked_sub((-delta) as usize)
                };
                if let Some(new_key) = new_key {
                    moved.push((new_key, line));
                }
            }
        }
        for (key, line) in moved {
            self.materialized.put(key, line);
        }
    }

    /// Drop all materialized lines at or after `start`.
    fn invalidate_materialized_from(&mut self, start: usize) {
        let keys: Vec<usize> = self
            .materialized
            .iter()
            .filter_map(|(k, _)| (*k >= start).then_some(*k))
            .collect();
        for key in keys {
            self.materialized.pop(&key);
        }
    }

    /// Drain everything into a full line vector (the thaw path).
    ///
    /// Linear in the size of the file. Plain-text edits no longer need this
    /// (see [`RopeStore::insert_text`] / [`RopeStore::delete_text`]); it
    /// remains the escape hatch for rich-text mutation, which needs real
    /// per-line [`BufferLine`]s to carry per-span attrs.
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

    /// Inserting between the \r and \n of a CRLF cleaves the terminator
    /// itself: the head keeps \r, the tail gets \n, and NO surviving line
    /// carries the original CrLf ending. A stale metadata override must not
    /// be migrated onto any of them — bytes are truth, and reconstruction
    /// must stay byte-exact.
    #[test]
    fn crlf_split_insert_does_not_migrate_ending_override() {
        let mut s = store("head\r\nnext\n");
        // A stale CrLf override on the line whose bytes currently agree.
        s.metadata.set_line_ending(0, LineEnding::CrLf);

        // Split the CRLF: "head\r" | "X\nY" | "\nnext\n"
        s.insert_text(0, 5, "X\nY");

        assert_eq!(reconstruct(&mut s), "head\rX\nY\nnext\n");
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
    fn ending_is_cold_capable() {
        let s = store("a\nb\r\nc");
        // &self — must not materialize or cache anything.
        assert_eq!(s.ending(0), LineEnding::Lf);
        assert_eq!(s.ending(1), LineEnding::CrLf);
        assert_eq!(s.ending(2), LineEnding::None);
        assert!(s.materialized(0).is_none());
    }

    #[test]
    fn insert_text_same_line() {
        let mut s = store("abc\ndef\nghi");
        s.materialize(0);
        s.materialize(1);
        s.materialize(2);
        let (start, end) = s.insert_text(1, 1, "XY");
        assert_eq!(start, (1, 1));
        assert_eq!(end, (1, 3));
        assert_eq!(s.line_count(), 3);
        // The edited line is popped (a present entry is never rebuilt);
        // untouched lines stay warm.
        assert!(s.materialized(1).is_none());
        assert_eq!(s.materialized(0).map(|l| l.text()), Some("abc"));
        assert_eq!(s.materialized(2).map(|l| l.text()), Some("ghi"));
        assert_eq!(s.line_text(1).as_deref(), Some("dXYef"));
        assert_eq!(reconstruct(&mut s), "abc\ndXYef\nghi");
    }

    #[test]
    fn insert_text_with_newline_shifts_caches_and_metadata() {
        let mut s = store("abc\ndef\nghi");
        // A self-consistent override (matches the detected ending) so the
        // shift is observable without changing reconstruction.
        s.metadata.set_line_ending(1, LineEnding::Lf);
        s.materialize(0);
        s.materialize(1);
        s.materialize(2);
        s.cache.shape.insert(2, crate::ShapeLine::empty());

        let (start, end) = s.insert_text(0, 3, "X\nY");
        assert_eq!(start, (0, 3));
        assert_eq!(end, (1, 1));
        assert_eq!(s.line_count(), 4);
        // Above-edit entries shifted to their new absolute indices.
        assert_eq!(s.materialized(2).map(|l| l.text()), Some("def"));
        assert_eq!(s.materialized(3).map(|l| l.text()), Some("ghi"));
        assert!(s.materialized(0).is_none(), "edited line must be popped");
        assert!(s.materialized(1).is_none(), "new line is cold");
        assert!(s.cache.shape.contains(3), "shape entry follows its line");
        assert!(!s.cache.shape.contains(2));
        // The metadata override followed its line.
        assert_eq!(s.metadata.line_ending(2), LineEnding::Lf);
        assert_eq!(s.metadata.line_ending(1), LineEnding::None);
        assert_eq!(reconstruct(&mut s), "abcX\nY\ndef\nghi");
    }

    #[test]
    fn insert_text_split_carries_ending_override_to_tail() {
        let mut s = store("abc\ndef");
        // Self-consistent override on the edited line.
        s.metadata.set_line_ending(0, LineEnding::Lf);
        let (start, end) = s.insert_text(0, 1, "1\n2");
        assert_eq!(start, (0, 1));
        assert_eq!(end, (1, 1));
        // The tail line carries the original ending override; the head
        // line's override is cleared (its ending is now the inserted break).
        assert_eq!(s.metadata.line_ending(0), LineEnding::None);
        assert_eq!(s.metadata.line_ending(1), LineEnding::Lf);
        assert_eq!(reconstruct(&mut s), "a1\n2bc\ndef");
    }

    #[test]
    fn insert_text_trailing_break_lands_on_next_line() {
        let mut s = store("abc");
        let (start, end) = s.insert_text(0, 1, "tail\n");
        assert_eq!(start, (0, 1));
        assert_eq!(end, (1, 0));
        assert_eq!(s.line_count(), 2);
        assert_eq!(reconstruct(&mut s), "atail\nbc");
    }

    #[test]
    fn delete_text_same_line_keeps_other_lines_warm() {
        let mut s = store("abc\ndef");
        s.materialize(0);
        s.materialize(1);
        let (start, removed) = s.delete_text(1, 0, 1, 2);
        assert_eq!(start, (1, 0));
        assert_eq!(removed, "de");
        assert!(s.materialized(1).is_none());
        assert_eq!(
            s.materialized(0).map(|l| l.text()),
            Some("abc"),
            "line below the edit stays warm"
        );
        assert_eq!(reconstruct(&mut s), "abc\nf");
    }

    #[test]
    fn delete_text_cross_line_invalidates_from_edit() {
        let mut s = store("abc\ndef\nghi\njkl");
        s.materialize(0);
        s.materialize(3);
        let (start, removed) = s.delete_text(1, 1, 2, 1);
        assert_eq!(start, (1, 1));
        assert_eq!(removed, "ef\ng");
        assert_eq!(s.line_count(), 3);
        assert_eq!(s.line_text(1).as_deref(), Some("dhi"));
        // invalidate_from(1): entries at or above the edit are dropped,
        // entries below survive.
        assert!(s.materialized(3).is_none());
        assert!(s.materialized(2).is_none());
        assert_eq!(s.materialized(0).map(|l| l.text()), Some("abc"));
        assert_eq!(reconstruct(&mut s), "abc\ndhi\njkl");
    }

    #[test]
    fn delete_text_merges_ending_override_from_end_line() {
        let mut s = store("one\r\ntwo\r\nthree");
        // Self-consistent overrides on both lines of the join.
        s.metadata.set_line_ending(0, LineEnding::CrLf);
        let (start, removed) = s.delete_text(0, 3, 1, 0);
        assert_eq!(start, (0, 3));
        assert_eq!(removed, "\r\n");
        assert_eq!(s.line_count(), 2);
        assert_eq!(s.line_text(0).as_deref(), Some("onetwo"));
        // The merged line takes the end line's (absent) override: the stale
        // start-line override must not survive.
        assert_eq!(s.metadata.line_ending(0), LineEnding::None);
        assert_eq!(reconstruct(&mut s), "onetwo\r\nthree");
    }

    #[test]
    fn delete_text_empty_range_is_noop() {
        let mut s = store("abc\ndef");
        s.materialize(0);
        let (start, removed) = s.delete_text(1, 2, 1, 2);
        assert_eq!(start, (1, 2));
        assert_eq!(removed, "");
        assert!(
            s.materialized(0).is_some(),
            "empty delete must not invalidate anything"
        );
        assert_eq!(reconstruct(&mut s), "abc\ndef");
    }

    #[test]
    fn delete_whole_content_leaves_one_empty_line() {
        let mut s = store("abc\ndef\n");
        let (start, removed) = s.delete_text(0, 0, 2, 0);
        assert_eq!(start, (0, 0));
        assert_eq!(removed, "abc\ndef\n");
        assert_eq!(s.line_count(), 1);
        assert_eq!(reconstruct(&mut s), "");
    }

    // The CR|LF adjacency merge family: ropey (unicode_lines) fuses an
    // adjacent \r and \n into one CRLF break, re-segmenting around the
    // edit. Returned positions are byte-true in the post-edit text.

    #[test]
    fn merge_shape_1_trailing_cr_before_lf() {
        let mut s = store("AB\ncd");
        s.materialize(0);
        let (start, end) = s.insert_text(0, 2, "\r");
        assert_eq!(start, (0, 2));
        assert_eq!(end, (0, 3), "end lands inside the merged CRLF ending");
        assert_eq!(s.line_count(), 2);
        assert!(s.materialized(0).is_none());
        assert_eq!(
            s.materialize(0).map(|l| l.ending()),
            Some(LineEnding::CrLf)
        );
        assert_eq!(reconstruct(&mut s), "AB\r\ncd");
    }

    #[test]
    fn merge_shape_2_leading_lf_after_cr() {
        let mut s = store("a\rb");
        s.materialize(0);
        s.materialize(1);
        let (start, end) = s.insert_text(1, 0, "\nX");
        assert_eq!(start, (0, 2), "start re-maps into the absorbing line");
        assert_eq!(end, (1, 1));
        assert_eq!(s.line_count(), 2);
        assert!(s.materialized(0).is_none(), "absorbing line must be popped");
        assert!(s.materialized(1).is_none(), "edited line must be popped");
        assert_eq!(
            s.materialize(0).map(|l| l.ending()),
            Some(LineEnding::CrLf),
            "cached Cr must not survive the fusion"
        );
        assert_eq!(reconstruct(&mut s), "a\r\nXb");
    }

    #[test]
    fn merge_shape_3_delete_joins_cr_and_lf() {
        let mut s = store("a\rX\nb");
        s.materialize(0);
        s.materialize(1);
        s.materialize(2);
        let (start, removed) = s.delete_text(1, 0, 1, 1);
        assert_eq!(start, (0, 2), "join point maps inside the fused CRLF");
        assert_eq!(removed, "X");
        assert_eq!(s.line_count(), 2);
        assert!(s.materialized(0).is_none());
        assert_eq!(
            s.materialize(0).map(|l| l.ending()),
            Some(LineEnding::CrLf)
        );
        assert_eq!(reconstruct(&mut s), "a\r\nb");
        // The undo hazard: replaying the byte-true join point restores the
        // original bytes exactly.
        s.insert_text(0, 2, "X");
        assert_eq!(reconstruct(&mut s), "a\rX\nb");
    }

    #[test]
    fn delete_lf_of_crlf_keeps_cr_break() {
        let mut s = store("a\r\nb");
        s.materialize(0);
        s.materialize(1);
        let (start, removed) = s.delete_text(0, 2, 0, 3);
        assert_eq!(removed, "\n");
        assert_eq!(start, (1, 0), "join maps to the start of the next line");
        assert_eq!(s.line_count(), 2);
        assert!(s.materialized(0).is_none(), "line 0 ending changed CrLf→Cr");
        assert_eq!(s.materialize(0).map(|l| l.ending()), Some(LineEnding::Cr));
        assert_eq!(reconstruct(&mut s), "a\rb");
    }

    #[cfg(feature = "std")]
    #[test]
    #[ignore = "perf smoke, run by hand: no-O(file) sanity check"]
    fn perf_insert_into_3m_line_store_is_bounded() {
        let text: String = (0..3_000_000).map(|i| format!("l{i}\n")).collect();
        let mut s = store(&text);
        let t0 = std::time::Instant::now();
        s.insert_text(1_500_000, 1, "x");
        let elapsed = t0.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "single-char insert took {elapsed:?}"
        );
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
