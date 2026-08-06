// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(not(feature = "std"))]
use alloc::{borrow::Cow, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::borrow::Cow;

use core::{cmp, fmt};

#[cfg(not(feature = "std"))]
use core_maths::CoreFloat;
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    render_decoration, Affinity, Align, Attrs, AttrsList, BidiParagraphs, BorrowedWithFontSystem,
    BufferLine, Color, Cursor, DecorationSpan, Direction, Ellipsize, FontSystem, Hinting,
    LayoutCursor, LayoutGlyph, LayoutLine, LineEnding, LineIter, Motion, Renderer, Scroll,
    ShapeLine, Shaping, Wrap,
};

bitflags::bitflags! {
    /// Tracks which buffer-wide properties have changed since the last layout.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct DirtyFlags: u8 {
        /// Layout caches are stale (wrap, size, metrics, hinting, ellipsize, monospace_width changed)
        const RELAYOUT  = 0b0001;
        /// tab_width changed — lines containing tabs need reshape
        const TAB_SHAPE = 0b0010;
        /// Text was replaced via set_text/set_rich_text — lines are fresh, just need shape_until_scroll
        const TEXT_SET  = 0b0100;
        /// Scroll position changed — visible region may have shifted to unshaped lines
        const SCROLL    = 0b1000;
        /// Base direction changed, reshape every line (some characters like '(' are shaped differently based on direction)
        const DIRECTION = 0b1_0000;
    }
}

/// A line of visible text for rendering
#[derive(Debug)]
pub struct LayoutRun<'a> {
    /// The index of the original text line
    pub line_i: usize,
    /// The original text line
    pub text: &'a str,
    /// True if the original paragraph direction is RTL
    pub rtl: bool,
    /// The array of layout glyphs to draw
    pub glyphs: &'a [LayoutGlyph],
    /// Text decoration spans covering ranges of glyphs
    pub decorations: &'a [DecorationSpan],
    /// Y offset to baseline of line
    pub line_y: f32,
    /// Y offset to top of line
    pub line_top: f32,
    /// Y offset to next line
    pub line_height: f32,
    /// Width of line
    pub line_w: f32,
}

impl LayoutRun<'_> {
    /// Return an iterator of `(x_left, x_width)` pixel spans for the highlighted areas
    /// between `cursor_start` and `cursor_end` within this run.
    ///
    /// For pure LTR or pure RTL runs this yields at most one span. For mixed BiDi runs
    /// (where selected and unselected glyphs interleave visually) it yields multiple
    /// disjoint spans.
    ///
    /// Returns an empty iterator if the cursor range does not intersect this run.
    pub fn highlight(
        &self,
        cursor_start: Cursor,
        cursor_end: Cursor,
    ) -> impl Iterator<Item = (f32, f32)> {
        let line_i = self.line_i;
        let mut results = Vec::new();
        let mut range_opt: Option<(f32, f32)> = None;

        for glyph in self.glyphs {
            let cluster = &self.text[glyph.start..glyph.end];
            let total = cluster.grapheme_indices(true).count().max(1);
            let c_w = glyph.w / total as f32;
            let mut c_x = glyph.x;

            for (i, c) in cluster.grapheme_indices(true) {
                let c_start = glyph.start + i;
                let c_end = glyph.start + i + c.len();

                let is_selected = (cursor_start.line != line_i || c_end > cursor_start.index)
                    && (cursor_end.line != line_i || c_start < cursor_end.index);

                if is_selected {
                    range_opt = Some(match range_opt {
                        Some((min, max)) => (min.min(c_x), max.max(c_x + c_w)),
                        None => (c_x, c_x + c_w),
                    });
                } else if let Some((min_x, max_x)) = range_opt.take() {
                    let width = max_x - min_x;
                    if width > 0.0 {
                        results.push((min_x, width));
                    }
                }

                c_x += c_w;
            }
        }

        // Flush remaining highlighted region
        if let Some((min_x, max_x)) = range_opt {
            let width = max_x - min_x;
            if width > 0.0 {
                results.push((min_x, width));
            }
        }

        results.into_iter()
    }

    /// Returns the visual x position (in pixels) of `cursor` within this run,
    /// or `None` if the cursor does not belong to this run.
    ///
    /// For RTL glyphs the cursor is placed at the right edge minus the offset;
    /// for LTR glyphs it is placed at the left edge plus the offset.
    pub fn cursor_position(&self, cursor: &Cursor) -> Option<f32> {
        let (glyph_idx, glyph_offset) = self.cursor_glyph(cursor)?;
        let x = self.glyphs.get(glyph_idx).map_or_else(
            || {
                // Past-the-end: position after the last glyph
                self.glyphs.last().map_or(0.0, |glyph| {
                    if glyph.level.is_rtl() {
                        glyph.x
                    } else {
                        glyph.x + glyph.w
                    }
                })
            },
            |glyph| {
                if glyph.level.is_rtl() {
                    glyph.x + glyph.w - glyph_offset
                } else {
                    glyph.x + glyph_offset
                }
            },
        );
        Some(x)
    }

    /// Find which glyph in this run contains `cursor`, returning
    /// `(glyph_index, pixel_offset_within_glyph)`, or `None` if the cursor
    /// is not on this run.
    pub fn cursor_glyph(&self, cursor: &Cursor) -> Option<(usize, f32)> {
        if cursor.line != self.line_i {
            return None;
        }
        for (glyph_i, glyph) in self.glyphs.iter().enumerate() {
            if cursor.index == glyph.start {
                return Some((glyph_i, 0.0));
            } else if cursor.index > glyph.start && cursor.index < glyph.end {
                // Guess x offset based on graphemes within the cluster
                let cluster = &self.text[glyph.start..glyph.end];
                let mut before = 0;
                let mut total = 0;
                for (i, _) in cluster.grapheme_indices(true) {
                    if glyph.start + i < cursor.index {
                        before += 1;
                    }
                    total += 1;
                }
                let offset = glyph.w * (before as f32) / (total as f32);
                return Some((glyph_i, offset));
            }
        }
        // in mixed BiDi the last logical glyph may not be the last visual glyph.
        for (glyph_i, glyph) in self.glyphs.iter().enumerate() {
            if cursor.index == glyph.end {
                return Some((glyph_i, glyph.w));
            }
        }
        if self.glyphs.is_empty() {
            return Some((0, 0.0));
        }
        None
    }

    /// Get the left-edge cursor position of a glyph, accounting for paragraph direction.
    pub const fn cursor_from_glyph_left(&self, glyph: &LayoutGlyph) -> Cursor {
        if self.rtl {
            Cursor::new_with_affinity(self.line_i, glyph.end, Affinity::Before)
        } else {
            Cursor::new_with_affinity(self.line_i, glyph.start, Affinity::After)
        }
    }

    /// Get the right-edge cursor position of a glyph, accounting for paragraph direction.
    pub const fn cursor_from_glyph_right(&self, glyph: &LayoutGlyph) -> Cursor {
        if self.rtl {
            Cursor::new_with_affinity(self.line_i, glyph.start, Affinity::After)
        } else {
            Cursor::new_with_affinity(self.line_i, glyph.end, Affinity::Before)
        }
    }
}

/// The line source a [`LayoutRunIter`] walks.
///
/// `Full` walks a slice of eagerly-materialized lines; `Rope` walks the rope
/// store's warm caches (materialized lines plus shape/layout entries), both
/// keyed by absolute line index — the iterator itself is arm-agnostic.
#[derive(Clone, Copy, Debug)]
enum RunSource<'b> {
    Full(&'b [BufferLine]),
    #[cfg(feature = "rope-buffer")]
    Rope(&'b crate::RopeStore),
}

/// An iterator of visible text lines, see [`LayoutRun`]
#[derive(Debug)]
pub struct LayoutRunIter<'b> {
    source: RunSource<'b>,
    height_opt: Option<f32>,
    line_height: f32,
    scroll: f32,
    line_i: usize,
    layout_i: usize,
    total_height: f32,
    line_top: f32,
}

impl<'b> LayoutRunIter<'b> {
    pub const fn new(buffer: &'b Buffer) -> Self {
        let source = match &buffer.store {
            LineStore::Full(lines) => RunSource::Full(lines.as_slice()),
            #[cfg(feature = "rope-buffer")]
            LineStore::Rope(store) => RunSource::Rope(store),
        };
        Self {
            source,
            height_opt: buffer.height_opt,
            line_height: buffer.metrics.line_height,
            scroll: buffer.scroll.vertical,
            line_i: buffer.scroll.line,
            layout_i: 0,
            total_height: 0.0,
            line_top: 0.0,
        }
    }

    pub const fn from_lines(
        lines: &'b [BufferLine],
        height_opt: Option<f32>,
        line_height: f32,
        scroll: f32,
        start: usize,
    ) -> Self {
        Self {
            source: RunSource::Full(lines),
            height_opt,
            line_height,
            scroll,
            line_i: start,
            layout_i: 0,
            total_height: 0.0,
            line_top: 0.0,
        }
    }
}

impl<'b> Iterator for LayoutRunIter<'b> {
    type Item = LayoutRun<'b>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (text, rtl, layout): (&'b str, bool, &'b [LayoutLine]) = match self.source {
                RunSource::Full(lines) => {
                    let line = lines.get(self.line_i)?;
                    let shape = line.shape_opt()?;
                    let layout = line.layout_opt()?;
                    (line.text(), shape.rtl, layout.as_slice())
                }
                #[cfg(feature = "rope-buffer")]
                RunSource::Rope(store) => {
                    // Warm reads only: `shape_until_scroll` materialized and
                    // shaped the visible region; iteration stops at the first
                    // cold line, exactly like an unshaped line in `Full`.
                    let line = store.materialized(self.line_i)?;
                    let shape = store.cache.shape.peek(self.line_i)?;
                    let layout = store.cache.layout.peek(self.line_i)?;
                    (line.text(), shape.rtl, layout.as_slice())
                }
            };
            while let Some(layout_line) = layout.get(self.layout_i) {
                self.layout_i += 1;

                let line_height = layout_line.line_height_opt.unwrap_or(self.line_height);
                self.total_height += line_height;

                let line_top = self.line_top - self.scroll;
                let glyph_height = layout_line.max_ascent + layout_line.max_descent;
                let centering_offset = (line_height - glyph_height) / 2.0;
                let line_y = line_top + centering_offset + layout_line.max_ascent;
                if let Some(height) = self.height_opt {
                    if line_y - layout_line.max_ascent > height {
                        return None;
                    }
                }
                self.line_top += line_height;
                if line_y + layout_line.max_descent < 0.0 {
                    continue;
                }

                return Some(LayoutRun {
                    line_i: self.line_i,
                    text,
                    rtl,
                    glyphs: &layout_line.glyphs,
                    decorations: &layout_line.decorations,
                    line_y,
                    line_top,
                    line_height,
                    line_w: layout_line.w,
                });
            }
            self.line_i += 1;
            self.layout_i = 0;
        }
    }
}

/// Metrics of text
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    /// Font size in pixels
    pub font_size: f32,
    /// Line height in pixels
    pub line_height: f32,
}

impl Metrics {
    /// Create metrics with given font size and line height
    pub const fn new(font_size: f32, line_height: f32) -> Self {
        Self {
            font_size,
            line_height,
        }
    }

    /// Create metrics with given font size and calculate line height using relative scale
    pub fn relative(font_size: f32, line_height_scale: f32) -> Self {
        Self {
            font_size,
            line_height: font_size * line_height_scale,
        }
    }

    /// Scale font size and line height
    pub fn scale(self, scale: f32) -> Self {
        Self {
            font_size: self.font_size * scale,
            line_height: self.line_height * scale,
        }
    }
}

impl fmt::Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}px / {}px", self.font_size, self.line_height)
    }
}

/// Line storage backing for a [`Buffer`].
///
/// `Full` is the classic eagerly-materialized vector. `Rope` materializes
/// lines on demand from a [`RopeStore`](crate::RopeStore); all consumers go
/// through the accessor methods below so the variant is invisible to them.
#[derive(Debug)]
pub(crate) enum LineStore {
    Full(Vec<BufferLine>),
    #[cfg(feature = "rope-buffer")]
    Rope(crate::RopeStore),
}

impl LineStore {
    /// The full line vector, for the mutation accessors.
    ///
    /// # Panics
    ///
    /// Panics on a rope-backed store: mutation requires thawing into a full
    /// store first, and a missed thaw must be loud rather than corrupting.
    fn full_mut(&mut self) -> &mut Vec<BufferLine> {
        match self {
            Self::Full(lines) => lines,
            #[cfg(feature = "rope-buffer")]
            Self::Rope(_) => {
                panic!("rope store mutated without thaw — Editor must call ensure_editable first")
            }
        }
    }
}

/// A buffer of text that is shaped and laid out
#[derive(Debug)]
pub struct Buffer {
    /// Line storage — [`BufferLine`]s (or paragraphs) of text in the buffer
    store: LineStore,
    metrics: Metrics,
    width_opt: Option<f32>,
    height_opt: Option<f32>,
    scroll: Scroll,
    /// True if a redraw is requires. Set to false after processing
    redraw: bool,
    wrap: Wrap,
    ellipsize: Ellipsize,
    monospace_width: Option<f32>,
    tab_width: u16,
    hinting: Hinting,
    direction: Direction,
    /// Dirty flags tracking which properties changed since last layout
    dirty: DirtyFlags,
    /// Cap on rope-store size for thaw-on-write; see [`Buffer::ensure_editable`]
    max_thaw_bytes: Option<usize>,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            store: match &self.store {
                LineStore::Full(lines) => LineStore::Full(lines.clone()),
                // Rope clones share the rope cheaply and start with cold caches.
                #[cfg(feature = "rope-buffer")]
                LineStore::Rope(store) => LineStore::Rope(store.clone()),
            },
            metrics: self.metrics,
            width_opt: self.width_opt,
            height_opt: self.height_opt,
            scroll: self.scroll,
            redraw: self.redraw,
            wrap: self.wrap,
            ellipsize: self.ellipsize,
            monospace_width: self.monospace_width,
            tab_width: self.tab_width,
            hinting: self.hinting,
            direction: self.direction,
            dirty: self.dirty,
            max_thaw_bytes: self.max_thaw_bytes,
        }
    }
}

impl Buffer {
    /// Create an empty [`Buffer`] with the provided [`Metrics`].
    /// This is useful for initializing a [`Buffer`] without a [`FontSystem`].
    ///
    /// You must populate the [`Buffer`] with at least one [`BufferLine`] before shaping and layout,
    /// for example by calling [`Buffer::set_text`].
    ///
    /// If you have a [`FontSystem`] in scope, you should use [`Buffer::new`] instead.
    ///
    /// # Panics
    ///
    /// Will panic if `metrics.line_height` is zero.
    pub fn new_empty(metrics: Metrics) -> Self {
        assert_ne!(metrics.line_height, 0.0, "line height cannot be 0");
        Self {
            store: LineStore::Full(Vec::new()),
            metrics,
            width_opt: None,
            height_opt: None,
            scroll: Scroll::default(),
            redraw: false,
            wrap: Wrap::WordOrGlyph,
            ellipsize: Ellipsize::None,
            monospace_width: None,
            tab_width: 8,
            hinting: Hinting::default(),
            direction: Direction::default(),
            dirty: DirtyFlags::empty(),
            max_thaw_bytes: Some(1 << 30),
        }
    }

    /// Create a new [`Buffer`] with the provided [`FontSystem`] and [`Metrics`]
    ///
    /// # Panics
    ///
    /// Will panic if `metrics.line_height` is zero.
    pub fn new(font_system: &mut FontSystem, metrics: Metrics) -> Self {
        let mut buffer = Self::new_empty(metrics);
        buffer.set_text("", &Attrs::new(), Shaping::Advanced, None);
        buffer.shape_until_scroll(font_system, false);
        buffer
    }

    /// A buffer whose lines are materialized on demand from a rope store.
    ///
    /// # Panics
    ///
    /// Will panic if `metrics.line_height` is zero.
    #[cfg(feature = "rope-buffer")]
    pub fn new_rope(metrics: Metrics, store: crate::RopeStore) -> Self {
        let mut buffer = Self::new_empty(metrics);
        buffer.store = LineStore::Rope(store);
        // A fresh store has nothing shaped; mark it like set_text so the
        // first shape_until_scroll actually shapes the visible region.
        buffer.dirty |= DirtyFlags::TEXT_SET;
        buffer.redraw = true;
        buffer
    }

    /// Whether this buffer is backed by a rope store.
    #[cfg(feature = "rope-buffer")]
    pub fn is_rope(&self) -> bool {
        matches!(self.store, LineStore::Rope(_))
    }

    /// Internal, feature-independent spelling of [`Buffer::is_rope`].
    #[inline]
    fn store_is_rope(&self) -> bool {
        match self.store {
            LineStore::Full(_) => false,
            #[cfg(feature = "rope-buffer")]
            LineStore::Rope(_) => true,
        }
    }

    /// Set the maximum rope-store size, in bytes, that [`Buffer::ensure_editable`]
    /// will thaw into a full line vector. `None` removes the cap. Defaults to
    /// 1 GiB.
    pub fn set_max_thaw_bytes(&mut self, max: Option<usize>) {
        self.max_thaw_bytes = max;
    }

    /// Make the buffer mutable. A rope-backed buffer is thawed into a full
    /// line vector (measured: ~317 ms + ~1.1 GB RSS for a 236 MB file on the
    /// reference machine — a one-time cost on first edit). Returns false if
    /// the store exceeds the thaw cap; the caller must treat the buffer as
    /// read-only in that case.
    pub fn ensure_editable(&mut self) -> bool {
        #[cfg(feature = "rope-buffer")]
        {
            // Borrow-check friendly: inspect first, replace after the borrow ends.
            let over_cap = match &self.store {
                LineStore::Rope(store) => match self.max_thaw_bytes {
                    Some(max) => store.text_len_bytes() > max,
                    None => false,
                },
                LineStore::Full(_) => return true,
            };
            if over_cap {
                return false;
            }
            let old = core::mem::replace(&mut self.store, LineStore::Full(Vec::new()));
            if let LineStore::Rope(store) = old {
                self.store = LineStore::Full(store.thaw());
            } else {
                self.store = old; // unreachable in practice; restore defensively
            }
            return true;
        }
        #[cfg(not(feature = "rope-buffer"))]
        true
    }

    /// Number of lines in the buffer.
    #[inline]
    pub fn line_count(&self) -> usize {
        match &self.store {
            LineStore::Full(lines) => lines.len(),
            #[cfg(feature = "rope-buffer")]
            LineStore::Rope(store) => store.line_count(),
        }
    }

    /// Get the line at the given index.
    ///
    /// For rope-backed buffers this is a materialized-cache hit only: it is
    /// warm only inside the shaped region (kept warm by
    /// [`Buffer::shape_until_scroll`] and friends) and returns `None` for
    /// cold lines. Reads that only need text and must work anywhere in the
    /// file use [`Buffer::line_text_cow`] instead.
    #[inline]
    pub fn line(&self, i: usize) -> Option<&BufferLine> {
        match &self.store {
            LineStore::Full(lines) => lines.get(i),
            #[cfg(feature = "rope-buffer")]
            LineStore::Rope(store) => store.materialized(i),
        }
    }

    /// Text of line `i`, without its line ending.
    ///
    /// Works on both storage arms anywhere in the file: borrowed from the
    /// line for full buffers, read out of the rope for rope-backed ones.
    /// This is the accessor for pure-text reads (cursor motion, selection,
    /// copy, search) that must not depend on the shaped region.
    pub fn line_text_cow(&self, i: usize) -> Option<Cow<'_, str>> {
        match &self.store {
            LineStore::Full(lines) => lines.get(i).map(|line| Cow::Borrowed(line.text())),
            #[cfg(feature = "rope-buffer")]
            LineStore::Rope(store) => store.line_text(i),
        }
    }

    /// Mutably get the line at the given index.
    #[inline]
    pub fn line_mut(&mut self, i: usize) -> Option<&mut BufferLine> {
        self.store.full_mut().get_mut(i)
    }

    /// Mutably get the last line.
    #[inline]
    pub fn last_line_mut(&mut self) -> Option<&mut BufferLine> {
        self.store.full_mut().last_mut()
    }

    /// Iterate over the lines in the buffer.
    ///
    /// For rope-backed buffers this yields only the warm (materialized)
    /// lines; callers needing the full text must use explicit text APIs
    /// such as [`Buffer::line_text_cow`].
    pub fn lines_iter(&self) -> impl Iterator<Item = &BufferLine> + '_ {
        #[cfg(feature = "rope-buffer")]
        {
            let (full, rope) = match &self.store {
                LineStore::Full(lines) => (Some(lines.iter()), None),
                LineStore::Rope(store) => (
                    None,
                    Some((0..store.line_count()).filter_map(move |i| store.materialized(i))),
                ),
            };
            return full.into_iter().flatten().chain(rope.into_iter().flatten());
        }
        #[cfg(not(feature = "rope-buffer"))]
        {
            let LineStore::Full(lines) = &self.store;
            lines.iter()
        }
    }

    /// Mutably iterate over the lines in the buffer.
    ///
    /// # Panics
    ///
    /// Panics on a rope-backed buffer: mutation requires thawing first.
    pub fn lines_iter_mut(&mut self) -> impl Iterator<Item = &mut BufferLine> + '_ {
        self.store.full_mut().iter_mut()
    }

    /// Append a line to the buffer.
    pub fn push_line(&mut self, line: BufferLine) {
        self.store.full_mut().push(line);
    }

    /// Insert a line at the given index.
    pub fn insert_line(&mut self, i: usize, line: BufferLine) {
        self.store.full_mut().insert(i, line);
    }

    /// Remove and return the line at the given index.
    pub fn remove_line(&mut self, i: usize) -> BufferLine {
        self.store.full_mut().remove(i)
    }

    /// Truncate the buffer to the given number of lines.
    pub fn truncate_lines(&mut self, len: usize) {
        self.store.full_mut().truncate(len);
    }

    /// Remove all lines from the buffer.
    pub fn clear_lines(&mut self) {
        self.store.full_mut().clear();
    }

    /// Whether the buffer has no lines.
    #[inline]
    pub fn lines_is_empty(&self) -> bool {
        match &self.store {
            LineStore::Full(lines) => lines.is_empty(),
            // A rope store always reports at least one line.
            #[cfg(feature = "rope-buffer")]
            LineStore::Rope(store) => store.line_count() == 0,
        }
    }

    /// Reset shaping, layout, and metadata caches for every line, forcing a
    /// full reshape on the next shaping pass. For use when the font
    /// environment changed out from under the buffer (for example, a
    /// different monospace font family was configured); text and attributes
    /// are untouched.
    ///
    /// Works over both storage arms: the full arm resets each line in place,
    /// the rope arm drops the store's shape/layout caches (rope lines hold
    /// no shaping state of their own).
    pub fn reset_shaping(&mut self) {
        match &mut self.store {
            LineStore::Full(lines) => {
                for line in lines.iter_mut() {
                    line.reset();
                }
            }
            #[cfg(feature = "rope-buffer")]
            LineStore::Rope(store) => {
                store.cache.clear();
            }
        }
        // Like set_text: everything is cold and the next pass must shape
        // from scratch. The rope arm of resolve_dirty never scans lines, so
        // without a dirty flag the reshape would not happen at all.
        self.dirty |= DirtyFlags::TEXT_SET;
        self.redraw = true;
    }

    /// Get the text of a line by index.
    ///
    /// For rope-backed buffers this is warm-only (see [`Buffer::line`]);
    /// use [`Buffer::line_text_cow`] for reads that must work cold.
    #[inline]
    pub fn line_text(&self, line_i: usize) -> Option<&str> {
        self.line(line_i).map(|l| l.text())
    }

    /// Mutably borrows the buffer together with an [`FontSystem`] for more convenient methods
    pub fn borrow_with<'a>(
        &'a mut self,
        font_system: &'a mut FontSystem,
    ) -> BorrowedWithFontSystem<'a, Self> {
        BorrowedWithFontSystem {
            inner: self,
            font_system,
        }
    }

    /// Process dirty flags: invalidate shape/layout caches as needed, then clear flags.
    /// Returns `true` if any flags were set (i.e., work may be needed).
    fn resolve_dirty(&mut self) -> bool {
        let dirty = self.dirty;
        #[cfg(feature = "rope-buffer")]
        if let LineStore::Rope(store) = &mut self.store {
            if dirty.is_empty() {
                // Rope lines cannot be externally invalidated (there is no
                // `line_mut` access without thaw), so no per-line scan.
                return false;
            }
            if !dirty.contains(DirtyFlags::TEXT_SET) {
                if dirty.contains(DirtyFlags::DIRECTION) || dirty.contains(DirtyFlags::TAB_SHAPE) {
                    // Reshape implies relayout: drop both caches. Cheaper than
                    // scanning millions of rope lines for affected ones.
                    store.cache.clear();
                } else if dirty.contains(DirtyFlags::RELAYOUT) {
                    store.cache.clear_layout();
                }
            }
            self.redraw = true;
            self.dirty = DirtyFlags::empty();
            return true;
        }
        if dirty.is_empty() {
            // individual lines may have been externally invalidated
            if self.lines_iter().any(|line| line.needs_reshaping()) {
                self.redraw = true;
                return true;
            }
            return false;
        }

        if dirty.contains(DirtyFlags::TEXT_SET) {
            // Lines were replaced — already fresh, no cache to invalidate.
        } else {
            if dirty.contains(DirtyFlags::DIRECTION) {
                for line in self.lines_iter_mut() {
                    if line.shape_opt().is_some() {
                        line.reset_shaping();
                    }
                }
            } else if dirty.contains(DirtyFlags::TAB_SHAPE) {
                for line in self.lines_iter_mut() {
                    if line.shape_opt().is_some() && line.text().contains('\t') {
                        line.reset_shaping();
                    }
                }
            }
            if dirty.contains(DirtyFlags::RELAYOUT) {
                for line in self.lines_iter_mut() {
                    if line.shape_opt().is_some() {
                        line.reset_layout();
                    }
                }
            }
        }

        self.redraw = true;
        self.dirty = DirtyFlags::empty();
        true
    }

    /// Shape lines until cursor, also scrolling to include cursor in view
    #[allow(clippy::missing_panics_doc)]
    pub fn shape_until_cursor(
        &mut self,
        font_system: &mut FontSystem,
        cursor: Cursor,
        prune: bool,
    ) {
        self.shape_until_scroll(font_system, prune);
        let metrics = self.metrics;
        let old_scroll = self.scroll;

        let layout_cursor = self
            .layout_cursor(font_system, cursor)
            .expect("shape_until_cursor invalid cursor");

        let mut layout_y = 0.0;
        let mut total_height = {
            let layout = self
                .line_layout(font_system, layout_cursor.line)
                .expect("shape_until_cursor failed to scroll forwards");
            (0..layout_cursor.layout).for_each(|layout_i| {
                layout_y += layout[layout_i]
                    .line_height_opt
                    .unwrap_or(metrics.line_height);
            });
            layout_y
                + layout[layout_cursor.layout]
                    .line_height_opt
                    .unwrap_or(metrics.line_height)
        };

        if self.scroll.line > layout_cursor.line
            || (self.scroll.line == layout_cursor.line && self.scroll.vertical > layout_y)
        {
            // Adjust scroll backwards if cursor is before it
            self.scroll.line = layout_cursor.line;
            self.scroll.vertical = layout_y;
        } else if let Some(height) = self.height_opt {
            // Adjust scroll forwards if cursor is after it
            let mut line_i = layout_cursor.line;
            if line_i <= self.scroll.line {
                // This is a single line that may wrap
                if total_height > height + self.scroll.vertical {
                    self.scroll.vertical = total_height - height;
                }
            } else {
                while line_i > self.scroll.line {
                    line_i -= 1;
                    let layout = self
                        .line_layout(font_system, line_i)
                        .expect("shape_until_cursor failed to scroll forwards");
                    for layout_line in layout {
                        total_height += layout_line.line_height_opt.unwrap_or(metrics.line_height);
                    }
                    if total_height > height + self.scroll.vertical {
                        self.scroll.line = line_i;
                        self.scroll.vertical = total_height - height;
                    }
                }
            }
        }

        if old_scroll != self.scroll {
            self.dirty |= DirtyFlags::SCROLL;
        }

        self.shape_until_scroll(font_system, prune);

        // Adjust horizontal scroll to include cursor
        if let Some(layout_cursor) = self.layout_cursor(font_system, cursor) {
            if let Some(layout_lines) = self.line_layout(font_system, layout_cursor.line) {
                if let Some(layout_line) = layout_lines.get(layout_cursor.layout) {
                    let (x_min, x_max) = layout_line
                        .glyphs
                        .get(layout_cursor.glyph)
                        .or_else(|| layout_line.glyphs.last())
                        .map_or((0.0, 0.0), |glyph| {
                            //TODO: use code from cursor_glyph_opt?
                            let x_a = glyph.x;
                            let x_b = glyph.x + glyph.w;
                            (x_a.min(x_b), x_a.max(x_b))
                        });
                    if x_min < self.scroll.horizontal {
                        self.scroll.horizontal = x_min;
                        self.redraw = true;
                    }
                    if let Some(width) = self.width_opt {
                        if x_max > self.scroll.horizontal + width {
                            self.scroll.horizontal = x_max - width;
                            self.redraw = true;
                        }
                    }
                }
            }
        }
    }

    /// Shape lines until scroll, resolving any pending dirty state first.
    ///
    /// This processes dirty flags (invalidating caches for lines that need
    /// reshaping or relayout) and then shapes/layouts visible lines.
    ///
    /// Call this before reading layout results via [`layout_runs`] or [`hit`]
    /// when working with the `Buffer` directly. The [`BorrowedWithFontSystem`]
    /// wrapper calls this automatically.
    ///
    /// [`layout_runs`]: Self::layout_runs
    /// [`hit`]: Self::hit
    #[allow(clippy::missing_panics_doc)]
    pub fn shape_until_scroll(&mut self, font_system: &mut FontSystem, prune: bool) {
        if !self.resolve_dirty() {
            return;
        }
        let metrics = self.metrics;
        // The rope arm differs from the full arm in three ways, all below:
        // pruning is a no-op (its LRU caches bound memory on their own), an
        // unset height is capped at ~50 lines instead of shaping the whole
        // file, and the end-of-buffer scroll-up adjustment only applies when
        // a height is actually set.
        let is_rope = self.store_is_rope();

        // Clamp scroll.line to valid range (lines may have been removed by editing)
        if self.scroll.line >= self.line_count() {
            self.scroll.line = self.line_count().saturating_sub(1);
            self.scroll.vertical = 0.0;
        }

        let old_scroll = self.scroll;

        loop {
            // Adjust scroll.layout to be positive by moving scroll.line backwards
            while self.scroll.vertical < 0.0 {
                if self.scroll.line > 0 {
                    let line_i = self.scroll.line - 1;
                    if let Some(layout) = self.line_layout(font_system, line_i) {
                        let mut layout_height = 0.0;
                        for layout_line in layout {
                            layout_height +=
                                layout_line.line_height_opt.unwrap_or(metrics.line_height);
                        }
                        self.scroll.line = line_i;
                        self.scroll.vertical += layout_height;
                    } else {
                        // If layout is missing, just assume line height
                        self.scroll.line = line_i;
                        self.scroll.vertical += metrics.line_height;
                    }
                } else {
                    self.scroll.vertical = 0.0;
                    break;
                }
            }

            let scroll_start = self.scroll.vertical;
            let default_height = if is_rope {
                // A rope store with unbounded height would materialize the
                // whole file; cap the shaped run instead.
                metrics.line_height * 50.0
            } else {
                f32::INFINITY
            };
            let scroll_end = scroll_start + self.height_opt.unwrap_or(default_height);

            if prune && !is_rope {
                for line_i in 0..self.scroll.line {
                    self.line_mut(line_i)
                        .expect("line index in bounds")
                        .reset_shaping();
                }
            }
            let mut total_height = 0.0;
            for line_i in self.scroll.line..self.line_count() {
                if total_height > scroll_end {
                    if prune && !is_rope {
                        self.line_mut(line_i)
                            .expect("line index in bounds")
                            .reset_shaping();
                        continue;
                    }
                    break;
                }

                let mut layout_height = 0.0;
                let layout = self
                    .line_layout(font_system, line_i)
                    .expect("shape_until_scroll invalid line");
                for layout_line in layout {
                    let line_height = layout_line.line_height_opt.unwrap_or(metrics.line_height);
                    layout_height += line_height;
                    total_height += line_height;
                }

                // Adjust scroll.vertical to be smaller by moving scroll.line forwards
                if line_i == self.scroll.line && layout_height <= self.scroll.vertical {
                    self.scroll.line += 1;
                    self.scroll.vertical -= layout_height;
                }
            }

            if total_height < scroll_end
                && self.scroll.line > 0
                && (!is_rope || self.height_opt.is_some())
            {
                // Need to scroll up to stay inside of buffer
                self.scroll.vertical -= scroll_end - total_height;
            } else {
                // Done adjusting scroll
                break;
            }
        }

        if old_scroll != self.scroll {
            self.redraw = true;
        }
    }

    /// Convert a [`Cursor`] to a [`LayoutCursor`]
    pub fn layout_cursor(
        &mut self,
        font_system: &mut FontSystem,
        cursor: Cursor,
    ) -> Option<LayoutCursor> {
        let layout = self.line_layout(font_system, cursor.line)?;
        for (layout_i, layout_line) in layout.iter().enumerate() {
            for (glyph_i, glyph) in layout_line.glyphs.iter().enumerate() {
                let cursor_end =
                    Cursor::new_with_affinity(cursor.line, glyph.end, Affinity::Before);
                let cursor_start =
                    Cursor::new_with_affinity(cursor.line, glyph.start, Affinity::After);
                let (cursor_left, cursor_right) = if glyph.level.is_ltr() {
                    (cursor_start, cursor_end)
                } else {
                    (cursor_end, cursor_start)
                };
                if cursor == cursor_left {
                    return Some(LayoutCursor::new(cursor.line, layout_i, glyph_i));
                }
                if cursor == cursor_right {
                    return Some(LayoutCursor::new(cursor.line, layout_i, glyph_i + 1));
                }
            }
        }

        // Fall back to start of line
        //TODO: should this be the end of the line?
        Some(LayoutCursor::new(cursor.line, 0, 0))
    }

    /// Shape the provided line index and return the result
    pub fn line_shape(
        &mut self,
        font_system: &mut FontSystem,
        line_i: usize,
    ) -> Option<&ShapeLine> {
        #[cfg(feature = "rope-buffer")]
        if self.store_is_rope() {
            return self.rope_line_shape(font_system, line_i);
        }
        let tab_width = self.tab_width;
        let direction = self.direction;
        let line = self.line_mut(line_i)?;
        Some(line.shape(font_system, tab_width, direction))
    }

    /// Rope arm of [`Buffer::line_shape`]: materialize the line, then shape
    /// into the store's cache keyed by absolute line index.
    #[cfg(feature = "rope-buffer")]
    fn rope_line_shape(
        &mut self,
        font_system: &mut FontSystem,
        line_i: usize,
    ) -> Option<&ShapeLine> {
        let tab_width = self.tab_width;
        let direction = self.direction;
        let store = match &mut self.store {
            LineStore::Rope(store) => store,
            LineStore::Full(_) => return None,
        };
        if store.cache.shape.contains(line_i) {
            // Keep the BufferLine warm so `line`/`layout_runs` can see it.
            store.materialize(line_i)?;
        } else {
            let (text, attrs_list, shaping) = {
                let line = store.materialize(line_i)?;
                (
                    line.text().to_string(),
                    line.attrs_list().clone(),
                    line.shaping(),
                )
            };
            let mut shape_line = ShapeLine::empty();
            shape_line.build(
                font_system,
                &text,
                &attrs_list,
                shaping,
                tab_width,
                direction,
            );
            store.cache.shape.insert(line_i, shape_line);
            store.cache.layout.remove(line_i);
        }
        store.cache.shape.get(line_i)
    }

    /// Lay out the provided line index and return the result
    pub fn line_layout(
        &mut self,
        font_system: &mut FontSystem,
        line_i: usize,
    ) -> Option<&[LayoutLine]> {
        #[cfg(feature = "rope-buffer")]
        if self.store_is_rope() {
            return self.rope_line_layout(font_system, line_i);
        }
        let font_size = self.metrics.font_size;
        let width_opt = self.width_opt;
        let wrap = self.wrap;
        let ellipsize = self.ellipsize;
        let monospace_width = self.monospace_width;
        let tab_width = self.tab_width;
        let hinting = self.hinting;
        let direction = self.direction;
        let line = self.line_mut(line_i)?;
        Some(line.layout(
            font_system,
            font_size,
            width_opt,
            wrap,
            ellipsize,
            monospace_width,
            tab_width,
            hinting,
            direction,
        ))
    }

    /// Rope arm of [`Buffer::line_layout`]: lay the cached shape out into the
    /// store's layout cache, keyed by absolute line index.
    #[cfg(feature = "rope-buffer")]
    fn rope_line_layout(
        &mut self,
        font_system: &mut FontSystem,
        line_i: usize,
    ) -> Option<&[LayoutLine]> {
        // Ensure the shape exists (this also warms the materialized line).
        self.rope_line_shape(font_system, line_i)?;
        let font_size = self.metrics.font_size;
        let width_opt = self.width_opt;
        let wrap = self.wrap;
        let ellipsize = self.ellipsize;
        let monospace_width = self.monospace_width;
        let hinting = self.hinting;
        let store = match &mut self.store {
            LineStore::Rope(store) => store,
            LineStore::Full(_) => return None,
        };
        if !store.cache.layout.contains(line_i) {
            let align = store.materialized(line_i).and_then(BufferLine::align);
            let shape = store.cache.shape.get(line_i)?;
            let mut layout = Vec::with_capacity(1);
            shape.layout_to_buffer(
                &mut font_system.shape_buffer,
                font_size,
                width_opt,
                wrap,
                ellipsize,
                align,
                &mut layout,
                monospace_width,
                hinting,
            );
            store.cache.layout.insert(line_i, layout);
        }
        store.cache.layout.get(line_i).map(Vec::as_slice)
    }

    /// Get the current [`Metrics`]
    pub const fn metrics(&self) -> Metrics {
        self.metrics
    }

    /// Set the current [`Metrics`].
    ///
    /// # Panics
    ///
    /// Will panic if `metrics.font_size` is zero.
    pub fn set_metrics(&mut self, metrics: Metrics) {
        if metrics != self.metrics {
            assert_ne!(metrics.font_size, 0.0, "font size cannot be 0");
            assert_ne!(metrics.line_height, 0.0, "line height cannot be 0");
            self.metrics = metrics;
            self.dirty |= DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
    }

    /// Get the current [`Hinting`] strategy.
    pub const fn hinting(&self) -> Hinting {
        self.hinting
    }

    /// Set the current [`Hinting`] strategy.
    pub fn set_hinting(&mut self, hinting: Hinting) {
        if hinting != self.hinting {
            self.hinting = hinting;
            self.dirty |= DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
    }

    /// Get the current [`Wrap`]
    pub const fn wrap(&self) -> Wrap {
        self.wrap
    }

    /// Set the current [`Wrap`].
    pub fn set_wrap(&mut self, wrap: Wrap) {
        if wrap != self.wrap {
            self.wrap = wrap;
            self.dirty |= DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
    }

    /// Get the current [`Ellipsize`]
    pub const fn ellipsize(&self) -> Ellipsize {
        self.ellipsize
    }

    /// Set the current [`Ellipsize`].
    pub fn set_ellipsize(&mut self, ellipsize: Ellipsize) {
        if ellipsize != self.ellipsize {
            self.ellipsize = ellipsize;
            self.dirty |= DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
    }

    /// Get the current `monospace_width`
    pub const fn monospace_width(&self) -> Option<f32> {
        self.monospace_width
    }

    /// Set monospace width monospace glyphs should be resized to match. `None` means don't resize.
    pub fn set_monospace_width(&mut self, monospace_width: Option<f32>) {
        if monospace_width != self.monospace_width {
            self.monospace_width = monospace_width;
            self.dirty |= DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
    }

    /// Get the current `tab_width`
    pub const fn tab_width(&self) -> u16 {
        self.tab_width
    }

    /// Set tab width (number of spaces between tab stops).
    pub fn set_tab_width(&mut self, tab_width: u16) {
        if tab_width == 0 {
            return;
        }
        if tab_width != self.tab_width {
            self.tab_width = tab_width;
            self.dirty |= DirtyFlags::TAB_SHAPE | DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
    }

    /// Get the current base [`Direction`].
    pub const fn direction(&self) -> Direction {
        self.direction
    }

    /// Set the base [`Direction`] used when shaping text.
    ///
    /// [`Direction::Auto`] (the default) detects each paragraph's base direction
    /// from its content. [`Direction::LeftToRight`] and [`Direction::RightToLeft`]
    /// force it for the whole buffer; use them when you know the direction from
    /// context such as the UI locale rather than from the text.
    pub fn set_direction(&mut self, direction: Direction) {
        if direction != self.direction {
            self.direction = direction;
            // DIRECTION reshapes every line, which resets layout as a side effect.
            self.dirty |= DirtyFlags::DIRECTION;
            self.redraw = true;
        }
    }

    /// Get the current buffer dimensions (width, height)
    pub const fn size(&self) -> (Option<f32>, Option<f32>) {
        (self.width_opt, self.height_opt)
    }

    /// Set the current buffer dimensions.
    pub fn set_size(&mut self, width_opt: Option<f32>, height_opt: Option<f32>) {
        let width_clamped = width_opt.map(|v| v.max(0.0));
        let height_clamped = height_opt.map(|v| v.max(0.0));
        if width_clamped != self.width_opt {
            self.width_opt = width_clamped;
            self.dirty |= DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
        if height_clamped != self.height_opt {
            self.height_opt = height_clamped;
            self.dirty |= DirtyFlags::RELAYOUT;
            self.redraw = true;
        }
    }

    /// Set the current [`Metrics`] and buffer dimensions at the same time.
    ///
    /// # Panics
    ///
    /// Will panic if `metrics.font_size` is zero.
    pub fn set_metrics_and_size(
        &mut self,
        metrics: Metrics,
        width_opt: Option<f32>,
        height_opt: Option<f32>,
    ) {
        self.set_metrics(metrics);
        self.set_size(width_opt, height_opt);
    }

    /// Get the current scroll location
    pub const fn scroll(&self) -> Scroll {
        self.scroll
    }

    /// Set the current scroll location
    pub fn set_scroll(&mut self, scroll: Scroll) {
        if scroll != self.scroll {
            self.scroll = scroll;
            self.dirty |= DirtyFlags::SCROLL;
            self.redraw = true;
        }
    }

    /// Internal: set text of buffer, reusing existing line allocations.
    ///
    /// Does NOT call `shape_until_scroll` — the caller is responsible for that.
    fn set_text_impl(
        &mut self,
        text: &str,
        attrs: &Attrs,
        shaping: Shaping,
        alignment: Option<Align>,
    ) {
        let mut line_count = 0;
        for (range, ending) in LineIter::new(text) {
            let line_text = &text[range];
            if line_count < self.line_count() {
                // Reuse existing line: reclaim String/AttrsList allocations
                let line = self.line_mut(line_count).expect("line index in bounds");
                let mut reused_text = line.reclaim_text();
                reused_text.push_str(line_text);
                let reused_attrs = line.reclaim_attrs().reset(attrs);
                line.reset_new(reused_text, ending, reused_attrs, shaping);
            } else {
                self.push_line(BufferLine::new(
                    line_text,
                    ending,
                    AttrsList::new(attrs),
                    shaping,
                ));
            }
            line_count += 1;
        }

        // Ensure there is an ending line with no line ending.
        // When no lines were produced (empty text), unwrap_or_default() returns
        // LineEnding::Lf (the Default), which is != None, so we add an empty line.
        let last_ending = if line_count > 0 {
            self.line(line_count - 1)
                .expect("line index in bounds")
                .ending()
        } else {
            LineEnding::default()
        };
        if last_ending != LineEnding::None {
            if line_count < self.line_count() {
                let line = self.line_mut(line_count).expect("line index in bounds");
                let reused_text = line.reclaim_text();
                let reused_attrs = line.reclaim_attrs().reset(attrs);
                line.reset_new(reused_text, LineEnding::None, reused_attrs, shaping);
            } else {
                self.push_line(BufferLine::new(
                    "",
                    LineEnding::None,
                    AttrsList::new(attrs),
                    shaping,
                ));
            }
            line_count += 1;
        }

        // Discard excess lines now that we have reused as much of the existing allocations as possible.
        self.truncate_lines(line_count);

        if alignment.is_some() {
            self.lines_iter_mut().for_each(|line| {
                line.set_align(alignment);
            });
        }

        self.scroll = Scroll::default();
    }

    /// Set text of buffer, using provided attributes for each line by default.
    pub fn set_text(
        &mut self,
        text: &str,
        attrs: &Attrs,
        shaping: Shaping,
        alignment: Option<Align>,
    ) {
        self.set_text_impl(text, attrs, shaping, alignment);
        self.dirty |= DirtyFlags::TEXT_SET;
        self.redraw = true;
    }

    /// Internal: set rich text of buffer, reusing existing line allocations.
    ///
    /// Does NOT call `shape_until_scroll` — the caller is responsible for that.
    fn set_rich_text_impl<'r, 's, I>(
        &mut self,
        spans: I,
        default_attrs: &Attrs,
        shaping: Shaping,
        alignment: Option<Align>,
    ) where
        I: IntoIterator<Item = (&'s str, Attrs<'r>)>,
    {
        let mut end = 0;
        // TODO: find a way to cache this string and vec for reuse
        let (string, spans_data): (String, Vec<_>) = spans
            .into_iter()
            .map(|(s, attrs)| {
                let start = end;
                end += s.len();
                (s, (attrs, start..end))
            })
            .unzip();

        let mut spans_iter = spans_data.into_iter();
        let mut maybe_span = spans_iter.next();

        // split the string into lines, as ranges
        let string_start = string.as_ptr() as usize;
        let mut lines_iter = BidiParagraphs::new(&string).map(|line: &str| {
            let start = line.as_ptr() as usize - string_start;
            let end = start + line.len();
            start..end
        });
        let mut maybe_line = lines_iter.next();
        //TODO: set this based on information from spans
        let line_ending = LineEnding::default();

        let mut line_count = 0;
        let mut attrs_list = self
            .line_mut(line_count)
            .map_or_else(|| AttrsList::new(&Attrs::new()), BufferLine::reclaim_attrs)
            .reset(default_attrs);
        let mut line_string = self
            .line_mut(line_count)
            .map(BufferLine::reclaim_text)
            .unwrap_or_default();

        loop {
            let (Some(line_range), Some((attrs, span_range))) = (&maybe_line, &maybe_span) else {
                // this is reached only if this text is empty
                if self.line_count() == line_count {
                    self.push_line(BufferLine::empty());
                }
                self.line_mut(line_count)
                    .expect("line index in bounds")
                    .reset_new(
                        String::new(),
                        line_ending,
                        AttrsList::new(default_attrs),
                        shaping,
                    );
                line_count += 1;
                break;
            };

            // start..end is the intersection of this line and this span
            let start = line_range.start.max(span_range.start);
            let end = line_range.end.min(span_range.end);
            if start < end {
                let text = &string[start..end];
                let text_start = line_string.len();
                line_string.push_str(text);
                let text_end = line_string.len();
                // Only add attrs if they don't match the defaults
                if *attrs != attrs_list.defaults() {
                    attrs_list.add_span(text_start..text_end, attrs);
                }
            } else if line_string.is_empty() && attrs.metrics_opt.is_some() {
                // reset the attrs list with the span's attrs so the line height
                // matches the span's font size rather than falling back to
                // the buffer default
                attrs_list = attrs_list.reset(attrs);
            }

            // we know that at the end of a line,
            // span text's end index is always >= line text's end index
            // so if this span ends before this line ends,
            // there is another span in this line.
            // otherwise, we move on to the next line.
            if span_range.end < line_range.end {
                maybe_span = spans_iter.next();
            } else {
                maybe_line = lines_iter.next();
                if maybe_line.is_some() {
                    // finalize this line and start a new line
                    let next_attrs_list = self
                        .line_mut(line_count + 1)
                        .map_or_else(|| AttrsList::new(&Attrs::new()), BufferLine::reclaim_attrs)
                        .reset(default_attrs);
                    let next_line_string = self
                        .line_mut(line_count + 1)
                        .map(BufferLine::reclaim_text)
                        .unwrap_or_default();
                    let prev_attrs_list = core::mem::replace(&mut attrs_list, next_attrs_list);
                    let prev_line_string = core::mem::replace(&mut line_string, next_line_string);
                    if self.line_count() == line_count {
                        self.push_line(BufferLine::empty());
                    }
                    self.line_mut(line_count)
                        .expect("line index in bounds")
                        .reset_new(prev_line_string, line_ending, prev_attrs_list, shaping);
                    line_count += 1;
                } else {
                    // finalize the final line
                    if self.line_count() == line_count {
                        self.push_line(BufferLine::empty());
                    }
                    self.line_mut(line_count)
                        .expect("line index in bounds")
                        .reset_new(line_string, line_ending, attrs_list, shaping);
                    line_count += 1;
                    break;
                }
            }
        }

        // Discard excess lines now that we have reused as much of the existing allocations as possible.
        self.truncate_lines(line_count);

        self.lines_iter_mut().for_each(|line| {
            line.set_align(alignment);
        });

        self.scroll = Scroll::default();
    }

    /// Set text of buffer, using an iterator of styled spans (pairs of text and attributes).
    ///
    /// ```
    /// # use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping};
    /// # let mut font_system = FontSystem::new();
    /// let mut buffer = Buffer::new_empty(Metrics::new(32.0, 44.0));
    /// let attrs = Attrs::new().family(Family::Serif);
    /// buffer.set_rich_text(
    ///     [
    ///         ("hello, ", attrs.clone()),
    ///         ("cosmic\ntext", attrs.clone().family(Family::Monospace)),
    ///     ],
    ///     &attrs,
    ///     Shaping::Advanced,
    ///     None,
    /// );
    /// ```
    pub fn set_rich_text<'r, 's, I>(
        &mut self,
        spans: I,
        default_attrs: &Attrs,
        shaping: Shaping,
        alignment: Option<Align>,
    ) where
        I: IntoIterator<Item = (&'s str, Attrs<'r>)>,
    {
        self.set_rich_text_impl(spans, default_attrs, shaping, alignment);
        self.dirty |= DirtyFlags::TEXT_SET;
        self.redraw = true;
    }

    /// True if a redraw is needed
    pub const fn redraw(&self) -> bool {
        self.redraw
    }

    /// Set redraw needed flag
    pub fn set_redraw(&mut self, redraw: bool) {
        self.redraw = redraw;
    }

    /// Get the visible layout runs for rendering and other tasks.
    ///
    /// This returns an iterator over the laid-out runs that are visible in the
    /// current scroll region. Call [`shape_until_scroll`] first to ensure the buffer
    /// is up to date, or use [`BorrowedWithFontSystem`] which calls it
    /// automatically.
    ///
    /// [`shape_until_scroll`]: Self::shape_until_scroll
    pub fn layout_runs(&self) -> LayoutRunIter<'_> {
        LayoutRunIter::new(self)
    }

    /// Convert x, y position to Cursor (hit detection).
    ///
    /// Call [`shape_until_scroll`] first to ensure the buffer is up to date,
    /// or use [`BorrowedWithFontSystem`] which calls it automatically.
    ///
    /// [`shape_until_scroll`]: Self::shape_until_scroll
    pub fn hit(&self, x: f32, y: f32) -> Option<Cursor> {
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        let instant = std::time::Instant::now();

        let mut new_cursor_opt = None;

        let mut runs = self.layout_runs().peekable();
        let mut first_run = true;
        while let Some(run) = runs.next() {
            let line_top = run.line_top;
            let line_height = run.line_height;

            if first_run && y < line_top {
                first_run = false;
                let new_cursor = Cursor::new(run.line_i, 0);
                new_cursor_opt = Some(new_cursor);
            } else if y >= line_top && y < line_top + line_height {
                let mut new_cursor_glyph = run.glyphs.len();
                let mut new_cursor_char = 0;
                let mut new_cursor_affinity = Affinity::After;

                let mut first_glyph = true;

                'hit: for (glyph_i, glyph) in run.glyphs.iter().enumerate() {
                    if first_glyph {
                        first_glyph = false;
                        if (run.rtl && x > glyph.x) || (!run.rtl && x < 0.0) {
                            new_cursor_glyph = 0;
                            new_cursor_char = 0;
                        }
                    }
                    if x >= glyph.x && x <= glyph.x + glyph.w {
                        new_cursor_glyph = glyph_i;

                        let cluster = &run.text[glyph.start..glyph.end];
                        let total = cluster.grapheme_indices(true).count();
                        let mut egc_x = glyph.x;
                        let egc_w = glyph.w / (total as f32);
                        for (egc_i, egc) in cluster.grapheme_indices(true) {
                            if x >= egc_x && x <= egc_x + egc_w {
                                new_cursor_char = egc_i;

                                let right_half = x >= egc_x + egc_w / 2.0;
                                if right_half != glyph.level.is_rtl() {
                                    // If clicking on last half of glyph, move cursor past glyph
                                    new_cursor_char += egc.len();
                                    new_cursor_affinity = Affinity::Before;
                                }
                                break 'hit;
                            }
                            egc_x += egc_w;
                        }

                        let right_half = x >= glyph.x + glyph.w / 2.0;
                        if right_half != glyph.level.is_rtl() {
                            // If clicking on last half of glyph, move cursor past glyph
                            new_cursor_char = cluster.len();
                            new_cursor_affinity = Affinity::Before;
                        }
                        break 'hit;
                    }
                }

                let mut new_cursor = Cursor::new(run.line_i, 0);

                match run.glyphs.get(new_cursor_glyph) {
                    Some(glyph) => {
                        // Position at glyph
                        new_cursor.index = glyph.start + new_cursor_char;
                        new_cursor.affinity = new_cursor_affinity;
                    }
                    None => {
                        // Click was past all glyphs in this visual run.
                        // Use the maximum glyph.end across all glyphs.
                        // this is the logical end of this visual line's byte coverage,
                        // correct for LTR, RTL, mixed-BiDi, and wrapped paragraphs.
                        let run_end = run.glyphs.iter().map(|g| g.end).max().unwrap_or(0);
                        new_cursor.index = run_end;
                        new_cursor.affinity = Affinity::Before;
                    }
                }

                new_cursor_opt = Some(new_cursor);

                break;
            } else if runs.peek().is_none() && y > run.line_y {
                // Click below the last run: place cursor at the logical end of the
                // line, regardless of paragraph direction or BiDi mixing.
                let new_cursor =
                    Cursor::new_with_affinity(run.line_i, run.text.len(), Affinity::Before);
                new_cursor_opt = Some(new_cursor);
            }
        }

        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        log::trace!("click({}, {}): {:?}", x, y, instant.elapsed());

        new_cursor_opt
    }

    /// Returns the visual (x, y) position of a cursor within the buffer.
    /// y is the top of the line containing the cursor.
    /// This is a convenience wrapper around [`LayoutRun::cursor_position`].
    pub fn cursor_position(&self, cursor: &Cursor) -> Option<(f32, f32)> {
        self.layout_runs()
            .filter(|run| run.line_i == cursor.line)
            .find_map(|run| run.cursor_position(cursor).map(|x| (x, run.line_top)))
    }

    /// Returns if the text direction for a given line is RTL
    /// Returns `None` if the line doesn't exist or hasn't been shaped yet.
    pub fn is_rtl(&self, line: usize) -> Option<bool> {
        #[cfg(feature = "rope-buffer")]
        if let LineStore::Rope(store) = &self.store {
            return store.cache.shape.peek(line).map(|shape| shape.rtl);
        }
        self.line(line)?.shape_opt().map(|shape| shape.rtl)
    }

    /// Apply a [`Motion`] to a [`Cursor`]
    pub fn cursor_motion(
        &mut self,
        font_system: &mut FontSystem,
        mut cursor: Cursor,
        mut cursor_x_opt: Option<i32>,
        motion: Motion,
    ) -> Option<(Cursor, Option<i32>)> {
        match motion {
            Motion::LayoutCursor(layout_cursor) => {
                let layout = self.line_layout(font_system, layout_cursor.line)?;

                let layout_line = match layout.get(layout_cursor.layout) {
                    Some(some) => some,
                    None => match layout.last() {
                        Some(some) => some,
                        None => {
                            return None;
                        }
                    },
                };

                let (new_index, new_affinity) =
                    layout_line.glyphs.get(layout_cursor.glyph).map_or_else(
                        || {
                            layout_line
                                .glyphs
                                .last()
                                .map_or((0, Affinity::After), |glyph| (glyph.end, Affinity::Before))
                        },
                        |glyph| (glyph.start, Affinity::After),
                    );

                if cursor.line != layout_cursor.line
                    || cursor.index != new_index
                    || cursor.affinity != new_affinity
                {
                    cursor.line = layout_cursor.line;
                    cursor.index = new_index;
                    cursor.affinity = new_affinity;
                }
            }
            Motion::Previous => {
                let text = self.line_text_cow(cursor.line)?;
                if cursor.index > 0 {
                    // Find previous character index
                    let mut prev_index = 0;
                    for (i, _) in text.grapheme_indices(true) {
                        if i < cursor.index {
                            prev_index = i;
                        } else {
                            break;
                        }
                    }

                    cursor.index = prev_index;
                    cursor.affinity = Affinity::After;
                } else if cursor.line > 0 {
                    cursor.line -= 1;
                    cursor.index = self.line_text_cow(cursor.line)?.len();
                    cursor.affinity = Affinity::After;
                }
                cursor_x_opt = None;
            }
            Motion::Next => {
                let text = self.line_text_cow(cursor.line)?;
                if cursor.index < text.len() {
                    for (i, c) in text.grapheme_indices(true) {
                        if i == cursor.index {
                            cursor.index += c.len();
                            cursor.affinity = Affinity::Before;
                            break;
                        }
                    }
                } else if cursor.line + 1 < self.line_count() {
                    cursor.line += 1;
                    cursor.index = 0;
                    cursor.affinity = Affinity::Before;
                }
                cursor_x_opt = None;
            }
            Motion::Left => {
                let rtl_opt = self
                    .line_shape(font_system, cursor.line)
                    .map(|shape| shape.rtl);
                if let Some(rtl) = rtl_opt {
                    if rtl {
                        (cursor, cursor_x_opt) =
                            self.cursor_motion(font_system, cursor, cursor_x_opt, Motion::Next)?;
                    } else {
                        (cursor, cursor_x_opt) = self.cursor_motion(
                            font_system,
                            cursor,
                            cursor_x_opt,
                            Motion::Previous,
                        )?;
                    }
                }
            }
            Motion::Right => {
                let rtl_opt = self
                    .line_shape(font_system, cursor.line)
                    .map(|shape| shape.rtl);
                if let Some(rtl) = rtl_opt {
                    if rtl {
                        (cursor, cursor_x_opt) = self.cursor_motion(
                            font_system,
                            cursor,
                            cursor_x_opt,
                            Motion::Previous,
                        )?;
                    } else {
                        (cursor, cursor_x_opt) =
                            self.cursor_motion(font_system, cursor, cursor_x_opt, Motion::Next)?;
                    }
                }
            }
            Motion::Up => {
                let mut layout_cursor = self.layout_cursor(font_system, cursor)?;

                if cursor_x_opt.is_none() {
                    cursor_x_opt = Some(
                        layout_cursor.glyph as i32, //TODO: glyph x position
                    );
                }

                if layout_cursor.layout > 0 {
                    layout_cursor.layout -= 1;
                } else if layout_cursor.line > 0 {
                    layout_cursor.line -= 1;
                    layout_cursor.layout = usize::MAX;
                }

                if let Some(cursor_x) = cursor_x_opt {
                    layout_cursor.glyph = cursor_x as usize; //TODO: glyph x position
                }

                (cursor, cursor_x_opt) = self.cursor_motion(
                    font_system,
                    cursor,
                    cursor_x_opt,
                    Motion::LayoutCursor(layout_cursor),
                )?;
            }
            Motion::Down => {
                let mut layout_cursor = self.layout_cursor(font_system, cursor)?;

                let layout_len = self.line_layout(font_system, layout_cursor.line)?.len();

                if cursor_x_opt.is_none() {
                    cursor_x_opt = Some(
                        layout_cursor.glyph as i32, //TODO: glyph x position
                    );
                }

                if layout_cursor.layout + 1 < layout_len {
                    layout_cursor.layout += 1;
                } else if layout_cursor.line + 1 < self.line_count() {
                    layout_cursor.line += 1;
                    layout_cursor.layout = 0;
                }

                if let Some(cursor_x) = cursor_x_opt {
                    layout_cursor.glyph = cursor_x as usize; //TODO: glyph x position
                }

                (cursor, cursor_x_opt) = self.cursor_motion(
                    font_system,
                    cursor,
                    cursor_x_opt,
                    Motion::LayoutCursor(layout_cursor),
                )?;
            }
            Motion::Home => {
                cursor.index = 0;
                cursor_x_opt = None;
            }
            Motion::SoftHome => {
                let text = self.line_text_cow(cursor.line)?;
                cursor.index = text
                    .char_indices()
                    .find_map(|(i, c)| if c.is_whitespace() { None } else { Some(i) })
                    .unwrap_or(0);
                cursor_x_opt = None;
            }
            Motion::End => {
                cursor.index = self.line_text_cow(cursor.line)?.len();
                cursor_x_opt = None;
            }
            Motion::ParagraphStart => {
                cursor.index = 0;
                cursor_x_opt = None;
            }
            Motion::ParagraphEnd => {
                cursor.index = self.line_text_cow(cursor.line)?.len();
                cursor_x_opt = None;
            }
            Motion::PageUp => {
                if let Some(height) = self.height_opt {
                    (cursor, cursor_x_opt) = self.cursor_motion(
                        font_system,
                        cursor,
                        cursor_x_opt,
                        Motion::Vertical(-height as i32),
                    )?;
                }
            }
            Motion::PageDown => {
                if let Some(height) = self.height_opt {
                    (cursor, cursor_x_opt) = self.cursor_motion(
                        font_system,
                        cursor,
                        cursor_x_opt,
                        Motion::Vertical(height as i32),
                    )?;
                }
            }
            Motion::Vertical(px) => {
                // TODO more efficient, use layout run line height
                let lines = px / self.metrics().line_height as i32;
                match lines.cmp(&0) {
                    cmp::Ordering::Less => {
                        for _ in 0..-lines {
                            (cursor, cursor_x_opt) =
                                self.cursor_motion(font_system, cursor, cursor_x_opt, Motion::Up)?;
                        }
                    }
                    cmp::Ordering::Greater => {
                        for _ in 0..lines {
                            (cursor, cursor_x_opt) = self.cursor_motion(
                                font_system,
                                cursor,
                                cursor_x_opt,
                                Motion::Down,
                            )?;
                        }
                    }
                    cmp::Ordering::Equal => {}
                }
            }
            Motion::PreviousWord => {
                let text = self.line_text_cow(cursor.line)?;
                if cursor.index > 0 {
                    cursor.index = text
                        .unicode_word_indices()
                        .rev()
                        .map(|(i, _)| i)
                        .find(|&i| i < cursor.index)
                        .unwrap_or(0);
                } else if cursor.line > 0 {
                    cursor.line -= 1;
                    cursor.index = self.line_text_cow(cursor.line)?.len();
                }
                cursor_x_opt = None;
            }
            Motion::NextWord => {
                let text = self.line_text_cow(cursor.line)?;
                if cursor.index < text.len() {
                    cursor.index = text
                        .unicode_word_indices()
                        .map(|(i, word)| i + word.len())
                        .find(|&i| i > cursor.index)
                        .unwrap_or_else(|| text.len());
                } else if cursor.line + 1 < self.line_count() {
                    cursor.line += 1;
                    cursor.index = 0;
                }
                cursor_x_opt = None;
            }
            Motion::LeftWord => {
                let rtl_opt = self
                    .line_shape(font_system, cursor.line)
                    .map(|shape| shape.rtl);
                if let Some(rtl) = rtl_opt {
                    if rtl {
                        (cursor, cursor_x_opt) = self.cursor_motion(
                            font_system,
                            cursor,
                            cursor_x_opt,
                            Motion::NextWord,
                        )?;
                    } else {
                        (cursor, cursor_x_opt) = self.cursor_motion(
                            font_system,
                            cursor,
                            cursor_x_opt,
                            Motion::PreviousWord,
                        )?;
                    }
                }
            }
            Motion::RightWord => {
                let rtl_opt = self
                    .line_shape(font_system, cursor.line)
                    .map(|shape| shape.rtl);
                if let Some(rtl) = rtl_opt {
                    if rtl {
                        (cursor, cursor_x_opt) = self.cursor_motion(
                            font_system,
                            cursor,
                            cursor_x_opt,
                            Motion::PreviousWord,
                        )?;
                    } else {
                        (cursor, cursor_x_opt) = self.cursor_motion(
                            font_system,
                            cursor,
                            cursor_x_opt,
                            Motion::NextWord,
                        )?;
                    }
                }
            }
            Motion::BufferStart => {
                cursor.line = 0;
                cursor.index = 0;
                cursor_x_opt = None;
            }
            Motion::BufferEnd => {
                cursor.line = self.line_count().saturating_sub(1);
                cursor.index = self.line_text_cow(cursor.line)?.len();
                cursor_x_opt = None;
            }
            Motion::GotoLine(line) => {
                let mut layout_cursor = self.layout_cursor(font_system, cursor)?;
                layout_cursor.line = line;
                (cursor, cursor_x_opt) = self.cursor_motion(
                    font_system,
                    cursor,
                    cursor_x_opt,
                    Motion::LayoutCursor(layout_cursor),
                )?;
            }
        }
        Some((cursor, cursor_x_opt))
    }

    /// Draw the buffer.
    ///
    /// Automatically resolves any pending dirty state before drawing.
    #[cfg(feature = "swash")]
    pub fn draw<F>(
        &mut self,
        font_system: &mut FontSystem,
        cache: &mut crate::SwashCache,
        color: Color,
        callback: F,
    ) where
        F: FnMut(i32, i32, u32, u32, Color),
    {
        self.shape_until_scroll(font_system, false);
        let mut renderer = crate::LegacyRenderer {
            font_system,
            cache,
            callback,
        };
        for run in self.layout_runs() {
            for glyph in run.glyphs {
                let physical_glyph = glyph.physical((0., run.line_y), 1.0);
                let glyph_color = glyph.color_opt.map_or(color, |some| some);
                renderer.glyph(physical_glyph, glyph_color);
            }
            render_decoration(&mut renderer, &run, color);
        }
    }

    /// Render the buffer using the provided renderer.
    ///
    /// Automatically resolves any pending dirty state before rendering.
    pub fn render<R: Renderer>(
        &mut self,
        font_system: &mut FontSystem,
        renderer: &mut R,
        color: Color,
    ) {
        self.shape_until_scroll(font_system, false);
        for run in self.layout_runs() {
            for glyph in run.glyphs {
                let physical_glyph = glyph.physical((0., run.line_y), 1.0);
                let glyph_color = glyph.color_opt.map_or(color, |some| some);
                renderer.glyph(physical_glyph, glyph_color);
            }
            // draw decorations after glyphs so strikethrough is over the glyphs
            render_decoration(renderer, &run, color);
        }
    }
}

impl BorrowedWithFontSystem<'_, Buffer> {
    /// Shape lines until cursor, also scrolling to include cursor in view
    pub fn shape_until_cursor(&mut self, cursor: Cursor, prune: bool) {
        self.inner
            .shape_until_cursor(self.font_system, cursor, prune);
    }

    /// Shape the provided line index and return the result
    pub fn line_shape(&mut self, line_i: usize) -> Option<&ShapeLine> {
        self.inner.line_shape(self.font_system, line_i)
    }

    /// Lay out the provided line index and return the result
    pub fn line_layout(&mut self, line_i: usize) -> Option<&[LayoutLine]> {
        self.inner.line_layout(self.font_system, line_i)
    }

    /// Set the current [`Metrics`].
    ///
    /// # Panics
    ///
    /// Will panic if `metrics.font_size` is zero.
    pub fn set_metrics(&mut self, metrics: Metrics) {
        self.inner.set_metrics(metrics);
    }

    /// Set the current [`Hinting`] strategy.
    pub fn set_hinting(&mut self, hinting: Hinting) {
        self.inner.set_hinting(hinting);
    }

    /// Set the current [`Wrap`].
    pub fn set_wrap(&mut self, wrap: Wrap) {
        self.inner.set_wrap(wrap);
    }

    /// Set the base [`Direction`] used when shaping text.
    pub fn set_direction(&mut self, direction: Direction) {
        self.inner.set_direction(direction);
    }

    /// Set the current [`Ellipsize`].
    pub fn set_ellipsize(&mut self, ellipsize: Ellipsize) {
        self.inner.set_ellipsize(ellipsize);
    }

    /// Set the current buffer dimensions.
    pub fn set_size(&mut self, width_opt: Option<f32>, height_opt: Option<f32>) {
        self.inner.set_size(width_opt, height_opt);
    }

    /// Set the current [`Metrics`] and buffer dimensions at the same time.
    ///
    /// # Panics
    ///
    /// Will panic if `metrics.font_size` is zero.
    pub fn set_metrics_and_size(
        &mut self,
        metrics: Metrics,
        width_opt: Option<f32>,
        height_opt: Option<f32>,
    ) {
        self.inner
            .set_metrics_and_size(metrics, width_opt, height_opt);
    }

    /// Set tab width (number of spaces between tab stops).
    ///
    /// A `tab_width` of 0 is ignored.
    pub fn set_tab_width(&mut self, tab_width: u16) {
        self.inner.set_tab_width(tab_width);
    }

    /// Set monospace width monospace glyphs should be resized to match. `None` means don't resize.
    pub fn set_monospace_width(&mut self, monospace_width: Option<f32>) {
        self.inner.set_monospace_width(monospace_width);
    }

    /// Set text of buffer, using provided attributes for each line by default.
    pub fn set_text(
        &mut self,
        text: &str,
        attrs: &Attrs,
        shaping: Shaping,
        alignment: Option<Align>,
    ) {
        self.inner.set_text(text, attrs, shaping, alignment);
    }

    /// Set text of buffer, using an iterator of styled spans (pairs of text and attributes).
    pub fn set_rich_text<'r, 's, I>(
        &mut self,
        spans: I,
        default_attrs: &Attrs,
        shaping: Shaping,
        alignment: Option<Align>,
    ) where
        I: IntoIterator<Item = (&'s str, Attrs<'r>)>,
    {
        self.inner
            .set_rich_text(spans, default_attrs, shaping, alignment);
    }

    /// Shape lines until scroll, resolving any pending dirty state first.
    ///
    /// See [`Buffer::shape_until_scroll`].
    pub fn shape_until_scroll(&mut self, prune: bool) {
        self.inner.shape_until_scroll(self.font_system, prune);
    }

    /// Get the visible layout runs for rendering and other tasks.
    ///
    /// Automatically resolves any pending dirty state.
    pub fn layout_runs(&mut self) -> LayoutRunIter<'_> {
        self.inner.shape_until_scroll(self.font_system, false);
        self.inner.layout_runs()
    }

    /// Convert x, y position to Cursor (hit detection).
    ///
    /// Automatically resolves any pending dirty state.
    pub fn hit(&mut self, x: f32, y: f32) -> Option<Cursor> {
        self.inner.shape_until_scroll(self.font_system, false);
        self.inner.hit(x, y)
    }

    /// Apply a [`Motion`] to a [`Cursor`]
    pub fn cursor_motion(
        &mut self,
        cursor: Cursor,
        cursor_x_opt: Option<i32>,
        motion: Motion,
    ) -> Option<(Cursor, Option<i32>)> {
        self.inner
            .cursor_motion(self.font_system, cursor, cursor_x_opt, motion)
    }

    /// Draw the buffer.
    ///
    /// Automatically resolves any pending dirty state.
    #[cfg(feature = "swash")]
    pub fn draw<F>(&mut self, cache: &mut crate::SwashCache, color: Color, f: F)
    where
        F: FnMut(i32, i32, u32, u32, Color),
    {
        self.inner.draw(self.font_system, cache, color, f);
    }
}

#[cfg(all(test, feature = "std"))]
mod long_line_tests {
    use super::{Buffer, Metrics};
    use crate::{Attrs, Shaping, MAX_SHAPE_BYTES};

    /// Regression net for the single-giant-line freeze (Ghost-export JSON:
    /// 12 MB, zero newlines). Shaping cost must be bounded per line no matter
    /// how long the line is — and the cap must never touch the document text.
    #[test]
    fn giant_single_line_shaping_is_capped() {
        let mut font_system = crate::FontSystem::new();
        let unit = r#"{"key":"value","n":12345},"#;
        let text: String = unit.repeat(8 * 1024); // ~208 KB, one line, no newlines
        assert!(text.len() > MAX_SHAPE_BYTES * 6);

        let mut buffer = Buffer::new_empty(Metrics::new(14.0, 20.0));
        buffer.set_size(Some(800.0), Some(600.0));
        buffer.set_text(&text, &Attrs::new(), Shaping::Advanced, None);

        let layout = buffer.line_layout(&mut font_system, 0).expect("layout of line 0");
        let max_end = layout
            .iter()
            .flat_map(|line| line.glyphs.iter())
            .map(|glyph| glyph.end)
            .max()
            .unwrap_or(0);
        assert!(
            max_end <= MAX_SHAPE_BYTES,
            "glyphs extend to byte {max_end}, shaping cap is {MAX_SHAPE_BYTES}"
        );
        // The cap bounds SHAPING only; the document itself stays whole.
        assert_eq!(buffer.line(0).expect("line 0").text().len(), text.len());
    }

    /// The cap must not split a multi-byte character.
    #[test]
    fn shape_cap_snaps_to_char_boundary() {
        let mut font_system = crate::FontSystem::new();
        // 3-byte chars ensure the cap lands mid-char unless snapped.
        let text: String = "€".repeat(MAX_SHAPE_BYTES / 3 + 64);

        let mut buffer = Buffer::new_empty(Metrics::new(14.0, 20.0));
        buffer.set_size(Some(800.0), Some(600.0));
        buffer.set_text(&text, &Attrs::new(), Shaping::Advanced, None);

        // Must not panic on a non-boundary slice.
        let layout = buffer.line_layout(&mut font_system, 0).expect("layout of line 0");
        let max_end = layout
            .iter()
            .flat_map(|line| line.glyphs.iter())
            .map(|glyph| glyph.end)
            .max()
            .unwrap_or(0);
        assert!(max_end <= MAX_SHAPE_BYTES);
        assert_eq!(text.len() % 3, 0);
    }
}

#[cfg(all(test, feature = "rope-buffer", feature = "std"))]
mod rope_arm_tests {
    use super::{Buffer, Metrics};
    use crate::{Attrs, Cursor, Edit, Editor, FontSystem, RopeStore, Shaping};

    fn rope_buffer(lines: usize) -> Buffer {
        let text: String = (0..lines)
            .map(|i| format!("line {i} padding padding\n"))
            .collect();
        let store = RopeStore::from_text(&text, &Attrs::new(), Shaping::Advanced);
        let mut buffer = Buffer::new_rope(Metrics::new(14.0, 20.0), store);
        buffer.set_size(Some(800.0), Some(600.0));
        buffer
    }

    #[test]
    fn rope_accessors_cold_and_warm() {
        let buffer = rope_buffer(10_000);
        assert!(buffer.is_rope());
        // +1: the trailing newline produces a final empty line.
        assert_eq!(buffer.line_count(), 10_001);
        assert!(!buffer.lines_is_empty());
        // Cold: no line materialized yet, but text reads work anywhere.
        assert!(buffer.line(9_999).is_none());
        assert_eq!(
            buffer.line_text_cow(9_999).as_deref(),
            Some("line 9999 padding padding")
        );
        assert!(buffer.line_text_cow(10_001).is_none());
    }

    #[test]
    #[should_panic(expected = "rope store mutated without thaw")]
    fn rope_mutation_without_thaw_is_loud() {
        let mut buffer = rope_buffer(10);
        buffer.push_line(crate::BufferLine::empty());
    }

    #[test]
    fn rope_layout_runs_start_at_scroll() {
        let mut font_system = FontSystem::new();
        let mut buffer = rope_buffer(10_000);
        buffer.shape_until_scroll(&mut font_system, false);
        let line_indices: Vec<usize> = buffer.layout_runs().map(|run| run.line_i).collect();
        assert!(!line_indices.is_empty());
        assert_eq!(line_indices[0], buffer.scroll().line);
        assert_eq!(line_indices[0], 0);
        // Unwrapped short lines: runs walk consecutive absolute indices.
        for pair in line_indices.windows(2) {
            assert_eq!(pair[1], pair[0] + 1);
        }
    }

    #[test]
    fn rope_deep_scroll_keeps_absolute_coordinates() {
        let mut font_system = FontSystem::new();
        let mut buffer = rope_buffer(10_000);
        buffer.shape_until_scroll(&mut font_system, false);

        let mut scroll = buffer.scroll();
        scroll.line = 7_000;
        buffer.set_scroll(scroll);
        buffer.shape_until_scroll(&mut font_system, false);

        let runs: Vec<(usize, String)> = buffer
            .layout_runs()
            .map(|run| (run.line_i, run.text.to_string()))
            .collect();
        assert!(!runs.is_empty());
        // THE absolute-coordinates proof: no window offset anywhere.
        assert_eq!(runs[0].0, 7_000);
        assert_eq!(runs[0].1, "line 7000 padding padding");
        assert!(runs.iter().all(|(line_i, _)| *line_i >= 7_000));

        let cursor = buffer.hit(10.0, 10.0).expect("hit inside shaped region");
        assert!(cursor.line >= 7_000, "hit cursor line {}", cursor.line);
    }

    #[test]
    fn thaw_on_first_edit() {
        let buffer = rope_buffer(1_000);
        assert!(buffer.is_rope());
        let mut editor = Editor::new(buffer);

        editor.insert_at(Cursor::new(500, 0), "hello ", None);

        assert!(
            !editor.with_buffer(|buffer| buffer.is_rope()),
            "first edit must thaw the rope store into a full one"
        );
        assert_eq!(
            editor.with_buffer(|buffer| buffer.line_text_cow(500).map(|cow| cow.into_owned())),
            Some("hello line 500 padding padding".to_string())
        );
        // No newline inserted: line count is unchanged (1000 lines + trailing empty).
        assert_eq!(editor.with_buffer(super::Buffer::line_count), 1_001);
    }

    #[test]
    fn thaw_cap_refuses_edit() {
        let mut buffer = rope_buffer(1_000);
        buffer.set_max_thaw_bytes(Some(10));
        let mut editor = Editor::new(buffer);

        let cursor = editor.insert_at(Cursor::new(500, 0), "hello ", None);

        assert_eq!(cursor, Cursor::new(500, 0), "rejected edit must be a no-op");
        assert!(
            editor.with_buffer(|buffer| buffer.is_rope()),
            "over-cap buffer must stay rope-backed"
        );
        assert_eq!(
            editor.with_buffer(|buffer| buffer.line_text_cow(500).map(|cow| cow.into_owned())),
            Some("line 500 padding padding".to_string())
        );
    }

    #[test]
    fn reset_shaping_reshapes_rope_at_absolute_scroll() {
        let mut font_system = FontSystem::new();
        let mut buffer = rope_buffer(10_000);
        let mut scroll = buffer.scroll();
        scroll.line = 7_000;
        buffer.set_scroll(scroll);
        buffer.shape_until_scroll(&mut font_system, false);
        assert!(buffer.layout_runs().next().is_some());

        // The font-change path (cosmic-edit's DefaultFont): drop all shaping
        // state, then the next shape pass must rebuild — not early-return on
        // clean dirty flags and render nothing.
        buffer.reset_shaping();
        buffer.shape_until_scroll(&mut font_system, false);

        let runs: Vec<usize> = buffer.layout_runs().map(|run| run.line_i).collect();
        assert!(
            !runs.is_empty(),
            "reset_shaping must leave the buffer reshapeable"
        );
        assert_eq!(runs[0], 7_000, "absolute coordinates survive the reset");
    }
}

#[cfg(all(test, feature = "std"))]
mod reset_shaping_full_tests {
    use super::{Buffer, Metrics};
    use crate::{Attrs, FontSystem, Shaping};

    #[test]
    fn reset_shaping_resets_full_lines_and_reshapes() {
        let mut font_system = FontSystem::new();
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(14.0, 20.0));
        buffer.set_size(Some(800.0), Some(600.0));
        buffer.set_text("alpha\nbeta\ngamma", &Attrs::new(), Shaping::Advanced, None);
        buffer.shape_until_scroll(&mut font_system, false);
        assert!(buffer.line(0).expect("line 0").shape_opt().is_some());
        buffer.line_mut(1).expect("line 1").set_metadata(7);

        buffer.reset_shaping();

        assert!(buffer.line(0).expect("line 0").shape_opt().is_none());
        assert_eq!(buffer.line(1).expect("line 1").metadata(), None);

        buffer.shape_until_scroll(&mut font_system, false);
        assert_eq!(buffer.layout_runs().count(), 3);
    }
}
