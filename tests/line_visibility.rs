// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-line visibility: hidden lines keep their text and ending but yield no
//! layout runs and contribute no height.

use cosmic_text::{
    Attrs, AttrsList, Buffer, BufferLine, Cursor, FontSystem, LineEnding, Metrics, Motion, Shaping,
};

// Plain attrs carry no metrics override, so every layout line is exactly
// LINE_HEIGHT tall and the height assertions below are font-independent.
const LINE_HEIGHT: f32 = 20.0;
const METRICS: Metrics = Metrics::new(14.0, LINE_HEIGHT);

fn buffer_with_lines(n: usize, height: Option<f32>) -> (FontSystem, Buffer) {
    let font_system = FontSystem::new();
    let text = (0..n)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut buffer = Buffer::new_empty(METRICS);
    buffer.set_size(Some(800.0), height);
    buffer.set_text(&text, &Attrs::new(), Shaping::Advanced, None);
    (font_system, buffer)
}

fn visible_runs(buffer: &Buffer) -> Vec<usize> {
    buffer.layout_runs().map(|run| run.line_i).collect()
}

#[test]
fn hidden_lines_are_skipped_by_layout_runs() {
    let (mut font_system, mut buffer) = buffer_with_lines(6, Some(600.0));
    assert!(buffer.set_line_hidden(1, true));
    assert!(buffer.set_line_hidden(2, true));
    buffer.shape_until_scroll(&mut font_system, false);

    let runs: Vec<(usize, f32)> = buffer
        .layout_runs()
        .map(|run| (run.line_i, run.line_top))
        .collect();
    let lines: Vec<usize> = runs.iter().map(|(line_i, _)| *line_i).collect();
    assert_eq!(lines, vec![0, 3, 4, 5]);
    // Hidden lines contribute no height: line 3 sits directly below line 0.
    assert_eq!(runs[1].1, LINE_HEIGHT);
    assert_eq!(runs[3].1, 3.0 * LINE_HEIGHT);
}

#[test]
fn hiding_after_shape_marks_dirty_and_reshapes() {
    let (mut font_system, mut buffer) = buffer_with_lines(6, Some(600.0));
    buffer.shape_until_scroll(&mut font_system, false);
    assert_eq!(visible_runs(&buffer), vec![0, 1, 2, 3, 4, 5]);

    buffer.set_redraw(false);
    assert!(buffer.set_line_hidden(2, true));
    assert!(buffer.redraw(), "hiding a line must request a redraw");
    buffer.shape_until_scroll(&mut font_system, false);
    assert_eq!(visible_runs(&buffer), vec![0, 1, 3, 4, 5]);

    // Setting the same value again is a no-op.
    buffer.set_redraw(false);
    assert!(!buffer.set_line_hidden(2, true));
    assert!(!buffer.redraw());

    // Unhiding restores the line without any reshaping loss.
    assert!(buffer.set_line_hidden(2, false));
    buffer.shape_until_scroll(&mut font_system, false);
    assert_eq!(visible_runs(&buffer), vec![0, 1, 2, 3, 4, 5]);
}

/// The viewport must fill with *visible* lines across a hidden gap: the
/// shaping pass has to keep going (and keep shaping) past folded-away
/// lines instead of counting their height against the scroll region.
#[test]
fn viewport_fills_with_visible_lines_across_hidden_gap() {
    let (mut font_system, mut buffer) = buffer_with_lines(20, Some(5.0 * LINE_HEIGHT));
    for line_i in 1..=10 {
        buffer.set_line_hidden(line_i, true);
    }
    buffer.shape_until_scroll(&mut font_system, false);

    let lines = visible_runs(&buffer);
    assert!(
        lines.len() >= 5,
        "viewport of 5 line-heights must fill with 5 visible lines, got {lines:?}"
    );
    assert_eq!(&lines[..5], &[0, 11, 12, 13, 14]);
    assert!(lines.iter().all(|&line_i| line_i == 0 || line_i >= 11));
}

#[test]
fn vertical_motion_lands_on_nearest_visible_line() {
    let (mut font_system, mut buffer) = buffer_with_lines(10, Some(600.0));
    for line_i in 1..=3 {
        buffer.set_line_hidden(line_i, true);
    }
    buffer.shape_until_scroll(&mut font_system, false);

    let (cursor, _) = buffer
        .cursor_motion(&mut font_system, Cursor::new(0, 0), None, Motion::Down)
        .expect("down motion");
    assert_eq!(cursor.line, 4, "Down from line 0 must skip hidden 1..=3");

    let (cursor, _) = buffer
        .cursor_motion(&mut font_system, Cursor::new(4, 0), None, Motion::Up)
        .expect("up motion");
    assert_eq!(cursor.line, 0, "Up from line 4 must skip hidden 1..=3");
}

/// With only hidden lines beyond the edge, vertical motion stays put —
/// the same behavior as pressing Up on the first or Down on the last line.
#[test]
fn vertical_motion_stops_at_hidden_edges() {
    let (mut font_system, mut buffer) = buffer_with_lines(10, Some(600.0));
    for line_i in (0..=2).chain(7..=9) {
        buffer.set_line_hidden(line_i, true);
    }
    buffer.shape_until_scroll(&mut font_system, false);

    let (cursor, _) = buffer
        .cursor_motion(&mut font_system, Cursor::new(3, 0), None, Motion::Up)
        .expect("up motion");
    assert_eq!(cursor.line, 3, "no visible line above line 3");

    let (cursor, _) = buffer
        .cursor_motion(&mut font_system, Cursor::new(6, 0), None, Motion::Down)
        .expect("down motion");
    assert_eq!(cursor.line, 6, "no visible line below line 6");
}

/// PageUp/PageDown run through Motion::Vertical, which loops Up/Down —
/// each step lands on visible lines, so pages cross folds entirely.
#[test]
fn page_motions_skip_hidden_lines() {
    let (mut font_system, mut buffer) = buffer_with_lines(30, Some(5.0 * LINE_HEIGHT));
    for line_i in 2..=25 {
        buffer.set_line_hidden(line_i, true);
    }
    buffer.shape_until_scroll(&mut font_system, false);

    // 5 Down steps: 0 -> 1 -> 26 -> 27 -> 28 -> 29
    let (cursor, _) = buffer
        .cursor_motion(&mut font_system, Cursor::new(0, 0), None, Motion::PageDown)
        .expect("page down");
    assert_eq!(cursor.line, 29);

    // And back: 29 -> 28 -> 27 -> 26 -> 1 -> 0
    let (cursor, _) = buffer
        .cursor_motion(&mut font_system, cursor, None, Motion::PageUp)
        .expect("page up");
    assert_eq!(cursor.line, 0);
}

/// Hidden lines keep their text and endings, so reconstruction stays
/// byte-exact while hidden and after unhiding.
#[test]
fn hidden_lines_keep_text_and_endings_byte_exact() {
    let text = "alpha\r\nbeta\ngamma\r\ndelta\nepsilon";
    let mut buffer = Buffer::new_empty(METRICS);
    buffer.set_text(text, &Attrs::new(), Shaping::Advanced, None);

    let reconstruct = |buffer: &Buffer| -> String {
        let mut out = String::new();
        for i in 0..buffer.line_count() {
            out.push_str(&buffer.line_text_cow(i).expect("line in bounds"));
            out.push_str(buffer.line_ending(i).expect("line in bounds").as_str());
        }
        out
    };

    for line_i in 1..=3 {
        buffer.set_line_hidden(line_i, true);
    }
    assert_eq!(reconstruct(&buffer), text);

    for line_i in 1..=3 {
        buffer.set_line_hidden(line_i, false);
    }
    assert_eq!(reconstruct(&buffer), text);
}

#[test]
fn visible_line_count_excludes_hidden() {
    let (_font_system, mut buffer) = buffer_with_lines(10, Some(600.0));
    assert_eq!(buffer.visible_line_count(), 10);

    for line_i in [2, 5, 8] {
        assert!(buffer.set_line_hidden(line_i, true));
    }
    assert_eq!(buffer.visible_line_count(), 7);
    assert!(buffer.line_hidden(5));
    assert!(!buffer.line_hidden(4));

    // Out of bounds: no-op, reports not hidden.
    assert!(!buffer.set_line_hidden(999, true));
    assert!(!buffer.line_hidden(999));

    assert!(buffer.set_line_hidden(5, false));
    assert_eq!(buffer.visible_line_count(), 8);
}

/// set_text is wholesale content replacement: stale hidden flags must not
/// survive line-allocation reuse and hide fresh, unrelated content.
#[test]
fn set_text_clears_hidden_on_reused_lines() {
    let (_font_system, mut buffer) = buffer_with_lines(5, Some(600.0));
    for line_i in 1..=3 {
        buffer.set_line_hidden(line_i, true);
    }
    assert_eq!(buffer.visible_line_count(), 2);

    buffer.set_text("a\nb\nc\nd\ne", &Attrs::new(), Shaping::Advanced, None);
    assert_eq!(buffer.visible_line_count(), buffer.line_count());
    for line_i in 0..buffer.line_count() {
        assert!(!buffer.line_hidden(line_i), "line {line_i} leaked hidden");
    }
}

/// Degenerate but must not hang: every line hidden. The shape pass has
/// to settle (no infinite scroll adjustment) and render nothing.
#[test]
fn fully_hidden_buffer_settles_without_hanging() {
    let (mut font_system, mut buffer) = buffer_with_lines(5, Some(600.0));
    for line_i in 0..5 {
        buffer.set_line_hidden(line_i, true);
    }
    buffer.shape_until_scroll(&mut font_system, false);
    assert_eq!(buffer.layout_runs().count(), 0);
    assert_eq!(buffer.visible_line_count(), 0);
}

/// A hidden prefix with a short visible tail: the end-of-buffer scroll
/// adjustment walks back over the zero-height prefix and must settle at
/// the top with the tail rendered — not ping-pong forever between the
/// forward advance and the scroll-up correction.
#[test]
fn hidden_prefix_scroll_settles_at_top() {
    let (mut font_system, mut buffer) = buffer_with_lines(13, Some(600.0));
    for line_i in 0..=9 {
        buffer.set_line_hidden(line_i, true);
    }
    let mut scroll = buffer.scroll();
    scroll.line = 10;
    buffer.set_scroll(scroll);
    buffer.shape_until_scroll(&mut font_system, false);

    let runs: Vec<(usize, f32)> = buffer
        .layout_runs()
        .map(|run| (run.line_i, run.line_top))
        .collect();
    let lines: Vec<usize> = runs.iter().map(|(line_i, _)| *line_i).collect();
    assert_eq!(lines, vec![10, 11, 12]);
    assert_eq!(runs[0].1, 0.0, "visible tail must render at the top");
}

/// The per-line lifecycle mirrors alignment: preserved through set_text,
/// inherited by split_off, cleared by reset_new.
#[test]
fn buffer_line_hidden_lifecycle_follows_align() {
    let mut line = BufferLine::new(
        "hello world",
        LineEnding::Lf,
        AttrsList::new(&Attrs::new()),
        Shaping::Advanced,
    );
    assert!(!line.hidden());
    assert!(line.set_hidden(true));
    assert!(!line.set_hidden(true), "same value is a no-op");

    line.set_text("other", LineEnding::Lf, AttrsList::new(&Attrs::new()));
    assert!(line.hidden(), "text edits keep display properties");

    let tail = line.split_off(2);
    assert!(
        line.hidden() && tail.hidden(),
        "both split halves stay hidden"
    );

    line.reset_new(
        "fresh",
        LineEnding::Lf,
        AttrsList::new(&Attrs::new()),
        Shaping::Advanced,
    );
    assert!(!line.hidden(), "wholesale replacement clears hidden");
}
