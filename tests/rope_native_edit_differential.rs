#![cfg(all(feature = "rope-buffer", feature = "vi"))]

//! Differential harness for native rope mutation: every editing operation
//! runs against a Full-backed and a Rope-backed editor in lockstep,
//! asserting identical bytes, cursors, line counts and Change records after
//! every step — with the rope arm never thawing (`is_rope()` stays true
//! through every edit, undo and redo).
//!
//! The corpus deliberately excludes the CR|LF adjacency merge family
//! (lone `\r` next to an edit boundary touching `\n`) and the exotic
//! breaks (`\n\r`, VT/FF/NEL/LS/PS) where the arms segment differently by
//! design; those get targeted byte-level tests below instead
//! (`cr_lf_merge_family_is_byte_exact`).

use std::sync::OnceLock;

use cosmic_text::{
    Attrs, Buffer, Change, Cursor, Edit, Editor, FontSystem, Metrics, RopeStore, Shaping,
    SyntaxEditor, SyntaxSystem, ViEditor,
};

const METRICS: Metrics = Metrics::new(14.0, 20.0);

fn full_buffer(content: &str) -> Buffer {
    let mut buffer = Buffer::new_empty(METRICS);
    buffer.set_text(content, &Attrs::new(), Shaping::Advanced, None);
    buffer
}

fn rope_buffer(content: &str) -> Buffer {
    Buffer::new_rope(
        METRICS,
        RopeStore::from_text(content, &Attrs::new(), Shaping::Advanced),
    )
}

/// Reconstruct the document byte-exactly from any storage arm: text plus
/// real ending per line (dogfooding the cold-capable `Buffer::line_ending`).
fn reconstruct(buffer: &Buffer) -> String {
    let mut text = String::new();
    for i in 0..buffer.line_count() {
        if let Some(cow) = buffer.line_text_cow(i) {
            text.push_str(&cow);
        }
        if let Some(ending) = buffer.line_ending(i) {
            text.push_str(ending.as_str());
        }
    }
    text
}

fn assert_change_eq(full: &Change, rope: &Change, ctx: &str) {
    assert_eq!(
        full.items.len(),
        rope.items.len(),
        "{ctx}: ChangeItem count diverged"
    );
    for (i, (f, r)) in full.items.iter().zip(rope.items.iter()).enumerate() {
        assert_eq!(f.start, r.start, "{ctx}: item[{i}] start diverged");
        assert_eq!(f.end, r.end, "{ctx}: item[{i}] end diverged");
        assert_eq!(f.text, r.text, "{ctx}: item[{i}] text diverged");
        assert_eq!(f.insert, r.insert, "{ctx}: item[{i}] insert flag diverged");
    }
}

#[derive(Clone, Debug)]
enum Op {
    InsertAt {
        line: usize,
        index: usize,
        text: String,
    },
    DeleteRange {
        start: (usize, usize),
        end: (usize, usize),
    },
    Undo,
    Redo,
}

fn insert(line: usize, index: usize, text: &str) -> Op {
    Op::InsertAt {
        line,
        index,
        text: text.to_string(),
    }
}

fn delete(start: (usize, usize), end: (usize, usize)) -> Op {
    Op::DeleteRange { start, end }
}

struct Twins {
    full: Editor<'static>,
    rope: Editor<'static>,
    /// Changes captured from the forward pass, for undo replay.
    captured: Vec<Change>,
    /// Redo stack.
    undone: Vec<Change>,
}

impl Twins {
    fn new(content: &str) -> Self {
        let twins = Self {
            full: Editor::new(full_buffer(content)),
            rope: Editor::new(rope_buffer(content)),
            captured: Vec::new(),
            undone: Vec::new(),
        };
        twins.assert_twins("initial state");
        twins
    }

    fn apply(&mut self, op: &Op, ctx: &str) {
        match op {
            Op::InsertAt { line, index, text } => {
                let cursor = Cursor::new(*line, *index);
                self.full.start_change();
                self.rope.start_change();
                let full_cursor = self.full.insert_at(cursor, text, None);
                let rope_cursor = self.rope.insert_at(cursor, text, None);
                assert_eq!(
                    full_cursor, rope_cursor,
                    "{ctx}: insert_at returned cursor diverged"
                );
                self.full.set_cursor(full_cursor);
                self.rope.set_cursor(rope_cursor);
                self.capture(ctx);
            }
            Op::DeleteRange { start, end } => {
                let start = Cursor::new(start.0, start.1);
                let end = Cursor::new(end.0, end.1);
                self.full.start_change();
                self.rope.start_change();
                self.full.delete_range(start, end);
                self.rope.delete_range(start, end);
                // Like delete_selection: the cursor collapses to the start.
                self.full.set_cursor(start);
                self.rope.set_cursor(start);
                self.capture(ctx);
            }
            Op::Undo => {
                let change = self
                    .captured
                    .pop()
                    .expect("undo requested with no captured change");
                let mut reversed = change.clone();
                reversed.reverse();
                assert!(self.full.apply_change(&reversed), "{ctx}: full undo failed");
                assert!(self.rope.apply_change(&reversed), "{ctx}: rope undo failed");
                self.undone.push(change);
            }
            Op::Redo => {
                let change = self
                    .undone
                    .pop()
                    .expect("redo requested with no undone change");
                assert!(self.full.apply_change(&change), "{ctx}: full redo failed");
                assert!(self.rope.apply_change(&change), "{ctx}: rope redo failed");
                self.captured.push(change);
            }
        }
        self.assert_twins(ctx);
    }

    /// Finish the pending change on both arms, assert record parity, keep it
    /// for undo replay.
    fn capture(&mut self, ctx: &str) {
        let full_change = self.full.finish_change();
        let rope_change = self.rope.finish_change();
        match (full_change, rope_change) {
            (Some(full), Some(rope)) => {
                assert_change_eq(&full, &rope, ctx);
                if !full.items.is_empty() {
                    self.captured.push(full);
                    self.undone.clear();
                }
            }
            (None, None) => {}
            (full, rope) => panic!(
                "{ctx}: change presence diverged: full={:?} rope={:?}",
                full.is_some(),
                rope.is_some()
            ),
        }
    }

    fn assert_twins(&self, ctx: &str) {
        assert!(
            self.rope.with_buffer(|b| b.is_rope()),
            "{ctx}: rope arm thawed"
        );
        assert!(
            !self.full.with_buffer(|b| b.is_rope()),
            "{ctx}: full arm became rope-backed"
        );
        let full_text = self.full.with_buffer(reconstruct);
        let rope_text = self.rope.with_buffer(reconstruct);
        assert_eq!(full_text, rope_text, "{ctx}: reconstructed text diverged");
        assert_eq!(
            self.full.cursor(),
            self.rope.cursor(),
            "{ctx}: cursor diverged"
        );
        assert_eq!(
            self.full.with_buffer(Buffer::line_count),
            self.rope.with_buffer(Buffer::line_count),
            "{ctx}: line_count diverged"
        );
    }

    fn text(&self) -> String {
        self.full.with_buffer(reconstruct)
    }

    fn line_count(&self) -> usize {
        self.full.with_buffer(Buffer::line_count)
    }

    fn line_len(&self, line: usize) -> usize {
        self.full
            .with_buffer(|b| b.line_text_cow(line).map_or(0, |c| c.len()))
    }

    /// Midpoint of a line's text, snapped down to a char boundary.
    fn mid_index(&self, line: usize) -> usize {
        self.full.with_buffer(|b| {
            let text = b
                .line_text_cow(line)
                .map(|c| c.into_owned())
                .unwrap_or_default();
            let mut i = text.len() / 2;
            while i > 0 && !text.is_char_boundary(i) {
                i -= 1;
            }
            i
        })
    }
}

/// LF, CRLF, mixed, trailing/no trailing newline, empty, multibyte-heavy,
/// and a 200-line file for cache-shift visibility. All lines well under the
/// 32K display-chunking threshold; no lone CR, `\n\r`, or exotic breaks.
fn corpus() -> Vec<(&'static str, String)> {
    let two_hundred: String = (0..200)
        .map(|i| format!("line {i} with some padding\n"))
        .collect();
    vec![
        (
            "lf",
            "alpha\nbravo charlie\ndelta echo foxtrot\ngolf\n".to_string(),
        ),
        (
            "crlf",
            "alpha\r\nbravo charlie\r\ndelta echo foxtrot\r\ngolf\r\n".to_string(),
        ),
        (
            "mixed",
            "alpha\r\nbravo charlie\ndelta echo foxtrot\r\ngolf\n".to_string(),
        ),
        (
            "no_trailing_newline",
            "alpha\nbravo charlie\ndelta echo".to_string(),
        ),
        ("trailing_newline", "alpha\nbravo\n".to_string()),
        ("empty", String::new()),
        (
            "multibyte",
            "héllo wörld\n日本語のテキスト日本\nэмодзи 😀🦀 😀\ndernière ligne é😀\n".to_string(),
        ),
        ("two_hundred_lines", two_hundred),
    ]
}

/// Run `script` against every corpus entry.
fn for_corpus(script: impl Fn(&mut Twins, &str)) {
    for (name, content) in corpus() {
        let mut twins = Twins::new(&content);
        script(&mut twins, name);
    }
}

// --- Scripts (§8.2) ---

#[test]
fn script_1_typing_run() {
    for_corpus(|t, name| {
        let keys = "abcdefghijklmnopqrst";
        let line = 1.min(t.line_count() - 1);
        let mut index = t.mid_index(line);
        for i in 0..keys.len() {
            let ch = &keys[i..=i];
            t.apply(&insert(line, index, ch), &format!("{name}: typing[{i}]"));
            index += ch.len();
        }
    });
}

#[test]
fn script_2_multiline_paste_mid_line() {
    for_corpus(|t, name| {
        let line = 1.min(t.line_count() - 1);
        let index = t.mid_index(line);
        t.apply(&insert(line, index, "X\nY\nZ"), &format!("{name}: paste lf"));
        let line = t.line_count() - 1;
        let index = t.mid_index(line);
        t.apply(&insert(line, index, "X\r\nY"), &format!("{name}: paste crlf"));
    });
}

#[test]
fn script_3_paste_ending_in_break() {
    for_corpus(|t, name| {
        let line = t.line_count() / 2;
        let index = t.mid_index(line);
        let before_count = t.line_count();
        t.apply(&insert(line, index, "tail\n"), &format!("{name}: paste tail"));
        assert_eq!(t.line_count(), before_count + 1, "{name}: sentinel segment");
        // The forced trailing sentinel: cursor lands at the start of the
        // next line.
        assert_eq!(t.full.cursor(), Cursor::new(line + 1, 0), "{name}: cursor");
    });
}

#[test]
fn script_4_cross_line_delete() {
    for_corpus(|t, name| {
        if t.line_count() < 3 {
            return;
        }
        let start = (0, t.mid_index(0));
        let end = (2, t.mid_index(2));
        t.apply(&delete(start, end), &format!("{name}: cross-line delete"));
    });
}

#[test]
fn script_5_line_join_deletes() {
    for_corpus(|t, name| {
        // Join the first pair (covers LF or CRLF ending removal depending
        // on the corpus entry), then the last pair.
        if t.line_count() < 2 {
            return;
        }
        let len = t.line_len(0);
        t.apply(&delete((0, len), (1, 0)), &format!("{name}: join first pair"));
        if t.line_count() < 2 {
            return;
        }
        let line = t.line_count() - 2;
        let len = t.line_len(line);
        t.apply(
            &delete((line, len), (line + 1, 0)),
            &format!("{name}: join last pair"),
        );
    });
}

#[test]
fn script_6_boundary_ops() {
    for_corpus(|t, name| {
        t.apply(&insert(0, 0, "head "), &format!("{name}: insert at origin"));
        if t.line_count() >= 2 {
            t.apply(
                &delete((0, 0), (1, 0)),
                &format!("{name}: whole-line delete including ending"),
            );
        }
        let first_char_len = t
            .full
            .with_buffer(|b| b.line_text_cow(0).and_then(|c| c.chars().next().map(char::len_utf8)));
        if let Some(n) = first_char_len {
            t.apply(&delete((0, 0), (0, n)), &format!("{name}: delete at (0,0)"));
        }
    });
}

#[test]
fn script_7_eof() {
    for_corpus(|t, name| {
        let last = t.line_count() - 1;
        let len = t.line_len(last);
        t.apply(&insert(last, len, "END"), &format!("{name}: insert at eof"));
        let beyond = t.line_count() + 2;
        t.apply(
            &insert(beyond, 0, "beyond"),
            &format!("{name}: insert beyond eof (backfill)"),
        );
        assert_eq!(t.line_count(), beyond + 1, "{name}: backfilled line count");
    });
}

#[test]
fn script_8_multibyte() {
    for_corpus(|t, name| {
        // Insert between 日 and 本 where present.
        let pos = t.full.with_buffer(|b| {
            (0..b.line_count()).find_map(|i| {
                b.line_text_cow(i)
                    .and_then(|text| text.find("日本").map(|at| (i, at + "日".len())))
            })
        });
        if let Some((line, index)) = pos {
            t.apply(&insert(line, index, "中"), &format!("{name}: insert 日|本"));
        }
        // Delete exactly one 😀 where present.
        let pos = t.full.with_buffer(|b| {
            (0..b.line_count())
                .find_map(|i| b.line_text_cow(i).and_then(|text| text.find('😀').map(|at| (i, at))))
        });
        if let Some((line, index)) = pos {
            t.apply(
                &delete((line, index), (line, index + '😀'.len_utf8())),
                &format!("{name}: delete one emoji"),
            );
        }
        // Multibyte data with an interior newline, at end of document.
        let line = t.line_count() - 1;
        let index = t.line_len(line);
        t.apply(
            &insert(line, index, "é😀\n日"),
            &format!("{name}: insert multibyte with newline"),
        );
    });
}

#[test]
fn script_9_whole_document_delete_and_reinsert() {
    for_corpus(|t, name| {
        let before = t.text();
        let last = t.line_count() - 1;
        let len = t.line_len(last);
        if last == 0 && len == 0 {
            return; // already empty
        }
        t.apply(
            &delete((0, 0), (last, len)),
            &format!("{name}: whole-document delete"),
        );
        assert_eq!(t.text(), "", "{name}: document should be empty");
        // The delete's captured text is byte-identical to the document —
        // the §5 slice-equals-join proof.
        let captured = t.captured.last().expect("captured whole-document delete");
        assert_eq!(captured.items.len(), 1);
        assert_eq!(
            captured.items[0].text, before,
            "{name}: removed text must equal the document bytes"
        );
        let removed = captured.items[0].text.clone();
        t.apply(&insert(0, 0, &removed), &format!("{name}: reinsert document"));
        assert_eq!(t.text(), before, "{name}: round trip");
    });
}

#[test]
fn script_10_degenerates() {
    for_corpus(|t, name| {
        let line = t.line_count() - 1;
        let index = t.line_len(line);
        let depth = t.captured.len();
        t.apply(&insert(line, index, ""), &format!("{name}: empty insert"));
        assert_eq!(
            t.captured.len(),
            depth,
            "{name}: empty insert must record nothing on either arm"
        );
        t.apply(
            &delete((line, index), (line, index)),
            &format!("{name}: empty delete"),
        );
        // The full arm records an empty-text item for an empty delete; the
        // rope arm mirrors it (asserted equal in capture()).
    });
}

#[test]
fn script_11_undo_redo_storm() {
    for_corpus(|t, name| {
        struct Step {
            before: String,
            after: String,
            /// Un-recorded residue this op leaves permanently: the beyond-EOF
            /// insert backfills empty lines that are not part of the recorded
            /// change, so its undo leaves them behind (reference full-arm
            /// semantics, matched by the rope arm — parity itself is asserted
            /// inside apply() for every op regardless).
            residue: String,
        }
        let base_depth = t.captured.len();
        let mut steps: Vec<Step> = Vec::new();

        let mut forward = |t: &mut Twins, op: Op, residue: String, ctx: String| {
            let depth = t.captured.len();
            let before = t.text();
            t.apply(&op, &ctx);
            if t.captured.len() > depth {
                steps.push(Step {
                    before,
                    after: t.text(),
                    residue,
                });
            }
        };

        // Script 2 ops.
        let line = 1.min(t.line_count() - 1);
        let index = t.mid_index(line);
        forward(t, insert(line, index, "X\nY\nZ"), String::new(), format!("{name}: storm paste lf"));
        let line = t.line_count() - 1;
        let index = t.mid_index(line);
        forward(t, insert(line, index, "X\r\nY"), String::new(), format!("{name}: storm paste crlf"));
        // Script 4 op.
        if t.line_count() >= 3 {
            let start = (0, t.mid_index(0));
            let end = (2, t.mid_index(2));
            forward(t, delete(start, end), String::new(), format!("{name}: storm cross-line delete"));
        }
        // Script 7 ops.
        let last = t.line_count() - 1;
        let len = t.line_len(last);
        forward(t, insert(last, len, "END"), String::new(), format!("{name}: storm insert at eof"));
        let beyond = t.line_count() + 2;
        let backfilled = "\n".repeat(beyond + 1 - t.line_count());
        forward(
            t,
            insert(beyond, 0, "beyond"),
            backfilled,
            format!("{name}: storm insert beyond eof"),
        );

        // Undo everything, checking each checkpoint on the way down. `acc`
        // carries the residue of already-undone ops (it sits at EOF, after
        // every checkpoint's text).
        let mut undone_steps: Vec<Step> = Vec::new();
        let mut acc = String::new();
        while t.captured.len() > base_depth {
            let step = steps.pop().expect("one checkpoint per captured change");
            assert_eq!(
                t.text(),
                format!("{}{acc}", step.after),
                "{name}: pre-undo state"
            );
            t.apply(&Op::Undo, &format!("{name}: storm undo"));
            acc = format!("{}{acc}", step.residue);
            assert_eq!(
                t.text(),
                format!("{}{acc}", step.before),
                "{name}: undo checkpoint"
            );
            undone_steps.push(step);
        }
        // And redo everything back up, oldest change first.
        while let Some(step) = undone_steps.pop() {
            acc = acc
                .strip_prefix(step.residue.as_str())
                .expect("residue accounting")
                .to_string();
            t.apply(&Op::Redo, &format!("{name}: storm redo"));
            assert_eq!(
                t.text(),
                format!("{}{acc}", step.after),
                "{name}: redo checkpoint"
            );
        }
    });
}

// --- Targeted divergence tier (§8.4): the CR|LF adjacency merge family ---
//
// ropey (unicode_lines) fuses an adjacent \r and \n into a single CRLF
// break; the full arm keeps them as separate lines. Bytes stay identical on
// both arms — tuple equality is exempted, and the rope arm records
// byte-true post-edit cursors so undo/redo round-trips stay byte-exact.

fn merge_case(
    content: &str,
    op: impl Fn(&mut Editor<'static>),
    expected_rope_item: (Cursor, Cursor, &str, bool),
    ctx: &str,
) {
    let mut full = Editor::new(full_buffer(content));
    let mut rope = Editor::new(rope_buffer(content));

    full.start_change();
    op(&mut full);
    let full_change = full.finish_change().expect("full change");
    rope.start_change();
    op(&mut rope);
    let rope_change = rope.finish_change().expect("rope change");

    assert!(rope.with_buffer(|b| b.is_rope()), "{ctx}: rope arm thawed");
    let edited_full = full.with_buffer(reconstruct);
    let edited_rope = rope.with_buffer(reconstruct);
    assert_eq!(edited_full, edited_rope, "{ctx}: bytes diverged after edit");

    // The rope arm's record is byte-true in the post-edit segmentation.
    let (start, end, text, insert) = expected_rope_item;
    assert_eq!(rope_change.items.len(), 1, "{ctx}: rope item count");
    assert_eq!(rope_change.items[0].start, start, "{ctx}: rope item start");
    assert_eq!(rope_change.items[0].end, end, "{ctx}: rope item end");
    assert_eq!(rope_change.items[0].text, text, "{ctx}: rope item text");
    assert_eq!(rope_change.items[0].insert, insert, "{ctx}: rope item flag");

    // Undo (each arm replays its own record) restores the original bytes.
    let mut full_undo = full_change.clone();
    full_undo.reverse();
    let mut rope_undo = rope_change.clone();
    rope_undo.reverse();
    assert!(full.apply_change(&full_undo), "{ctx}: full undo");
    assert!(rope.apply_change(&rope_undo), "{ctx}: rope undo");
    assert_eq!(full.with_buffer(reconstruct), content, "{ctx}: full undo bytes");
    assert_eq!(rope.with_buffer(reconstruct), content, "{ctx}: rope undo bytes");
    assert!(rope.with_buffer(|b| b.is_rope()), "{ctx}: thawed during undo");

    // Redo restores the edited bytes.
    assert!(full.apply_change(&full_change), "{ctx}: full redo");
    assert!(rope.apply_change(&rope_change), "{ctx}: rope redo");
    assert_eq!(full.with_buffer(reconstruct), edited_full, "{ctx}: full redo bytes");
    assert_eq!(rope.with_buffer(reconstruct), edited_rope, "{ctx}: rope redo bytes");
    assert!(rope.with_buffer(|b| b.is_rope()), "{ctx}: thawed during redo");
}

#[test]
fn cr_lf_merge_family_is_byte_exact() {
    // Shape 1: data ends with \r and the byte at the insertion point is \n.
    // The recorded end lands inside the merged CRLF ending of line 0.
    merge_case(
        "AB\ncd",
        |ed| {
            ed.insert_at(Cursor::new(0, 2), "\r", None);
        },
        (Cursor::new(0, 2), Cursor::new(0, 3), "\r", true),
        "shape 1 (insert trailing CR before LF)",
    );
    // Shape 2: data begins with \n and the byte before the insertion point
    // is \r. The recorded start re-maps into the absorbing line.
    merge_case(
        "a\rb",
        |ed| {
            ed.insert_at(Cursor::new(1, 0), "\nX", None);
        },
        (Cursor::new(0, 2), Cursor::new(1, 1), "\nX", true),
        "shape 2 (insert leading LF after CR)",
    );
    // Shape 3: a delete joins \r…\n. The §3.3 worked example: the recorded
    // start is the post-delete byte-true join point (0,2), inside the fused
    // CRLF; the end stays as passed for redo.
    merge_case(
        "a\rX\nb",
        |ed| {
            ed.delete_range(Cursor::new(1, 0), Cursor::new(1, 1));
        },
        (Cursor::new(0, 2), Cursor::new(1, 1), "X", false),
        "shape 3 (delete joining CR and LF)",
    );
}

// --- ViEditor tier (§8.4): changed() parity and reset_history ---

static SYNTAX_SYSTEM: OnceLock<SyntaxSystem> = OnceLock::new();

fn rope_vi_editor(content: &str) -> ViEditor<'static, 'static> {
    let editor = SyntaxEditor::new(
        rope_buffer(content),
        SYNTAX_SYSTEM.get_or_init(SyntaxSystem::new),
        "base16-eighties.dark",
    )
    .expect("default theme should be found");
    ViEditor::new(editor)
}

fn vi_text(editor: &ViEditor<'static, 'static>) -> String {
    editor.with_buffer(reconstruct)
}

#[test]
fn rope_vi_editor_changed_parity() {
    let mut editor = rope_vi_editor("one\ntwo\nthree\n");
    assert!(!editor.changed());
    assert!(editor.with_buffer(|b| b.is_rope()));

    editor.start_change();
    editor.insert_at(Cursor::new(1, 0), "edited ", None);
    editor.finish_change();
    assert!(editor.changed(), "edit must set changed()");
    assert_eq!(vi_text(&editor), "one\nedited two\nthree\n");

    editor.undo();
    assert!(!editor.changed(), "undo to pristine must clear changed()");
    assert_eq!(vi_text(&editor), "one\ntwo\nthree\n");

    editor.redo();
    assert!(editor.changed(), "redo past pristine must set changed()");
    assert_eq!(vi_text(&editor), "one\nedited two\nthree\n");

    assert!(
        editor.with_buffer(|b| b.is_rope()),
        "the whole cycle must stay rope-native"
    );
}

#[test]
fn reset_history_clears_changed_and_undo_stack() {
    let mut editor = rope_vi_editor("one\ntwo\n");

    editor.start_change();
    editor.insert_at(Cursor::new(0, 3), " edited", None);
    editor.finish_change();
    assert!(editor.changed());
    assert_eq!(vi_text(&editor), "one edited\ntwo\n");

    editor.reset_history();
    assert!(!editor.changed(), "reset_history must clear changed()");

    // A following undo must be a no-op: the recorded Changes are gone.
    editor.undo();
    assert_eq!(
        vi_text(&editor),
        "one edited\ntwo\n",
        "undo after reset_history must not replay stale changes"
    );
    assert!(!editor.changed());
    assert!(
        editor.with_buffer(|b| b.is_rope()),
        "the whole flow must stay rope-native"
    );
}

// --- Tier 2 (§8.4): shaped consistency after native edits ---
//
// The invalidation contract's proof against stale-cache rendering: after
// native edits at, above and below a deep viewport, every visible layout
// run must show the store's current text at its absolute line index.

#[test]
fn shaped_consistency_after_native_edits_deep_in_file() {
    let mut font_system = FontSystem::new();
    let content: String = (0..10_000).map(|i| format!("line {i} padding padding\n")).collect();
    let mut editor = Editor::new(rope_buffer(&content));
    editor.with_buffer_mut(|buffer| {
        buffer.set_size(Some(800.0), Some(600.0));
        let mut scroll = buffer.scroll();
        scroll.line = 5_000;
        buffer.set_scroll(scroll);
        buffer.shape_until_scroll(&mut font_system, false);
    });

    let mut assert_visible_consistent = |editor: &mut Editor<'static>, ctx: &str| {
        editor.with_buffer_mut(|buffer| {
            buffer.shape_until_scroll(&mut font_system, false);
            let runs: Vec<(usize, String)> = buffer
                .layout_runs()
                .map(|run| (run.line_i, run.text.to_string()))
                .collect();
            assert!(!runs.is_empty(), "{ctx}: viewport must have runs");
            for pair in runs.windows(2) {
                assert_eq!(
                    pair[1].0,
                    pair[0].0 + 1,
                    "{ctx}: run line indices must stay consecutive"
                );
            }
            for (line_i, run_text) in &runs {
                let store_text = buffer
                    .line_text_cow(*line_i)
                    .map(|c| c.into_owned())
                    .expect("visible line in bounds");
                assert_eq!(
                    &store_text, run_text,
                    "{ctx}: run text must match store text at absolute line {line_i}"
                );
            }
            assert!(buffer.is_rope(), "{ctx}: must stay rope-backed");
        });
    };

    assert_visible_consistent(&mut editor, "before edits");

    // Edit inside the viewport.
    editor.insert_at(Cursor::new(5_002, 5), "IN-VIEW ", None);
    assert_visible_consistent(&mut editor, "after in-viewport insert");

    // Edit above the viewport: shifts every visible line's absolute index.
    editor.insert_at(Cursor::new(100, 0), "ABOVE\n", None);
    assert_visible_consistent(&mut editor, "after above-viewport insert");

    // Line-join delete above the viewport: shifts the other way.
    let join_len = editor.with_buffer(|b| b.line_text_cow(200).map_or(0, |c| c.len()));
    editor.delete_range(Cursor::new(200, join_len), Cursor::new(201, 0));
    assert_visible_consistent(&mut editor, "after above-viewport delete");

    // Edit below the viewport: must not disturb the visible window.
    editor.insert_at(Cursor::new(9_000, 0), "BELOW ", None);
    assert_visible_consistent(&mut editor, "after below-viewport insert");

    // The document reflects all four edits, byte-exactly.
    let text = editor.with_buffer(reconstruct);
    assert!(text.contains("IN-VIEW "));
    assert!(text.contains("ABOVE\n"));
    assert!(text.contains("BELOW "));
}
