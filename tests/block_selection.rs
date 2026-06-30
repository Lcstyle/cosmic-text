// Behavioral spec for column/block selection (Selection::Block).
//
// A block selection is a rectangle: rows [min..=max] of the two corner cursors,
// and byte-columns [min..=max] of their indices, applied independently to each
// row and clamped to that row's length / char boundaries. Copy and delete operate
// on pure line text (no shaping), so these are deterministic unit tests.

use cosmic_text::{Attrs, Buffer, Cursor, Edit, Editor, Metrics, Selection, Shaping};

fn block_editor(text: &str) -> Editor<'static> {
    let mut buffer = Buffer::new_empty(Metrics::new(14.0, 20.0));
    buffer.set_text(text, &Attrs::new(), Shaping::Advanced, None);
    Editor::new(buffer)
}

fn buffer_text(editor: &Editor<'static>) -> String {
    let mut out = String::new();
    editor.with_buffer(|buffer| {
        for line in buffer.lines.iter() {
            out.push_str(line.text());
            out.push('\n');
        }
    });
    out
}

#[test]
fn block_selection_copies_rectangle() {
    let mut editor = block_editor("abcde\nfghij\nklmno");
    // anchor at row 0 col 1, active corner at row 2 col 3 => columns [1,3) over rows 0..=2
    editor.set_selection(Selection::Block(Cursor::new(0, 1)));
    editor.set_cursor(Cursor::new(2, 3));
    assert_eq!(editor.copy_selection().as_deref(), Some("bc\ngh\nlm"));
}

#[test]
fn block_selection_normalizes_reversed_corners() {
    let mut editor = block_editor("abcde\nfghij\nklmno");
    // corners given bottom-right first; result must be the same rectangle
    editor.set_selection(Selection::Block(Cursor::new(2, 3)));
    editor.set_cursor(Cursor::new(0, 1));
    assert_eq!(editor.copy_selection().as_deref(), Some("bc\ngh\nlm"));
}

#[test]
fn block_selection_clamps_to_short_lines() {
    // middle row "xy" is shorter than the block's right column
    let mut editor = block_editor("abcdef\nxy\nklmnop");
    editor.set_selection(Selection::Block(Cursor::new(0, 2)));
    editor.set_cursor(Cursor::new(2, 5));
    // cols [2,5): row0 "cde", row1 "" (clamped), row2 "mno"
    assert_eq!(editor.copy_selection().as_deref(), Some("cde\n\nmno"));
}

#[test]
fn block_selection_deletes_rectangle() {
    let mut editor = block_editor("abcde\nfghij\nklmno");
    editor.set_selection(Selection::Block(Cursor::new(0, 1)));
    editor.set_cursor(Cursor::new(2, 3));

    assert!(editor.delete_selection());
    // each row loses cols [1,3)
    assert_eq!(buffer_text(&editor), "ade\nfij\nkno\n");
    // cursor collapses to the top-left corner, selection cleared
    assert_eq!(editor.cursor(), Cursor::new(0, 1));
    assert_eq!(editor.selection(), Selection::None);
}
