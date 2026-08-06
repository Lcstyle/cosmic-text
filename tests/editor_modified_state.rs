#![cfg(feature = "vi")]

use std::sync::OnceLock;

use cosmic_text::{
    Attrs, Buffer, Cursor, Edit, Metrics, Shaping, SyntaxEditor, SyntaxSystem, ViEditor,
};

static SYNTAX_SYSTEM: OnceLock<SyntaxSystem> = OnceLock::new();

// New editor for tests
fn editor() -> ViEditor<'static, 'static> {
    // More or less copied from cosmic-edit
    let font_size: f32 = 14.0;
    let line_height = (font_size * 1.4).ceil();

    let metrics = Metrics::new(font_size, line_height);
    let buffer = Buffer::new_empty(metrics);
    let editor = SyntaxEditor::new(
        buffer,
        SYNTAX_SYSTEM.get_or_init(SyntaxSystem::new),
        "base16-eighties.dark",
    )
    .expect("Default theme `base16-eighties.dark` should be found");

    ViEditor::new(editor)
}

fn editor_text(editor: &ViEditor<'static, 'static>) -> String {
    let mut text = String::new();
    editor.with_buffer(|buffer| {
        for line in buffer.lines_iter() {
            text.push_str(line.text());
            text.push_str(line.ending().as_str());
        }
    });
    text
}

#[test]
fn editor_line_endings_preserved() {
    let mut editor = editor();
    assert_eq!(editor_text(&editor), "");

    let start = Cursor::new(0, 0);
    for &text in &[
        "No newlines",
        "One Newline\n",
        "Two\nNewlines\n",
        "LF\nCRLF\r\nCR\rLFCR\n\rNONE",
    ] {
        editor.start_change();
        let end = editor.insert_at(start, text, None);
        editor.finish_change();
        assert_eq!(editor_text(&editor), text);

        editor.start_change();
        editor.delete_range(start, end);
        editor.finish_change();
        assert_eq!(editor_text(&editor), "");

        editor.start_change();
        editor.undo();
        editor.finish_change();
        assert_eq!(editor_text(&editor), text);

        editor.start_change();
        editor.redo();
        editor.finish_change();
        assert_eq!(editor_text(&editor), "");
    }
}

// Replaying a delete whose region was display-chunked when recorded must
// derive its end from the recorded text: undo re-inserts those bytes as
// unsplit lines (chunk joins put no bytes in the record), so the recorded
// end cursor's line indices no longer exist by the time redo replays the
// delete. This is the whole-document rewrite shape a format command uses.
#[test]
fn redo_after_chunked_whole_document_replace() {
    let mut editor = editor();

    // One giant JSON-ish line, no newlines: set_text display-chunks it.
    let unit = r#"{"key":"value","n":12345},"#;
    let original: String = unit.repeat(4 * 1024); // ~104 KB, one logical line
    editor.with_buffer_mut(|buffer| {
        buffer.set_text(&original, &Attrs::new(), Shaping::Advanced, None);
    });
    let chunked_lines = editor.with_buffer(Buffer::line_count);
    assert!(chunked_lines > 1, "the giant line must be display-chunked");

    // Whole-document rewrite as ONE change record: select-all delete plus
    // insert of the replacement.
    let start = Cursor::new(0, 0);
    let end = editor.with_buffer(|buffer| {
        let last = buffer.line_count() - 1;
        Cursor::new(
            last,
            buffer.line_text_cow(last).map(|t| t.len()).unwrap_or(0),
        )
    });
    let replacement = "{\n  \"short\": true\n}";
    editor.start_change();
    editor.delete_range(start, end);
    editor.insert_at(start, replacement, None);
    editor.finish_change();
    assert_eq!(editor_text(&editor), replacement);

    // Undo restores the original bytes as ONE unsplit line.
    editor.undo();
    assert_eq!(editor_text(&editor), original);
    assert_eq!(
        editor.with_buffer(Buffer::line_count),
        1,
        "undo re-inserts the recorded bytes unsplit"
    );

    // Redo replays the recorded delete against that unsplit shape.
    editor.redo();
    assert_eq!(editor_text(&editor), replacement);

    // And the cycle stays stable.
    editor.undo();
    assert_eq!(editor_text(&editor), original);
    editor.redo();
    assert_eq!(editor_text(&editor), replacement);
}

// Tests that inserting into an empty editor correctly sets the editor as modified.
#[test]
fn insert_in_empty_editor_sets_changed() {
    let mut editor = editor();

    assert!(!editor.changed());
    editor.start_change();
    editor.insert_at(Cursor::new(0, 0), "Robert'); DROP TABLE Students;--", None);
    editor.finish_change();
    assert!(editor.changed());
}

// Tests an edge case where a save point is never set.
// Undoing changes should set the editor back to unmodified.
#[test]
fn insert_and_undo_in_unsaved_editor_is_unchanged() {
    let mut editor = editor();

    assert!(!editor.changed());
    editor.start_change();
    editor.insert_at(Cursor::new(0, 0), "loop {}", None);
    editor.finish_change();
    assert!(editor.changed());

    // Undoing the above change should set the editor as unchanged even if the save state is unset
    editor.start_change();
    editor.undo();
    editor.finish_change();
    assert!(!editor.changed());
}

#[test]
fn undo_to_save_point_sets_editor_to_unchanged() {
    let mut editor = editor();

    // Latest saved change is the first change
    editor.start_change();
    let cursor = editor.insert_at(Cursor::new(0, 0), "Ferris is Rust's ", None);
    editor.finish_change();
    assert!(
        editor.changed(),
        "Editor should be set to changed after insertion"
    );
    editor.save_point();
    assert!(
        !editor.changed(),
        "Editor should be set to unchanged after setting a save point"
    );

    // A new insert should set the editor as modified and the pivot should still be on the first
    // change from earlier
    editor.start_change();
    editor.insert_at(cursor, "mascot", None);
    editor.finish_change();
    assert!(
        editor.changed(),
        "Editor should be set to changed after inserting text after a save point"
    );

    // Undoing the latest change should set the editor to unmodified again
    editor.start_change();
    editor.undo();
    editor.finish_change();
    assert!(
        !editor.changed(),
        "Editor should be set to unchanged after undoing to save point"
    );
}

#[test]
fn redoing_to_save_point_sets_editor_as_unchanged() {
    let mut editor = editor();

    // Initial change
    assert!(
        !editor.changed(),
        "Editor should start in an unchanged state"
    );
    editor.start_change();
    editor.insert_at(Cursor::new(0, 0), "editor.start_change();", None);
    editor.finish_change();
    assert!(
        editor.changed(),
        "Editor should be set as modified after insert() and finish_change()"
    );
    editor.save_point();
    assert!(
        !editor.changed(),
        "Editor should be unchanged after setting a save point"
    );

    // Change to undo then redo
    editor.start_change();
    editor.insert_at(Cursor::new(1, 0), "editor.finish_change()", None);
    editor.finish_change();
    assert!(
        editor.changed(),
        "Editor should be set as modified after insert() and finish_change()"
    );
    editor.save_point();
    assert!(
        !editor.changed(),
        "Editor should be unchanged after setting a save point"
    );

    editor.undo();
    assert!(
        editor.changed(),
        "Undoing past save point should set editor as changed"
    );
    editor.redo();
    assert!(
        !editor.changed(),
        "Redoing to save point should set editor as unchanged"
    );
}

#[test]
fn redoing_past_save_point_sets_editor_to_changed() {
    let mut editor = editor();

    // Save point change to undo to and then redo past.
    editor.start_change();
    editor.insert_string("Walt Whitman ", None);
    editor.finish_change();

    // Set save point to the change above.
    assert!(
        editor.changed(),
        "Editor should be set as modified after insert() and finish_change()"
    );
    editor.save_point();
    assert!(
        !editor.changed(),
        "Editor should be unchanged after setting a save point"
    );

    editor.start_change();
    editor.insert_string("Allen Ginsberg ", None);
    editor.finish_change();

    editor.start_change();
    editor.insert_string("Jack Kerouac ", None);
    editor.finish_change();

    assert!(editor.changed(), "Editor should be modified insertion");

    // Undo to Whitman
    editor.undo();
    editor.undo();
    assert!(
        !editor.changed(),
        "Editor should be unmodified after undoing to the save point"
    );

    // Redo to Kerouac
    editor.redo();
    editor.redo();
    assert!(
        editor.changed(),
        "Editor should be modified after redoing past the save point"
    );
}

#[test]
fn undoing_past_save_point_sets_editor_to_changed() {
    let mut editor = editor();

    editor.start_change();
    editor.insert_string("Robert Fripp ", None);
    editor.finish_change();

    // Save point change to undo past.
    editor.start_change();
    editor.insert_string("Thurston Moore ", None);
    editor.finish_change();

    assert!(editor.changed(), "Editor should be changed after insertion");
    editor.save_point();
    assert!(
        !editor.changed(),
        "Editor should be unchanged after setting a save point"
    );

    editor.start_change();
    editor.insert_string("Kim Deal ", None);
    editor.finish_change();

    // Undo to the first change
    editor.undo();
    editor.undo();
    assert!(
        editor.changed(),
        "Editor should be changed after undoing past save point"
    );
}

// #[test]
// fn undo_all_changes() {
//     unimplemented!()
// }
