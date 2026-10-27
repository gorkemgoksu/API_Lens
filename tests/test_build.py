"""Tests for stage 1 documentation extraction and Sphinx-style rendering."""

from __future__ import annotations

import pytest

from scripts.build_dataset import (
    BuildRecord,
    Candidate,
    CleanedSentence,
    Paragraph,
    ResolvedParagraph,
    assign_split,
    classify_source_type,
    clean_sentence,
    decode_source_bytes,
    deduplicate_records,
    extract_document,
    has_markup_residue,
    make_test_noparen,
    normalize_candidate_target,
    parse_badge_substitutions,
    render_inline,
    resolve_candidate,
    resolve_paragraph,
    sample_zero_entity_records,
    split_sentences,
)


def rendered(markup: str, default_role: str | None = None):
    text, candidates, literals = render_inline(markup, default_role)
    return text, candidates, literals


@pytest.mark.parametrize(
    ("markup", "expected", "role", "raw_target"),
    [
        (":func:`numpy.mean`", "numpy.mean()", "func", "numpy.mean"),
        (":py:meth:`DataFrame.groupby`", "DataFrame.groupby()", "py:meth", "DataFrame.groupby"),
        ("{class}`numpy.ndarray`", "numpy.ndarray", "class", "numpy.ndarray"),
        ("{py:attr}`DataFrame.shape`", "DataFrame.shape", "py:attr", "DataFrame.shape"),
        (":func:`~numpy.linalg.norm`", "norm()", "func", "~numpy.linalg.norm"),
        (":class:`array <numpy.ndarray>`", "array", "class", "array <numpy.ndarray>"),
        (":func:`!numpy.mean`", "numpy.mean()", "func", "!numpy.mean"),
    ],
)
def test_api_roles_render_with_exact_spans(markup, expected, role, raw_target) -> None:
    text, candidates, _ = rendered(markup)
    assert text == expected
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.role == role
    assert candidate.raw_target == raw_target
    assert text[candidate.char_start : candidate.char_end] == expected


def test_non_api_role_renders_visible_text_without_candidate() -> None:
    text, candidates, _ = rendered("See :doc:`the guide <user_guide/index>`.")
    assert text == "See the guide."
    assert candidates == ()


def test_non_python_domain_role_is_not_an_api_candidate() -> None:
    text, candidates, _ = rendered("Return :c:data:`NPY_TRUE`.")
    assert text == "Return NPY_TRUE."
    assert candidates == ()


def test_double_backticks_become_plain_text_and_are_tracked() -> None:
    text, candidates, literals = rendered("Use ``DataFrame`` here.")
    assert text == "Use DataFrame here."
    assert candidates == ()
    assert text[literals[0].char_start : literals[0].char_end] == "DataFrame"


def test_emphasis_markup_renders_visible_text() -> None:
    text, candidates, _ = rendered("Use *context* and **structure**.")
    assert text == "Use context and structure."
    assert candidates == ()


def test_rst_escape_before_candidate_does_not_shift_offsets() -> None:
    text, candidates, _ = rendered("Plural\\ s and :class:`DataFrame`.")
    assert text == "Plurals and DataFrame."
    assert text[candidates[0].char_start : candidates[0].char_end] == "DataFrame"


@pytest.mark.parametrize("default_role", ["obj", "py:obj", "autolink"])
def test_single_backticks_follow_api_default_role(default_role: str) -> None:
    text, candidates, _ = rendered("Use `numpy.mean`.", default_role)
    assert text == "Use numpy.mean."
    assert text[candidates[0].char_start : candidates[0].char_end] == "numpy.mean"
    assert candidates[0].role == "obj"


def test_single_backticks_are_plain_for_literal_default_role() -> None:
    text, candidates, _ = rendered("Use `numpy.mean`.", "literal")
    assert text == "Use numpy.mean."
    assert candidates == ()


def test_current_module_is_tracked_and_directive_is_not_rendered() -> None:
    source = """.. currentmodule:: pandas

Use :class:`DataFrame` here.
"""
    paragraphs = extract_document("pandas", "doc/source/user_guide/a.rst", source, None)
    assert len(paragraphs) == 1
    assert paragraphs[0].current_module == "pandas"
    assert paragraphs[0].text == "Use DataFrame here."


def test_jinja_header_sets_current_module_or_raises_without_one() -> None:
    source = "{{ header }}\n\nUse :class:`DataFrame` here."
    with pytest.raises(ValueError, match="header.*currentmodule"):
        extract_document("pandas", "doc/source/user_guide/a.rst", source, None)

    paragraph = extract_document(
        "pandas",
        "doc/source/user_guide/a.rst",
        source,
        None,
        header_current_module="pandas",
    )[0]
    assert paragraph.current_module == "pandas"
    assert paragraph.text == "Use DataFrame here."


def test_myst_current_module_is_tracked() -> None:
    source = """```{py:currentmodule} numpy.linalg
```

Use {func}`norm` here.
"""
    paragraph = extract_document("numpy", "doc/source/guide/a.md", source, None)[0]
    assert paragraph.current_module == "numpy.linalg"


@pytest.mark.parametrize(
    "source",
    [
        "Before.\n\n.. code-block:: python\n\n   hidden()\n\nAfter.",
        "Before.\n\n```python\nhidden()\n```\n\nAfter.",
        "Before.\n\n>>> hidden()\n\nAfter.",
        "Before.\n\n.. toctree::\n   :maxdepth: 2\n\n   hidden\n\nAfter.",
    ],
)
def test_code_prompts_and_toctrees_are_skipped(source: str) -> None:
    paragraphs = extract_document("numpy", "doc/source/reference/a.rst", source, None)
    assert [paragraph.text for paragraph in paragraphs] == ["Before.", "After."]


def test_literal_block_is_skipped_but_intro_is_rendered() -> None:
    source = """Example::

   hidden()

Visible again.
"""
    paragraphs = extract_document("numpy", "doc/source/reference/a.rst", source, None)
    assert [paragraph.text for paragraph in paragraphs] == ["Example:", "Visible again."]


def test_headings_comments_substitutions_and_table_are_skipped() -> None:
    source = """Heading
=======

.. a comment
   hidden continuation

.. |x| replace:: hidden

+---+---+
| A | B |
+===+===+
| 1 | 2 |
+---+---+

Visible prose.
"""
    paragraphs = extract_document("numpy", "doc/source/reference/a.rst", source, None)
    assert [paragraph.text for paragraph in paragraphs] == ["Visible prose."]


def test_raw_html_badge_substitution_renders_visible_label() -> None:
    source = """.. |Fix| replace:: :raw-html:`<span class="badge text-bg-danger">Fix</span>`

This |Fix| entry corrects the documented behavior.
"""
    substitutions = parse_badge_substitutions(source)
    paragraphs = extract_document(
        "scikit-learn",
        "doc/whats_new/v1.9.rst",
        source,
        None,
        substitutions=substitutions,
    )
    assert substitutions == {"Fix": "Fix"}
    assert [paragraph.text for paragraph in paragraphs] == [
        "This Fix entry corrects the documented behavior."
    ]


def test_markdown_heading_is_skipped() -> None:
    source = "(quick-start)=\n\n## Quick start\n\nVisible prose."
    paragraphs = extract_document("pytorch", "docs/source/guide.md", source, None)
    assert [paragraph.text for paragraph in paragraphs] == ["Visible prose."]


def test_list_items_become_separate_paragraphs() -> None:
    source = "- First item.\n- Second item."
    paragraphs = extract_document("numpy", "doc/source/guide.rst", source, None)
    assert [paragraph.text for paragraph in paragraphs] == ["First item.", "Second item."]


def test_list_item_ending_in_connector_joins_the_next_item() -> None:
    source = "1) dimensions are equal, or\n2) one dimension is one."
    paragraphs = extract_document("numpy", "doc/source/guide.rst", source, None)
    assert [paragraph.text for paragraph in paragraphs] == [
        "dimensions are equal, or one dimension is one."
    ]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("doc/source/whatsnew/v3.1.0.rst", "changelog"),
        ("doc/whats_new/v1.9.rst", "changelog"),
        ("doc/tutorials/intro.rst", "tutorial"),
        ("doc/source/user_guide/groupby.rst", "guide"),
        ("doc/source/user/basics.indexing.rst", "guide"),
        ("doc/source/reference/generated/numpy.mean.rst", "reference"),
    ],
)
def test_path_source_type_mapping(path: str, expected: str) -> None:
    assert classify_source_type(path) == expected


def test_web_link_renders_title_without_becoming_default_role_candidate() -> None:
    text, candidates, _ = rendered("Read `NumPy <https://numpy.org>`_.", "autolink")
    assert text == "Read NumPy."
    assert candidates == ()


def test_markdown_link_renders_title_without_raw_url() -> None:
    text, candidates, _ = rendered("Read the [user guide](https://example.com/guide).")
    assert text == "Read the user guide."
    assert candidates == ()


def test_paragraph_continuation_may_start_with_an_api_role() -> None:
    source = """Conversion uses
:class:`int`, :class:`float`, and :class:`complex`.
"""
    paragraphs = extract_document("numpy", "doc/source/reference/a.rst", source, None)
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "Conversion uses int, float, and complex."
    assert [candidate.raw_target for candidate in paragraphs[0].candidates] == [
        "int",
        "float",
        "complex",
    ]


def test_paragraph_continuation_may_start_with_a_non_python_role() -> None:
    source = """The corresponding C structure is the
:c:type:`PyArrayObject` used by the extension.
"""
    paragraphs = extract_document("numpy", "doc/source/reference/a.rst", source, None)
    assert [paragraph.text for paragraph in paragraphs] == [
        "The corresponding C structure is the PyArrayObject used by the extension."
    ]


def test_myst_admonition_keeps_prose_but_removes_container_markers() -> None:
    source = """:::{note}
See {class}`torch.Tensor` documentation for how to
achieve this.
:::
"""
    paragraphs = extract_document("pytorch", "docs/source/guide.md", source, None)
    assert [paragraph.text for paragraph in paragraphs] == [
        "See torch.Tensor documentation for how to achieve this."
    ]


def test_double_underscore_link_with_escaped_backtick_in_title_is_plain() -> None:
    markup = "`Issue about \\`minimize\\` <https://example.com/1>`__: fixed."
    text, candidates, _ = rendered(markup, "autolink")
    assert text == "Issue about `minimize`: fixed."
    assert candidates == ()


def test_source_reader_falls_back_to_windows_1252() -> None:
    assert decode_source_bytes("we\u2019d".encode("cp1252")) == "we\u2019d"


def candidate(target: str, role: str = "obj") -> Candidate:
    return Candidate(0, 1, role, target)


def test_resolve_exact_name_uses_inventory_type() -> None:
    result = resolve_candidate(candidate("numpy.mean", "class"), None, {"numpy.mean": {"function"}})
    assert result.status == "resolved"
    assert result.entity_type == "FUNCTION"
    assert result.resolved_name == "numpy.mean"


def test_resolve_uses_current_module_after_exact_lookup() -> None:
    result = resolve_candidate(candidate("DataFrame.groupby"), "pandas", {"pandas.DataFrame.groupby": {"method"}})
    assert result.status == "resolved"
    assert result.entity_type == "METHOD"


def test_resolve_leading_dot_requires_unique_suffix() -> None:
    result = resolve_candidate(candidate(".DataFrame.shape"), None, {"pandas.DataFrame.shape": {"attribute"}})
    assert result.status == "resolved"
    assert result.resolved_name == "pandas.DataFrame.shape"


def test_resolve_leading_dot_rejects_non_unique_suffix() -> None:
    index = {"pandas.DataFrame.shape": {"attribute"}, "other.DataFrame.shape": {"attribute"}}
    result = resolve_candidate(candidate(".DataFrame.shape"), None, index)
    assert result.status == "ambiguous"


def test_resolve_rejects_multiple_inventory_types() -> None:
    result = resolve_candidate(candidate("thing"), None, {"thing": {"class", "function"}})
    assert result.status == "ambiguous"


def test_resolve_unknown_inventory_role_is_unresolved() -> None:
    result = resolve_candidate(candidate("thing"), None, {"thing": {"object"}})
    assert result.status == "unresolved"


def test_target_normalization_strips_display_markup() -> None:
    assert normalize_candidate_target("title <~numpy.mean>") == "numpy.mean"
    assert normalize_candidate_target("!.DataFrame.shape") == ".DataFrame.shape"


def test_sentence_split_masks_entity_dots_and_abbreviations() -> None:
    paragraph = Paragraph(
        "pandas",
        "doc.rst",
        "guide",
        None,
        "Use pandas.concat(). It works, e.g. with frames. Another sentence.",
        (Candidate(4, 19, "func", "pandas.concat"),),
        (),
    )
    sentences = split_sentences(resolve_paragraph(paragraph, {"pandas.concat": {"function"}}))
    assert [sentence.text for sentence in sentences] == [
        "Use pandas.concat().",
        "It works, e.g. with frames.",
        "Another sentence.",
    ]


def test_clean_sentence_builds_tokens_tags_and_spans() -> None:
    paragraph = Paragraph(
        "pandas",
        "doc.rst",
        "guide",
        None,
        "Use pandas.concat() for these frames.",
        (Candidate(4, 19, "func", "pandas.concat"),),
        (),
    )
    sentence = split_sentences(resolve_paragraph(paragraph, {"pandas.concat": {"function"}}))[0]
    outcome = clean_sentence(sentence, {"pandas.concat"}, {"concat"})
    assert outcome.drop_rule is None
    assert outcome.cleaned is not None
    assert outcome.cleaned.tokens[:6] == ("Use", "pandas", ".", "concat", "(", ")")
    assert outcome.cleaned.tags[:6] == (
        "O",
        "B-FUNCTION",
        "I-FUNCTION",
        "I-FUNCTION",
        "I-FUNCTION",
        "I-FUNCTION",
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bad |markup| remains in this sentence.", True),
        (r"The singular values are \\sigma_i of \\mathbf{A}.", True),
        ("A leftover <style> tag remains here.", True),
        ("A Jinja {# comment #} remains here.", True),
        ("The following formula is given by", True),
        ("This sentence has enough plain alphabetic words.", False),
    ],
)
def test_markup_residue_detection(text: str, expected: bool) -> None:
    assert has_markup_residue(text) is expected


def test_cleaning_rule_order_prefers_unresolved_before_short_length() -> None:
    paragraph = Paragraph(
        "x",
        "doc.rst",
        "guide",
        None,
        "Use missing API.",
        (Candidate(4, 11, "obj", "missing"),),
        (),
    )
    sentence = split_sentences(resolve_paragraph(paragraph, {}))[0]
    assert clean_sentence(sentence, set(), set()).drop_rule == "R2"


def test_unlabeled_inventory_literal_is_r5() -> None:
    paragraph = extract_document("pandas", "doc.rst", "Use ``pandas.concat`` with these frames now.", None)[0]
    sentence = split_sentences(ResolvedParagraph(paragraph, (), False, False))[0]
    assert clean_sentence(sentence, {"pandas.concat"}, {"concat"}).drop_rule == "R5"


def test_short_sentence_is_r6() -> None:
    paragraph = Paragraph("x", "doc.rst", "guide", None, "Too short.", (), ())
    sentence = split_sentences(ResolvedParagraph(paragraph, (), False, False))[0]
    assert clean_sentence(sentence, set(), set()).drop_rule == "R6"


def test_symbol_heavy_sentence_is_r7() -> None:
    paragraph = Paragraph("x", "doc.rst", "guide", None, "A +++ === 12345 !!! B.", (), ())
    sentence = split_sentences(ResolvedParagraph(paragraph, (), False, False))[0]
    assert clean_sentence(sentence, set(), set()).drop_rule == "R7"


def build_record(
    record_id: str,
    split: str,
    text: str,
    *,
    library: str = "numpy",
    doc_path: str = "doc/source/guide.rst",
    entity: bool = True,
) -> BuildRecord:
    """Create a compact valid record for split-level unit tests."""
    spans = (
        ({"start": 1, "end": 6, "type": "FUNCTION", "target": "numpy.mean"},)
        if entity
        else ()
    )
    tokens = ("Use", "numpy", ".", "mean", "(", ")") if entity else tuple(text.split())
    tags = (
        ("O", "B-FUNCTION", "I-FUNCTION", "I-FUNCTION", "I-FUNCTION", "I-FUNCTION")
        if entity
        else tuple("O" for _ in tokens)
    )
    cleaned = CleanedSentence(library, doc_path, "guide", text, tokens, tags, spans)
    return BuildRecord(record_id, split, cleaned)


def test_split_assignment_is_document_level_and_pytorch_is_heldout() -> None:
    first = build_record("a", "train", "First text.").cleaned
    second = build_record("b", "train", "Second text.").cleaned
    assert assign_split(first) == assign_split(second)

    pytorch = build_record("c", "train", "PyTorch text.", library="pytorch").cleaned
    assert assign_split(pytorch) == "heldout"


def test_r8_keeps_highest_priority_split() -> None:
    train = build_record("train-id", "train", "Repeated text.")
    heldout = build_record("heldout-id", "heldout", "Repeated   text.", library="pytorch")
    kept, drops = deduplicate_records([train, heldout])
    assert [record.id for record in kept] == ["heldout-id"]
    assert drops == {"numpy": 1}


def test_zero_entity_sampling_is_deterministic_and_targets_twenty_percent() -> None:
    entities = [build_record(f"entity-{index}", "train", f"Entity {index}.") for index in range(8)]
    zeros = [
        build_record(f"zero-{index}", "train", f"Plain sentence number {index}.", entity=False)
        for index in range(10)
    ]
    first, first_drops = sample_zero_entity_records(entities + zeros)
    second, second_drops = sample_zero_entity_records(list(reversed(entities + zeros)))
    assert sorted(record.id for record in first) == sorted(record.id for record in second)
    assert sum(not record.cleaned.spans for record in first) == 2
    assert len(first) == 10
    assert first_drops == second_drops == {"numpy": 8}


def test_test_noparen_removes_only_terminal_function_parentheses() -> None:
    record = build_record("test-id", "test", "Use numpy.mean().")
    converted = make_test_noparen(record)
    assert converted.split == "test_noparen"
    assert converted.cleaned.tokens == ("Use", "numpy", ".", "mean")
    assert converted.cleaned.spans == (
        {"start": 1, "end": 4, "type": "FUNCTION", "target": "numpy.mean"},
    )
    assert converted.cleaned.tags == ("O", "B-FUNCTION", "I-FUNCTION", "I-FUNCTION")
