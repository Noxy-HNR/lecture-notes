import pytest
import vocabulary_retry as retry


def test_agreed_single_known_term_preserves_source_punctuation():
    text = 'It is exegonic, with 2 samples; not all samples.'
    selection = retry.target(text, ['exergonic'])
    result = retry.accepted_text(text, selection,
        'It is exergonic with 2 samples not all samples',
        'It is exergonic, with 2 samples; not all samples.')
    assert result == 'It is exergonic, with 2 samples; not all samples.'


@pytest.mark.parametrize('other', [
    'It is exergonic with 3 samples not all samples',
    'It is exergonic with 2 samples all samples',
    'It is exergonic with 2 samples not most samples',
    'This is exergonic with 2 samples not all samples'])
def test_quantity_negation_scope_or_other_word_changes_are_rejected(other):
    text = 'It is exegonic with 2 samples not all samples'
    assert retry.accepted_text(text, retry.target(text, ['exergonic']), other, other) is None


def test_disagreement_and_ambiguous_glossary_are_rejected():
    text = 'This is exegonic.'
    selection = retry.target(text, ['exergonic'])
    assert retry.accepted_text(text, selection, 'This is exergonic', 'This is endergonic') is None
    assert retry.target('abcdeg', ['abcdef', 'abcdeh']) is None
    assert retry.target('exergonic', ['exergonic']) is None
