import pytest
import alignment
import diarize
import main


def test_word_turns_preserve_punctuation_and_questions():
    segment = {'start': 10, 'end': 15, 'text': 'Quiz? Yes, only two.'}
    timed = alignment.attach_words(segment, [
        {'text': 'Quiz', 'start_time': 0, 'end_time': .5},
        {'text': 'Yes', 'start_time': 1, 'end_time': 1.4},
        {'text': 'only', 'start_time': 2, 'end_time': 2.4},
        {'text': 'two', 'start_time': 3, 'end_time': 3.4}])
    turns = diarize.assign_speakers([timed], [
        {'start': 10, 'end': 10.7, 'speaker': 'SPEAKER_01'},
        {'start': 10.8, 'end': 15, 'speaker': 'SPEAKER_00'}])
    assert [turn['text'] for turn in turns] == ['Quiz?', 'Yes, only two.']
    assert [turn['speaker'] for turn in turns] == ['SPEAKER_01', 'SPEAKER_00']
    assert timed['text'] == segment['text']


@pytest.mark.parametrize('text,start,end', [('other', 0, 1), ('Test', -1, 1),
                                         ('Test', 1, .5), ('Test', 0, 4)])
def test_invalid_alignment_does_not_change_original(text, start, end):
    original = {'text': 'Test.', 'start': 0, 'end': 2}
    with pytest.raises(ValueError):
        alignment.attach_words(original, [{'text': text, 'start_time': start, 'end_time': end}])
    assert 'words' not in original and original['text'] == 'Test.'


def test_alignment_failure_leaves_durable_original(tmp_path, monkeypatch):
    monkeypatch.setattr(alignment, 'available', lambda: True)
    session = main.Session(tmp_path / 'test.wav')
    session.record_segments([{'text': 'Saved words.', 'start': 0, 'end': 1}], 0)
    before = session.segments_path.read_text()
    def unavailable(): raise RuntimeError('model unavailable')
    result = alignment.align_session(session, factory=unavailable)
    session.close()
    assert result['failed'] == 1 and 'model unavailable' in result['reason']
    assert session.segments_path.read_text() == before
