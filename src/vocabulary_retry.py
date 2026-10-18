"""Selective glossary retries with agreement and a strict single-word acceptance gate."""
import difflib
import gc
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'state/vocabulary_asr'
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def available():
    return (MODEL_PATH / 'model.safetensors').is_file()


def normalized(text):
    return [match.group().lower().replace('’', "'") for match in WORD.finditer(text)]


def target(text, terms):
    tokens = list(WORD.finditer(text))
    known = {term.lower() for term in terms}
    choices = []
    for index, token in enumerate(tokens):
        word = token.group().lower()
        if len(word) < 6 or word in known or not word.isalpha(): continue
        for term in terms:
            if len(normalized(term)) != 1 or not term.isalpha() or abs(len(term) - len(word)) > 2: continue
            score = difflib.SequenceMatcher(None, word, term.lower()).ratio()
            if .8 <= score < 1:
                choices.append((score, index, token.start(), token.end(), term))
    if not choices: return None
    choices.sort(reverse=True)
    if len(choices) > 1 and choices[0][0] == choices[1][0] and choices[0][4] != choices[1][4]:
        return None  # no arbitrary resolution of an ambiguous scientific term
    _, index, start, end, term = choices[0]
    return {'index': index, 'char_start': start, 'char_end': end, 'term': term}


def accepted_text(original, selection, unprompted, contextual):
    """Accept only paired agreement changing exactly the selected known glossary word."""
    before, first, second = normalized(original), normalized(unprompted), normalized(contextual)
    if first != second or len(first) != len(before): return None
    index = selection['index']
    if first[index] != selection['term'].lower(): return None
    if any(a != b for i, (a, b) in enumerate(zip(before, first)) if i != index): return None
    if before[index] == first[index]: return None
    return original[:selection['char_start']] + selection['term'] + original[selection['char_end']:]


class VocabularyASR:
    def __init__(self):
        from health import _read_memory
        if _read_memory()[0] < 3 * 2**30:
            raise MemoryError('Vocabulary retry needs 3 GiB available RAM')
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, device_map='cuda',
            attn_implementation='sdpa').eval()

    def transcribe(self, samples, terms=()):
        prompt = ('Relevant vocabulary: ' + ', '.join(list(terms)[:64])) if terms else None
        inputs = self.processor.apply_transcription_request(audio=[samples], language='English',
                                                            prompt=prompt, return_tensors='pt')
        inputs = inputs.to('cuda', dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return self.processor.decode(ids[0, inputs['input_ids'].shape[-1]:], return_format='transcription_only')

    def close(self):
        self.model = self.processor = None
        gc.collect()
        self.torch.cuda.empty_cache()


def retry_session(session, terms, *, factory=VocabularyASR, max_pieces=8, budget_seconds=60):
    selections = [(segment, found) for segment in session.all_segments
                  if (found := target(segment['text'], terms)) is not None][:max_pieces]
    report = {'attempted': 0, 'accepted': 0, 'reason': None}
    if not selections or not available(): return report
    model = None
    began = time.monotonic()
    try:
        model = factory()
        for segment, selection in selections:
            if time.monotonic() - began > budget_seconds:
                report['reason'] = 'Retry budget reached'
                break
            start, finish = round(segment['start'] * 16000), round(segment['end'] * 16000)
            samples = session.read_audio(start, finish - start)
            original = segment['text']
            unprompted = model.transcribe(samples)
            selected_terms = [selection['term']] + [t for t in terms if t.lower() != selection['term'].lower()]
            contextual = model.transcribe(samples, selected_terms)
            replacement = accepted_text(original, selection, unprompted, contextual)
            report['attempted'] += 1
            segment['vocabulary_retry'] = {'term': selection['term'], 'original_text': original,
                                           'unprompted': unprompted, 'contextual': contextual,
                                           'accepted': replacement is not None}
            if replacement is not None:
                segment['text'] = replacement
                segment.pop('words', None)
                report['accepted'] += 1
    except Exception as error:
        report['reason'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            if report['attempted']:
                from storage import atomic_write
                atomic_write(session.segments_path, ''.join(json.dumps(s, ensure_ascii=False) + '\n'
                                                           for s in session.all_segments))
        except Exception as error:
            report['reason'] = f'{type(error).__name__}: {error}'
        finally:
            if model is not None: model.close()
    return report
