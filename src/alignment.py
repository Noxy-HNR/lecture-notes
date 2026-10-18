"""Local word timing for the accepted transcript; never a second transcription."""
import gc
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'state' / 'forced_aligner'
MODE = 'auto'


def available():
    return MODE != 'off' and (MODEL_PATH / 'model.safetensors').is_file()


def _clean(word):
    return ''.join(c for c in word if c == "'" or unicodedata.category(c)[0] in 'LN')


def attach_words(segment, predictions):
    """Validate timings and map the aligner's stripped tokens back to exact source spans."""
    spans = [match for match in re.finditer(r'\S+', segment['text']) if _clean(match.group())]
    if len(spans) != len(predictions):
        raise ValueError('Alignment word count differs from the transcript')
    duration = segment['end'] - segment['start']
    words, previous = [], 0.0
    for span, result in zip(spans, predictions):
        if _clean(span.group()) != result['text']:
            raise ValueError('Alignment changed the transcript word sequence')
        start, end = float(result['start_time']), float(result['end_time'])
        if not all(map(math.isfinite, (start, end))) or not 0 <= previous <= start <= min(end, duration) or end > duration + .05:
            raise ValueError('Alignment timestamps are invalid or outside the audio')
        previous = min(end, duration)
        words.append({'text': span.group(), 'start': segment['start'] + start,
                      'end': segment['start'] + min(end, duration),
                      'char_start': span.start(), 'char_end': span.end()})
    return {**segment, 'words': words, 'alignment': 'qwen3-forced-aligner-0.6b'}


class ForcedAligner:
    def __init__(self):
        from health import _read_memory
        if _read_memory()[0] < 3 * 2**30:
            raise MemoryError('Word alignment needs 3 GiB available RAM')
        import torch
        from transformers import AutoProcessor, Qwen3ASRForTokenClassification
        if not torch.cuda.is_available():
            raise RuntimeError('Word alignment requires CUDA')
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = Qwen3ASRForTokenClassification.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='sdpa').eval()

    def align(self, samples, segment):
        inputs, words = self.processor.prepare_forced_aligner_inputs(
            audio=[samples], transcript=[segment['text']], language=['English'], return_tensors='pt')
        inputs = inputs.to('cuda', dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        results = self.processor.decode_forced_alignment(
            logits, inputs['input_ids'], words, self.model.config.timestamp_token_id,
            self.model.config.timestamp_segment_time)
        return attach_words(segment, results[0])

    def close(self):
        self.model = self.processor = None
        gc.collect()
        self.torch.cuda.empty_cache()


def align_session(session, factory=ForcedAligner):
    """One bounded piece at a time, with original text retained on any failure."""
    if not available() or not session.all_segments:
        return {'aligned': 0, 'failed': 0, 'reason': 'disabled or assets unavailable'}
    aligner = None
    aligned = failed = 0
    reason = None
    try:
        aligner = factory()
        for segment in session.all_segments:
            try:
                begin, finish = round(segment['start'] * 16000), round(segment['end'] * 16000)
                samples = session.read_audio(begin, finish - begin)
                result = aligner.align(samples, segment)
                segment.update(result)
                aligned += 1
            except ValueError as error:
                failed += 1
                reason = str(error)
        if aligned:
            import json
            from storage import atomic_write
            atomic_write(session.segments_path, ''.join(json.dumps(s, ensure_ascii=False) + '\n'
                                                       for s in session.all_segments))
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        failed += len(session.all_segments) - aligned - failed
    finally:
        if aligner is not None:
            aligner.close()
    return {'aligned': aligned, 'failed': failed, 'reason': reason}
