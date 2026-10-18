"""Pinned NeMo-Speech C ABI. Used inside the isolated caption worker only."""
import ctypes as C
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'state/nemo_speech/nemo-speech-0.2.0-windows-x86_64-cuda/bin'
MODEL = ROOT / 'state/nemo_speech/models/nemotron-en.q8_0.gguf'


def available():
    return (BIN / 'nemo_speech_asr_c.dll').is_file() and MODEL.is_file()


class Backend(C.Structure):
    _fields_ = [('size', C.c_size_t), ('gpu', C.c_int32)]


class Model(C.Structure):
    _fields_ = [('size', C.c_size_t), ('path', C.c_char_p), ('name', C.c_char_p)]


class Streaming(C.Structure):
    _fields_ = [('size', C.c_size_t), ('chunk_size', C.c_float), ('left', C.c_float),
                ('right', C.c_float), ('rnnt_right_context', C.c_int32)]


class Config(C.Structure):
    _fields_ = [('size', C.c_size_t)] + [(name, C.c_void_p) for name in
        ['backend', 'model', 'streaming', 'decoder', 'vad', 'endpointing', 'postproc', 'diar', 'batching']]


class SpeechContext(C.Structure):
    _fields_ = [('size', C.c_size_t), ('phrases', C.POINTER(C.c_char_p)),
                ('phrase_count', C.c_size_t), ('boost', C.c_float)]


class Options(C.Structure):
    _fields_ = [('size', C.c_size_t), ('request_id', C.c_char_p), ('language', C.c_char_p),
                ('interim', C.c_bool), ('word_times', C.c_bool), ('punctuation', C.c_bool),
                ('verbatim', C.c_bool), ('profanity', C.c_bool), ('eou_ms', C.c_int32),
                ('contexts', C.POINTER(SpeechContext)), ('context_count', C.c_size_t),
                ('alternatives', C.c_int32), ('diarization', C.c_bool), ('speakers', C.c_int32)]


class Recognizer:
    def __init__(self, terms=(), gpu=-1):
        from health import _read_memory
        if _read_memory()[0] < 2 * 2**30:
            raise MemoryError('Native captions need 2 GiB available RAM')
        self.directory = os.add_dll_directory(str(BIN))
        self.lib = C.CDLL(str(BIN / 'nemo_speech_asr_c.dll'))
        ptr = C.c_void_p
        signatures = {
            'create': ([C.POINTER(Config), C.POINTER(ptr)], C.c_int),
            'destroy': ([ptr], None),
            'last_error': ([], C.c_char_p),
            'recognition_options_default': ([], Options),
            'streaming_recognize': ([ptr, C.POINTER(Options), C.POINTER(ptr)], C.c_int),
            'stream_push_f32': ([ptr, C.POINTER(C.c_float), C.c_size_t, C.c_int32], C.c_int),
            'stream_next': ([ptr, C.POINTER(ptr)], C.c_int),
            'stream_finish': ([ptr], C.c_int), 'stream_close': ([ptr], None),
            'recognize_f32': ([ptr, C.POINTER(Options), C.POINTER(C.c_float), C.c_size_t,
                               C.c_int32, C.POINTER(ptr)], C.c_int),
            'result_destroy': ([ptr], None), 'result_is_final': ([ptr], C.c_bool),
            'result_audio_processed': ([ptr], C.c_float),
            'result_transcript': ([ptr, C.c_size_t], C.c_char_p),
            'result_word_count': ([ptr, C.c_size_t], C.c_size_t),
            'result_word_text': ([ptr, C.c_size_t, C.c_size_t], C.c_char_p),
            'result_word_start_time': ([ptr, C.c_size_t, C.c_size_t], C.c_int32),
            'result_word_end_time': ([ptr, C.c_size_t, C.c_size_t], C.c_int32),
            'result_word_confidence': ([ptr, C.c_size_t, C.c_size_t], C.c_float),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, 'nemo_speech_asr_' + name)
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)
        self.handle = ptr()
        self.stream = ptr()
        self._path = os.fsencode(MODEL)
        backend = Backend(C.sizeof(Backend), gpu)
        model = Model(C.sizeof(Model), self._path, None)
        streaming = Streaming(C.sizeof(Streaming), .16, 0, 0, 1)
        config = Config(size=C.sizeof(Config), backend=C.cast(C.pointer(backend), ptr),
                        model=C.cast(C.pointer(model), ptr), streaming=C.cast(C.pointer(streaming), ptr))
        try:
            self.check(self.create(C.byref(config), C.byref(self.handle)))
            self.options = self.recognition_options_default()
            self.options.interim = self.options.word_times = True
            self.options.verbatim = True
            self._terms = [t.encode('utf-8') for t in list(terms)[:64]]
            self._phrases = (C.c_char_p * len(self._terms))(*self._terms)
            self._context = SpeechContext(C.sizeof(SpeechContext), self._phrases, len(self._terms), 3.0)
            if self._terms:
                self.options.contexts = C.pointer(self._context)
                self.options.context_count = 1
            self.reset()
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError((self.last_error() or b'Native ASR failed').decode('utf-8', 'replace'))

    def reset(self):
        if self.stream:
            self.stream_close(self.stream)
        self.stream = C.c_void_p()
        self.check(self.streaming_recognize(self.handle, C.byref(self.options), C.byref(self.stream)))

    def _result(self, result):
        try:
            words = []
            for i in range(self.result_word_count(result, 0)):
                words.append({'text': (self.result_word_text(result, 0, i) or b'').decode('utf-8'),
                              'start': self.result_word_start_time(result, 0, i) / 1000,
                              'end': self.result_word_end_time(result, 0, i) / 1000,
                              'confidence': self.result_word_confidence(result, 0, i)})
            return {'text': (self.result_transcript(result, 0) or b'').decode('utf-8'),
                    'final': self.result_is_final(result),
                    'audio_seconds': self.result_audio_processed(result), 'words': words}
        finally:
            self.result_destroy(result)

    def poll(self):
        results = []
        while True:
            result = C.c_void_p()
            self.check(self.stream_next(self.stream, C.byref(result)))
            if not result:
                break
            results.append(self._result(result))
        return results

    def push(self, samples):
        import numpy as np
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        self.check(self.stream_push_f32(self.stream, samples.ctypes.data_as(C.POINTER(C.c_float)), len(samples), 16000))
        return self.poll()

    def finish(self):
        self.check(self.stream_finish(self.stream))
        return self.poll()

    def recognize(self, samples):
        import numpy as np
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        result = C.c_void_p()
        self.check(self.recognize_f32(self.handle, C.byref(self.options),
            samples.ctypes.data_as(C.POINTER(C.c_float)), len(samples), 16000, C.byref(result)))
        return self._result(result) if result else {'text': '', 'words': [], 'final': True}

    def close(self):
        if self.stream:
            self.stream_close(self.stream)
            self.stream = C.c_void_p()
        if self.handle:
            self.destroy(self.handle)
            self.handle = C.c_void_p()
        self.directory.close()
