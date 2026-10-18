"""Native ASR isolation: bounded float32 packets in, caption JSON lines out."""
import json
import struct
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def emit(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


def read_exact(handle, size):
    data = bytearray()
    while len(data) < size:
        block = handle.read(size - len(data))
        if not block:
            raise EOFError('Truncated caption packet')
        data.extend(block)
    return data


def run():
    import numpy as np
    from native_speech import Recognizer
    terms = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    model = Recognizer(terms)
    model.options.word_times = False
    model.reset()
    emit({'type': 'ready'})
    try:
        while header := sys.stdin.buffer.read(4):
            if len(header) != 4:
                raise EOFError('Truncated caption packet header')
            frames, = struct.unpack('<I', header)
            if frames == 0xffffffff:
                model.reset()
                emit({'type': 'reset'})
                continue
            if frames == 0:
                for result in model.finish(): emit({'type': 'caption', **result})
                break
            if frames > 16000:
                raise ValueError('Caption packets are limited to one second')
            samples = np.frombuffer(read_exact(sys.stdin.buffer, frames * 4), dtype='<f4')
            began = time.monotonic()
            for result in model.push(samples):
                emit({'type': 'caption', 'compute_seconds': time.monotonic() - began, **result})
    finally:
        model.close()


if __name__ == '__main__':
    try: run()
    except Exception as error:
        emit({'type': 'error', 'message': f'{type(error).__name__}: {error}'})
        sys.exit(1)
