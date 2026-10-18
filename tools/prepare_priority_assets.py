"""Download pinned public assets into state/, without touching credentials or models in use."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import os
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'aligner': ('Qwen/Qwen3-ForcedAligner-0.6B-hf', 'c07281df297b9905d24a508279258cccf987a064', 'forced_aligner'),
    'vocabulary': ('Qwen/Qwen3-ASR-0.6B-hf', '7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c', 'vocabulary_asr'),
}


def prepare_aligner(asset='aligner'):
    from huggingface_hub import snapshot_download
    repo, revision, folder = ASSETS[asset]
    target = ROOT / 'state' / folder
    snapshot_download(repo, revision=revision, local_dir=target, token=False,
                      max_workers=1, allow_patterns=['*.json', '*.jinja', '*.safetensors', 'README.md'])
    (target / 'provenance.json').write_text(json.dumps({'repo': repo, 'revision': revision}, indent=2))
    print(f'Prepared {target}')


def prepare_native():
    target = ROOT / 'state' / 'nemo_speech'
    target.mkdir(parents=True, exist_ok=True)
    name = 'nemo-speech-0.2.0-windows-x86_64-cuda.zip'
    url = 'https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/' + name
    with urllib.request.urlopen(url + '.sha256', timeout=30) as response:
        expected = response.read().decode().split()[0]
    archive = target / name
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as output:
            while block := source.read(2**20):
                output.write(block)
        temporary.replace(archive)
    with archive.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError('Native runtime archive checksum mismatch')
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            resolved = (target / entry.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe archive member')
        zipped.extractall(target)
    (target / 'provenance.json').write_text(json.dumps({'release': 'v0.2.0', 'url': url, 'sha256': actual}, indent=2))
    model = target / 'models/nemotron-en.q8_0.gguf'
    if not model.exists():
        cache = target / 'downloaded_models'
        exe = target / 'nemo-speech-0.2.0-windows-x86_64-cuda/bin/nemo-speech.exe'
        subprocess.run([str(exe), 'pull', 'nemotron-en'], check=True, timeout=600,
                       env=dict(os.environ, NEMO_SPEECH_MODEL_DIR=str(cache)))
        source = cache / ('nvidia/nemotron-speech-streaming-en-0.6b/'
                         'ebe59e5a817142986528bbbee5dba8db7b38ed50/nemotron-speech-streaming-en-0.6b.q8_0.gguf')
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, model)
    print(f'Prepared {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=['aligner', 'vocabulary', 'native'])
    args = parser.parse_args()
    if args.asset == 'native': prepare_native()
    else: prepare_aligner(args.asset)
