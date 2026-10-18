"""Download hash-pinned public evaluation assets into ignored state (no installs).

Existing files must match their expected hash; mismatches fail without overwriting.
Does not download ASR weights, datasets, or read any authentication stores.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "state/efficiency_experiment"
ASSETS = [{'destination': 'nano/nano_cohere_transcribe/__init__.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/__init__.py',
  'sha256': '86d49f631afd33929b29c50b8650cf2e921ec493d2ccee6b8f3c08de3cdaa951'},
 {'destination': 'nano/nano_cohere_transcribe/_loader.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/_loader.py',
  'sha256': 'c98f1ce114e28e362e5e3730e3adfae9cea6c68caf8f2a475409b79893cead5c'},
 {'destination': 'nano/nano_cohere_transcribe/api.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/api.py',
  'sha256': 'a129d6e4084888bebd8c7446c0a5e8636dca11aef150e74bff16130381eb1b63'},
 {'destination': 'nano/nano_cohere_transcribe/audio.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/audio.py',
  'sha256': 'a2bafb2eb8466f105fd24a3164733d2aeef86f24986280c638546a04887b40e6'},
 {'destination': 'nano/nano_cohere_transcribe/chunk.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/chunk.py',
  'sha256': '7fc85bd88c7201bbc24d41336de0c46d4756b97b18f8ecb363562b28b6a17002'},
 {'destination': 'nano/nano_cohere_transcribe/generate.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/generate.py',
  'sha256': '23461e31bf41324d9332d8464b3c01fa8ef0993286eea4642232109fa52dbeb2'},
 {'destination': 'nano/nano_cohere_transcribe/mel.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/mel.py',
  'sha256': '06a9820fa4552f24b6cf162699e3c2db14dbb61fec0ea38ffd0a804c9d3afdac'},
 {'destination': 'nano/nano_cohere_transcribe/model.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/model.py',
  'sha256': 'f8e11f53610d2f170879bdd7da79c87de0d2b34c0baeaabad4fc8a08cd4fd196'},
 {'destination': 'nano/nano_cohere_transcribe/tokenizer.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/tokenizer.py',
  'sha256': '9dbfcc747c5193b385843af059bc93ab2abc439953026b3673f78f5626d66295'},
 {'destination': 'nano/nano_cohere_transcribe/_graph.py',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/nano_cohere_transcribe/_graph.py',
  'sha256': 'e8d25fef6943b898bfce609a98b5cf9ef82c9368e13947a3969bdb82db95bd8c'},
 {'destination': 'nano/LICENSE',
  'url': 'https://raw.githubusercontent.com/Deep-unlearning/nano-cohere-transcribe/f8e1cd94554a812f35d56dbdc42c461c681da774/LICENSE',
  'sha256': '7ecd8ce1d30b8aa26232f5c7c878cca53bc273547ce52f1678f955154746e64f'},
 {'destination': 'silero/utils_vad.py',
  'url': 'https://raw.githubusercontent.com/snakers4/silero-vad/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/utils_vad.py',
  'sha256': '7e0f9b3334ee4feb79abcf2251be12458329077bce28c8540878ed9d35e5aa6b'},
 {'destination': 'silero/LICENSE',
  'url': 'https://raw.githubusercontent.com/snakers4/silero-vad/1e261b036686cd0017d500ee96acd1c4ba572a9d/LICENSE',
  'sha256': '2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b'},
 {'destination': 'silero/silero_vad.onnx',
  'url': 'https://raw.githubusercontent.com/snakers4/silero-vad/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/data/silero_vad.onnx',
  'sha256': '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'}]


def main():
    for asset in ASSETS:
        path = ROOT / asset["destination"]
        data = path.read_bytes() if path.exists() else urllib.request.urlopen(asset["url"], timeout=60).read()
        if hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise RuntimeError("Hash mismatch: " + asset["destination"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        print("Verified: " + asset["destination"])
    metadata = ROOT / "assets_manifest.json"
    if not metadata.exists():
        metadata.write_text(json.dumps(ASSETS, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
