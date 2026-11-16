#!/bin/bash
# model files for the tests / self-test -> $1 (default models/): both served graphs + vocabs + the manifest
# Manifest from Nest (fallback: the GitHub models-v1 copy); files from each model's base_urls (Hugging Face, then Nest),
# then the top-level base_urls.
set -euo pipefail
OUT="${1:-models}"
NEST=https://autocorrect.mostlime.hackclub.app/autocorrect-android
mkdir -p "$OUT"
curl -fsSL --retry 3 "$NEST/manifest.json?cb=$RANDOM$RANDOM" -o "$OUT/manifest.json" \
  || curl -fsSL --retry 3 "https://github.com/Mostlime12195/keylime-models/releases/download/models-v1/manifest.json?cb=$RANDOM$RANDOM" -o "$OUT/manifest.json"
python3 - "$OUT" "$NEST/" "$(dirname "$0")/m3_entry.json" <<'PY'
import hashlib, json, os, subprocess, sys
out, nest, local_v2 = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(f'{out}/manifest.json'))
# models_v2 (m3): the live manifest's entries; until it has one, the entry pinned in scripts/m3_entry.json (written into
# the local copy only, so the self-test sees it)
if not any(v.get('type') == 'm3dict' for v in m.get('models_v2', [])) and os.path.exists(local_v2):
    m.setdefault('models_v2', []).append(json.load(open(local_v2)))
    json.dump(m, open(f'{out}/manifest.json', 'w'), indent=1)
    print('models_v2: using', local_v2)
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def get(spec, files):
    bases = list(dict.fromkeys(spec.get('base_urls', []) + m.get('base_urls', []) + [nest]))
    for f in files:
        p = f'{out}/{f["file"]}'
        ok = os.path.exists(p) and sha(p) == f['sha256'].lower()
        for base in bases:
            if ok: break
            r = subprocess.run(['curl', '-fsSL', '--retry', '3', f'{base}{f["file"]}', '-o', p])
            ok = r.returncode == 0 and sha(p) == f['sha256'].lower()
            if not ok: print('miss', base, f['file'], file=sys.stderr)
        assert ok, f'no mirror has {f["file"]} with sha256 {f["sha256"]}'
        print('ok', p, os.path.getsize(p))
for spec in m['models']:
    if 'graph' in spec and 'vocab' in spec: get(spec, [spec['graph'], spec['vocab']])
for v in m.get('models_v2', []):
    if v.get('type') != 'm3dict': continue
    fs = v['files']
    if isinstance(fs, dict): fs = [dict(f, file=f.get('file', k)) for k, f in fs.items()]
    # the context encoder isn't used by the app yet
    get(v, [f for f in fs if 'encoder_ctx' not in f['file']])
PY
