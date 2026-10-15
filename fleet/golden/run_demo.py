#!/usr/bin/env python3
"""make demo: reproduce the golden transcripts of the current wave from a clean checkout and a
clean install, byte for byte outside the fields declared in VOLATILE.md.

    python3 fleet/golden/run_demo.py                 full: git archive HEAD -> tmp, uv build, fresh venv
    python3 fleet/golden/run_demo.py --no-install    run against the CLI already on PATH (recorded as
                                                     clean_checkout=false; the terminal predicate refuses it)
    python3 fleet/golden/run_demo.py --only first-hour.txt

Writes fleet/golden/RESULTS.json. Exit 0 only when every golden of the current wave matches.
"""
import difflib, glob, json, os, re, shutil, subprocess, sys, tempfile, time, uuid
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

def sh(c, cwd=ROOT, env=None, timeout=1800):
    return subprocess.run(c, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)

def volatile():
    txt = open(os.path.join(HERE, 'VOLATILE.md'), encoding='utf-8').read()
    m = re.search(r'```json\n(.*?)\n```', txt, re.S)
    return json.loads(m.group(1))

def mask(text, fname, vol, numerals):
    import fnmatch
    for r in vol['rules']:
        if any(fnmatch.fnmatch(fname, g) for g in r['files']):
            text = re.sub(r['pattern'], r['replace'], text)
    nm = vol.get('numerals_until_demo_numbers')
    if numerals and nm and fname in nm['files']:
        text = re.sub(nm['pattern'], nm['replace'], text)
    return text

def rules_only(text, fname, vol):
    """The volatile rules without the numeral mask."""
    import fnmatch
    for r in vol['rules']:
        if any(fnmatch.fnmatch(fname, g) for g in r['files']):
            text = re.sub(r['pattern'], r['replace'], text)
    return text

def json_leafmask(v, fname, vol, numerals, key=None):
    """The volatile rules applied to a parsed JSON value, leaf by leaf. A leaf is probed in its
    `"key": value` text form as well as bare, so a rule written for the text (`"duration_s": 31`,
    `"sandbox_tier": "L2"`) masks the leaf it was written for; numbers go to `<n>` while the numeral
    mask is on (a masked number inside JSON text is not JSON, so the mask runs after parsing);
    booleans and nulls stay as they are."""
    nm = vol.get('numerals_until_demo_numbers')
    numeric = bool(numerals and nm and fname in nm['files'])
    if isinstance(v, dict):
        return {k: json_leafmask(x, fname, vol, numerals, k) for k, x in v.items()}
    if isinstance(v, list):
        return [json_leafmask(x, fname, vol, numerals, key) for x in v]
    if isinstance(v, bool) or v is None:
        return v
    if key is not None:
        probe = f'"{key}": {json.dumps(v)}'
        if rules_only(probe, fname, vol) != probe:
            return '<volatile>'
    if isinstance(v, str):
        return mask(v, fname, vol, numerals)
    if isinstance(v, (int, float)):
        return '<n>' if numeric else v
    return v

def first_pointer(l, r, path='$'):
    """The first JSON pointer at which two parsed values differ (golden on the left), or None."""
    if type(l) != type(r):
        return f'{path}: golden {l!r}, actual {r!r}'
    if isinstance(l, dict):
        for k in sorted(set(l) | set(r)):
            if k not in l:
                return f'{path}.{k}: absent from the golden, actual has {r[k]!r}'
            if k not in r:
                return f'{path}.{k}: absent from the actual, golden has {l[k]!r}'
            d = first_pointer(l[k], r[k], f'{path}.{k}')
            if d:
                return d
        return None
    if isinstance(l, list):
        if len(l) != len(r):
            return f'{path}: golden has {len(l)} items, actual {len(r)}'
        for i, (a, b) in enumerate(zip(l, r)):
            d = first_pointer(a, b, f'{path}[{i}]')
            if d:
                return d
        return None
    return None if l == r else f'{path}: golden {l!r}, actual {r!r}'

def json_same(fname, actual_parts, expected_parts, vol, numerals):
    """A `.json` golden is compared as a structure. The golden is hand-wrapped JSON and no serializer
    reproduces its line breaks, so a string comparison can never match it and would only test the
    wrapping; what the commission fixed is the values under the volatile rules. Returns (same, why)."""
    def body(parts):
        text = '\n\n'.join(parts)
        return text.split('\n', 1)[1] if text.startswith('$ ') else text
    try:
        l = json.loads(body(expected_parts)); r = json.loads(body(actual_parts))
    except ValueError as e:
        return False, f'not JSON: {e}'
    lm = json_leafmask(l, fname, vol, numerals); rm = json_leafmask(r, fname, vol, numerals)
    if lm == rm:
        return True, ''
    return False, 'structural comparison, first difference: ' + (first_pointer(lm, rm) or '?')

def parse_golden(text):
    """-> list of (command, expected_output_lines)."""
    blocks = []; cmd = None; buf = []
    for l in text.split('\n'):
        if l.startswith('$ '):
            if cmd is not None: blocks.append((cmd, buf))
            cmd = l[2:]; buf = []
        else:
            buf.append(l)
    if cmd is not None: blocks.append((cmd, buf))
    return blocks


def blocks_for(fname, text, spec):
    """A golden with `$ ` lines is parsed; one without is the whole expected output of its single command (a JSON golden)."""
    parsed = parse_golden(text)
    if parsed: return parsed
    cmds = spec.get('commands') or ([spec['command']] if spec.get('command') else [])
    return [(cmds[0], text.rstrip('\n').split('\n'))] if len(cmds) == 1 else []

def main():
    only = None; install = True
    if '--only' in sys.argv: only = sys.argv[sys.argv.index('--only') + 1]
    if '--no-install' in sys.argv: install = False
    wave = open(os.path.join(HERE, 'CURRENT_WAVE')).read().strip()
    gdir = os.path.join(HERE, wave)
    cmds = json.load(open(os.path.join(HERE, 'COMMANDS.json')))
    vol = volatile()
    numerals = not os.path.exists(os.path.join(HERE, 'DEMO_NUMBERS.md'))
    t0 = time.time(); results = {}; build_exit = install_exit = None; wheel_sha = None
    work = tempfile.mkdtemp(prefix='rl-demo-')
    repo = os.path.join(work, 'repo'); os.makedirs(repo)
    env = dict(os.environ)
    if install:
        r = sh(f'git archive HEAD | tar -x -C {repo}')
        if r.returncode: print(r.stderr); return 2
        r = sh('uv build --no-sources --wheel -o dist', cwd=repo); build_exit = r.returncode
        if r.returncode:
            print('build failed:', r.stderr[-800:]); results['_install'] = {'status': 'ERROR', 'reason': 'uv build failed'}
        else:
            venv = os.path.join(work, 'venv')
            wheels = sorted(glob.glob(os.path.join(repo, 'dist', '*.whl')))
            if wheels:
                import hashlib; wheel_sha = hashlib.sha256(open(wheels[0], 'rb').read()).hexdigest()
            r = sh(f'uv venv {venv} -q && uv pip install -q --python {venv}/bin/python dist/*.whl', cwd=repo); install_exit = r.returncode
            if r.returncode:
                print('install failed:', r.stderr[-800:]); results['_install'] = {'status': 'ERROR', 'reason': 'install failed'}
            env['PATH'] = os.path.join(venv, 'bin') + os.pathsep + env['PATH']
            env['VIRTUAL_ENV'] = venv
    else:
        r = sh(f'git archive HEAD | tar -x -C {repo}')
    env.update({'NO_COLOR': '1', 'REWARD_LENS_NO_PROGRESS': '1', 'REWARD_LENS_COLOR': 'never', 'COLUMNS': '100', 'TERM': 'dumb', 'REPO': repo})
    # A wave's goldens may sit in packet-owned subdirectories (wave-2/connect/, wave-2/trace/); every
    # golden is keyed by its basename in COMMANDS.json, so basenames must be unique across the wave.
    paths = {}
    for f in sorted(glob.glob(os.path.join(gdir, '**', '*'), recursive=True)) if os.path.isdir(gdir) else []:
        if os.path.isfile(f) and not os.path.basename(f).startswith('.'):
            if os.path.basename(f) in paths:
                print(f'two goldens share the name {os.path.basename(f)}: {paths[os.path.basename(f)]} and {f}'); return 2
            paths[os.path.basename(f)] = f
    files = sorted(paths)
    if not files:
        print(f'no golden files under {gdir}'); results['_wave'] = {'status': 'ERROR', 'reason': 'empty golden dir'}
    for fname in files:
        if only and fname != only: continue
        spec = cmds.get(fname, {})
        if spec.get('skip'):
            results[fname] = {'status': 'SKIP', 'reason': spec['skip']}; print(f'{fname:<24} SKIP  {spec["skip"]}'); continue
        golden = open(paths[fname], encoding='utf-8').read().rstrip('\n')
        blocks = blocks_for(fname, golden, spec)
        if spec.get('commands'):
            blocks = [(spec['commands'][i] if i < len(spec['commands']) else c, o) for i, (c, o) in enumerate(blocks)] if blocks else [(c, []) for c in spec['commands']]
        cwd = repo if spec.get('cwd') == 'repo' else tempfile.mkdtemp(prefix='rl-golden-', dir=work)
        e = dict(env); e.update(spec.get('env', {}))
        if fname.endswith('.txt'): e['REWARD_LENS_FORMAT'] = 'text'
        ok = True
        for s in spec.get('setup', []):
            r = sh(s, cwd=cwd, env=e)
            if r.returncode: ok = False; results[fname] = {'status': 'ERROR', 'reason': f'setup failed: {s} -> {r.stderr.strip()[-200:]}'}; break
        if not ok: print(f'{fname:<24} ERROR {results[fname]["reason"][:90]}'); continue
        actual_parts = []; expected_parts = []; last_rc = None
        for cmd, expected in blocks:
            if any(cmd.startswith(n) for n in vol.get('noexec_commands', [])):
                continue  # volatile by declaration: the command and its output are excluded
            r = sh(cmd, cwd=cwd, env=e, timeout=1200); last_rc = r.returncode
            out = r.stderr if spec.get('stream') == 'stderr' else r.stdout
            actual_parts.append('$ ' + cmd + '\n' + out.rstrip('\n'))
            expected_parts.append('$ ' + cmd + '\n' + '\n'.join(expected).rstrip('\n'))
        a = mask('\n\n'.join(actual_parts), fname, vol, numerals); x = mask('\n\n'.join(expected_parts), fname, vol, numerals)
        exp_rc = spec.get('exit')
        same, why = (json_same(fname, actual_parts, expected_parts, vol, numerals) if fname.endswith('.json') else (a == x, ''))
        if same and (exp_rc is None or exp_rc == last_rc):
            results[fname] = {'status': 'MATCH', 'exit': last_rc, 'expected_exit': exp_rc if exp_rc is not None else last_rc}; print(f'{fname:<24} MATCH')
        else:
            d = why if fname.endswith('.json') else '\n'.join(list(difflib.unified_diff(x.split('\n'), a.split('\n'), 'golden', 'actual', lineterm=''))[:60])
            results[fname] = {'status': 'DIFF' if not same else 'EXIT', 'exit': last_rc, 'expected_exit': exp_rc, 'diff_head': d}
            print(f'{fname:<24} {"DIFF" if not same else "EXIT"}  exit {last_rc} (expected {exp_rc})'); print(d[:1500])
    doc = {'schema': 'fleet-demo-results/1', 'run_id': uuid.uuid4().hex[:12], 'commit': sh('git rev-parse HEAD').stdout.strip(),
           'dirty': bool(sh('git status --porcelain --untracked-files=no').stdout.strip()), 'untracked_clean': not bool(sh('git status --porcelain --untracked-files=all').stdout.strip()), 'clean_checkout': install,
           'build_exit': build_exit, 'install_exit': install_exit, 'wheel_sha256': wheel_sha,
           'wave': wave, 'numerals_masked': numerals, 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0)),
           'seconds': round(time.time() - t0, 1), 'partial': bool(only), 'results': results}
    out = os.path.join(HERE, 'RESULTS.partial.json' if only else 'RESULTS.json')
    json.dump(doc, open(out, 'w'), indent=1)
    shutil.rmtree(work, ignore_errors=True)
    bad = [k for k, v in results.items() if v['status'] != 'MATCH']
    print(f"demo: {len(results) - len(bad)} match, {len(bad)} not  -> {os.path.relpath(out, ROOT)}")
    return 0 if not bad else 1

if __name__ == '__main__':
    sys.exit(main())
