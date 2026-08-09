#!/usr/bin/env python3
"""Controller check of the W0.4 freeze: the reference instance and every fixture under
fixtures/valid/ validate, every invalid instance
fails under the schema (or, for caught_by == quantiser, under the x-scale rule this script applies).
Run: uv run --no-project --with 'jsonschema>=4.23' python3 schema/assay/1.0/check_fixtures.py"""
import json, os, sys, decimal
HERE = os.path.dirname(os.path.abspath(__file__))
import jsonschema
from jsonschema import Draft202012Validator
schema = json.load(open(os.path.join(HERE, 'assay.schema.json')))
Draft202012Validator.check_schema(schema)
V = Draft202012Validator(schema)

def scale_violations(inst, sch, path=''):
    """Walk instance and schema together; report numbers whose digits exceed x-scale."""
    out = []
    def walk(i, s, p):
        if not isinstance(s, dict): return
        if '$ref' in s:
            ref = s['$ref'].split('/')[-1]; s = {**schema['$defs'][ref], **{k: v for k, v in s.items() if k != '$ref'}}
        for alt in s.get('oneOf', []) + s.get('anyOf', []):
            walk(i, alt, p)
        if isinstance(i, (int, float)) and not isinstance(i, bool) and 'x-scale' in s:
            d = decimal.Decimal(repr(i)); exp = -d.as_tuple().exponent if d.as_tuple().exponent < 0 else 0
            if exp > s['x-scale']: out.append(f'{p}: {i} has {exp} decimals, scale {s["x-scale"]}')
        if isinstance(i, dict):
            for k, v in i.items():
                ps = s.get('properties', {}).get(k)
                if ps is not None: walk(v, ps, f'{p}.{k}')
                elif isinstance(s.get('additionalProperties'), dict): walk(v, s['additionalProperties'], f'{p}.{k}')
        if isinstance(i, list) and isinstance(s.get('items'), dict):
            for n, v in enumerate(i): walk(v, s['items'], f'{p}[{n}]')
    walk(inst, sch, path)
    return out

ok = True

def check_valid(path, label):
    """A fixture the record admits: no schema error and no number past its x-scale."""
    global ok
    inst = json.load(open(path))
    errs = sorted(V.iter_errors(inst), key=lambda e: e.path)
    sv = scale_violations(inst, schema)
    if errs or sv:
        ok = False; print(f'{label} INVALID:'); [print('  ', list(e.path), e.message[:120]) for e in errs]; [print('  scale:', s) for s in sv]
    else:
        print(f'{label} valid (schema + quantiser)')

check_valid(os.path.join(HERE, 'fixtures', 'reference.json'), 'reference.json')
VALID_DIR = os.path.join(HERE, 'fixtures', 'valid')
for name in sorted(os.listdir(VALID_DIR)) if os.path.isdir(VALID_DIR) else []:
    if name.endswith('.json'):
        check_valid(os.path.join(VALID_DIR, name), 'valid/' + name)
idx = json.load(open(os.path.join(HERE, 'fixtures', 'invalid', 'INDEX.json')))
for name, meta in idx.items():
    inst = json.load(open(os.path.join(HERE, 'fixtures', 'invalid', name + '.json')))
    e = list(V.iter_errors(inst)); s = scale_violations(inst, schema)
    caught = bool(e) if meta['caught_by'] == 'schema' else bool(s)
    wrong_layer = (meta['caught_by'] == 'quantiser' and e)
    if caught and not wrong_layer:
        print(f'  {name:<34} rejected by {meta["caught_by"]}  ({(e[0].message if e else s[0])[:70]})')
    else:
        ok = False; print(f'  {name:<34} NOT REJECTED as expected (schema errors {len(e)}, scale {len(s)})')
print('FREEZE CHECK', 'OK' if ok else 'FAILED'); sys.exit(0 if ok else 1)
