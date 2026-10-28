"""Run isolated local tests and emit/check their deterministic source receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.out and args.check:
        parser.error('choose --out or --check')
    paths = [HERE]
    bluebeam = HERE / 'handoffs/BlueBeam/research/qwenputer_v1'
    if bluebeam.exists():
        paths.append(bluebeam)
    suite = unittest.TestSuite()
    for path in paths:
        suite.addTests(unittest.TestLoader().discover(str(path), pattern='test_*.py'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    sources = sorted(p for p in HERE.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.name != 'validation.json')
    receipt = dict(schema_version=1, evidence_level='synthetic_fixture',
                   tests_run=result.testsRun, errors=0, failures=0, model_run=False,
                   source_sha256={p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources})
    text = json.dumps(receipt, indent=2, sort_keys=True) + '\n'
    if args.check:
        if args.check.read_text() != text:
            raise SystemExit('receipt or source mismatch')
        print('receipt verified')
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
