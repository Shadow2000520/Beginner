from pathlib import Path
import runpy


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    checks = runpy.run_path(str(root / 'eval/run.py'))
    results = checks['run_tests']([checks['test_tools_individual']], root)
    raise SystemExit(0 if all(item['pass'] for item in results) else 1)
