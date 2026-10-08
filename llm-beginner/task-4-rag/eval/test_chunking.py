import json
import runpy
from pathlib import Path


if __name__ == '__main__':
    checks = runpy.run_path(str(Path(__file__).with_name('run.py')))
    result = checks['test_chunking_sanity']()
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['pass'] else 1)
