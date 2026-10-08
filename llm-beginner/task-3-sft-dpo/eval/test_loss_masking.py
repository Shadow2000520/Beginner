import json
from pathlib import Path
import runpy


if __name__ == '__main__':
    checks = runpy.run_path(str(Path(__file__).with_name('run.py')))
    result = checks['test_loss_masking']()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['pass'] is True else 1)
