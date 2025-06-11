# filepath: tests/test_run_mpc.py
import pytest
from scripts import run_mpc

def test_main_runs_without_exception():
    try:
        run_mpc.main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")