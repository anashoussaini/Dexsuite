# tests/robot/conftest.py
import os


# Mark slow integration tests so you can -m "not slow" locally if needed
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow/integration")


# Optional: reduce noisy logs in CI
os.environ.setdefault("GENESIS_QUIET", "1")
