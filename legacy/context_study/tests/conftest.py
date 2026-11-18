import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


@pytest.fixture(scope="session")
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
