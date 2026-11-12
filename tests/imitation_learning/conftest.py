"""Shared fixtures for the imitation learning tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Run each test with torch's default device on the CPU.

    gs.init() makes CUDA torch's default device for the whole process, so once an earlier test has built an
    environment, tensors these tests create without a device would land on the GPU next to CPU generators.
    """
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(previous)
