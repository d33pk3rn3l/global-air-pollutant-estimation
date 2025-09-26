import pytest
import torch

from ibm.models import AddConstant


def test_add_constant():
    model = AddConstant(c=5.0)
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output_tensor = model(input_tensor)
    expected_output = torch.tensor([6.0, 7.0, 8.0])
    assert torch.allclose(
        output_tensor, expected_output
    ), f"Expected {expected_output}, but got {output_tensor}"
