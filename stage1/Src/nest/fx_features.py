""" PyTorch FX Based Feature Extraction Helpers
Using https://pytorch.org/vision/stable/feature_extraction.html
"""
from typing import Callable


def register_notrace_function(func: Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    return func
