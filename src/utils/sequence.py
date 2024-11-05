import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

from typing import List

from src.utils.sample import Sample


class Sequence:
    """
    A sequence is a series of consecutive samples.
    """

    def __init__(self, sequence: List[Sample]) -> None:
        """
        Initialize the sequence.

        Args:
            sequence (List[Sample]): The sequence of samples.
        """
        self.sequence = sequence

    def __len__(self) -> int:
        """
        The length of the sequence.

        Returns:
            int: The length of the sequence.
        """
        return len(self.sequence)
