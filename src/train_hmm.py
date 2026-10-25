"""Train and evaluate the APILens hidden Markov model baseline."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
