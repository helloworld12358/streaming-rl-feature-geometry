"""Validate a completed experiment directory."""

from __future__ import annotations

import argparse
import json

from streaming_rl_feature_geometry.experiment import load_config, validate_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--config")
    arguments = parser.parse_args()
    config = load_config(arguments.config) if arguments.config else None
    print(json.dumps(validate_results(arguments.result_dir, config), indent=2))


if __name__ == "__main__":
    main()
