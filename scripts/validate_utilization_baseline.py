import argparse
import json
from pathlib import Path

from streaming_rl_feature_geometry.experiment import write_json
from streaming_rl_feature_geometry.utilization_experiment import dry_run_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = dry_run_report(["stage-a"], baseline_root=Path(args.baseline_root))
    value = {
        "status": "compatible" if report["baseline_ready"] else "not_fully_compatible",
        "logical_cells": report["logical_cells"],
        "reused_identity_cells": report["reusable_identity_cells"],
        "missing_baseline_cells": report["missing_baseline_cells"],
        "incompatible_baseline_cells": report["incompatible_baseline_cells"],
        "duplicate_cells": report["duplicate_cells"],
        "pending_runs": report["new_pending_runs"],
        "records": report["reuse_validation"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
