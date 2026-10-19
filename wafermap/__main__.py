import argparse
import json

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Wafer-map classification, yield modelling and SPC demo")
    parser.add_argument("--out", default="reports", help="output folder for figures and metrics.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="smaller dataset for a fast smoke test")
    args = parser.parse_args()

    metrics = run(args.out, seed=args.seed, quick=args.quick)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
