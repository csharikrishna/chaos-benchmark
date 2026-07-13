"""Command-line interface: `chaos-benchmark-generate --rows_per_class 100 --out data.csv`"""

import argparse
from .generate import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate the chaos-benchmark nonlinear dynamics dataset.")
    parser.add_argument("--rows_per_class", type=int, default=100,
                         help="Target rows per (system, class) combination")
    parser.add_argument("--max_attempts", type=int, default=500,
                         help="Max simulation attempts per (system, class)")
    parser.add_argument("--out", type=str, default="chaos_benchmark_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", type=str, default=None,
                         help="Comma-separated subset, e.g. Logistic,Henon "
                              "(default: all five)")
    args = parser.parse_args()

    systems = args.systems.split(",") if args.systems else None
    df = build_dataset(rows_per_class=args.rows_per_class,
                        max_attempts_per_class=args.max_attempts,
                        seed=args.seed, systems=systems)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")
    print(df["label"].value_counts())
    print(df["system"].value_counts())


if __name__ == "__main__":
    main()
