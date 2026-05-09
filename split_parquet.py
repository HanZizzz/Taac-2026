"""Split a Parquet file into two Parquet files.

Usage:
  python split_parquet.py --input /path/to/input.parquet \
    --output-dir /path/to/out \
    --prefix demo_1000 \
    --ratio 0.9

Notes:
- Splits by row count; preserves schema.
- Writes output files: <prefix>_part1.parquet and <prefix>_part2.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a Parquet file into two files")
    parser.add_argument("--input", required=True, help="Input parquet file path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="split", help="Output file prefix")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="Fraction of rows to put in part1 (0-1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if args.ratio <= 0 or args.ratio >= 1:
        raise ValueError("--ratio must be between 0 and 1 (exclusive)")

    table = pq.read_table(input_path)
    total_rows = table.num_rows
    split_rows = int(total_rows * args.ratio)

    if split_rows <= 0 or split_rows >= total_rows:
        raise ValueError("Split size results in empty output; adjust --ratio")

    part1 = table.slice(0, split_rows)
    part2 = table.slice(split_rows, total_rows - split_rows)

    out1 = output_dir / f"{args.prefix}_part1.parquet"
    out2 = output_dir / f"{args.prefix}_part2.parquet"

    pq.write_table(part1, out1)
    pq.write_table(part2, out2)

    print(f"Wrote {out1} ({part1.num_rows} rows)")
    print(f"Wrote {out2} ({part2.num_rows} rows)")


if __name__ == "__main__":
    main()
