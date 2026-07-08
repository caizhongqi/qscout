"""Build deterministic stratified CyberSecEval autocomplete subsets."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("third_party/purplellama/CybersecurityBenchmarks/datasets/autocomplete/autocomplete.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default="data_public/cyberseceval_autocomplete_subset_240.json")
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260708)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("cwe_identifier", "CWE-UNKNOWN")), str(row.get("language", "text")))
        buckets[key].append(row)
    for items in buckets.values():
        items.sort(key=lambda item: int(item.get("prompt_id", 0)))
        rng.shuffle(items)

    selected: list[dict] = []
    bucket_keys = sorted(buckets, key=lambda key: (-len(buckets[key]), key[0], key[1]))
    while len(selected) < args.limit and any(buckets[key] for key in bucket_keys):
        for key in bucket_keys:
            if len(selected) >= args.limit:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())

    selected.sort(key=lambda item: int(item.get("prompt_id", 0)))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    languages = sorted({str(row.get("language", "text")) for row in selected})
    cwes = sorted({str(row.get("cwe_identifier", "CWE-UNKNOWN")) for row in selected})
    print(
        f"wrote {len(selected)} rows to {out_path}; "
        f"languages={len(languages)} cwes={len(cwes)} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
