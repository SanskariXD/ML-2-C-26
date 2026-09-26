"""Deterministic synthetic data for software tests. Not competition training data."""
from __future__ import annotations
import csv
from pathlib import Path


def write_tsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def make_fixture(directory, n=360):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    header = ["entity_id", "business_name", "business_address", "country"]
    q, s2, s3, gt = [], [], [], []
    brands = ["Aster", "Mango", "Northstar", "Silver Finch", "Lotus", "Orbit"]
    for i in range(n):
        country = "US" if i % 2 else "India"
        name = f"{brands[i % len(brands)]} Workshop {i} Ltd"
        address = f"{100 + i} Cedar Road Unit {i % 15 + 1} {560000 + i}"
        qid = f"S1-{i:06d}"
        q.append([qid, name, address, country])
        targets = []
        if i % 7:
            targets = [f"S2-p{i:06d}", f"S3-p{i:06d}"]
            s2.append([targets[0], name.replace("Ltd", "Limited"), address.replace("Road", "Rd"), country])
            s3.append([targets[1], " ".join(reversed(name.split())), address if i % 5 else "", country])
        s2.append([f"S2-n{i:06d}", name, f"{9000 + i} Ash Avenue {770000 + i}", country])
        s3.append([f"S3-n{i:06d}", f"Unrelated Bakery {i}", address, country])
        gt.append([qid, ",".join(targets)])
    for prefix in ("train", "test"):
        # Independent prefixes are convenience fixtures, not claims of test/GT isolation.
        write_tsv(directory / f"{prefix}_source1.tsv", header, q)
        write_tsv(directory / f"{prefix}_source2.tsv", header, s2)
        write_tsv(directory / f"{prefix}_source3.tsv", header, s3)
    write_tsv(directory / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    return directory

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--queries", type=int, default=360)
    args = p.parse_args()
    make_fixture(args.directory, args.queries)
