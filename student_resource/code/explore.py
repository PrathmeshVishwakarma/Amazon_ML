"""Memory-conscious EDA over the training data."""
import collections
import csv
import sys

BASE = "dataset/train"


def count_matches():
    counts = collections.Counter()
    s2 = s3 = 0
    total = 0
    with open(f"{BASE}/train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, _, rest = line.partition("\t")
            total += 1
            if not rest.strip():
                counts[0] += 1
                continue
            ids = rest.split(",")
            counts[len(ids)] += 1
            for mid in ids:
                if mid.startswith("S2-"):
                    s2 += 1
                elif mid.startswith("S3-"):
                    s3 += 1
    print("total S1:", total)
    print("singletons:", counts[0], f"{counts[0]/total:.2%}")
    print("has match:", total - counts[0], f"{(total-counts[0])/total:.2%}")
    nz = sum(k * v for k, v in counts.items() if k > 0)
    print("mean matches (non-singleton):", nz / max(1, total - counts[0]))
    print("S2 refs:", s2, "S3 refs:", s3)
    print("match-count histogram (0..12):")
    for k in range(0, 13):
        print(f"  {k}: {counts[k]}")
    print("  max:", max(counts))


def country_mix(path, label):
    counts = collections.Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            n += 1
            if len(parts) >= 4:
                counts[parts[3]] += 1
            else:
                counts["<missing>"] += 1
    print(f"{label}: {n} rows; countries:")
    for c, v in counts.most_common(10):
        print(f"  {c!r}: {v}")


def show_examples(n=6):
    gt = {}
    with open(f"{BASE}/train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if rest.strip():
                gt[s1] = rest
    # collect ids we need to look up
    need = set()
    picked = []
    for s1, mids in gt.items():
        picked.append((s1, mids))
        for m in mids.split(","):
            need.add(m)
        if len(picked) >= n:
            break
    print("picked", len(picked), "groups; need", len(need), "other ids")

    def scan(path, prefix):
        out = {}
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                eid, _, rest = line.rstrip("\n").partition("\t")
                if eid in need:
                    out[eid] = rest
        return out

    s2 = scan(f"{BASE}/train_source2.tsv", "S2-")
    s3 = scan(f"{BASE}/train_source3.tsv", "S3-")
    s1info = {}
    want_s1 = {p[0] for p in picked}
    with open(f"{BASE}/train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            eid, _, rest = line.rstrip("\n").partition("\t")
            if eid in want_s1:
                s1info[eid] = rest
    print("\n=== examples ===")
    for s1, mids in picked:
        print(f"\nS1 {s1}: {s1info.get(s1)}")
        for m in mids.split(","):
            src = s2 if m.startswith("S2-") else s3
            print(f"  {m}: {src.get(m)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "gt"):
        count_matches()
    if which in ("all", "country"):
        country_mix(f"{BASE}/train_source1.tsv", "S1")
        country_mix(f"{BASE}/train_source2.tsv", "S2")
        country_mix(f"{BASE}/train_source3.tsv", "S3")
    if which in ("all", "ex"):
        show_examples()
