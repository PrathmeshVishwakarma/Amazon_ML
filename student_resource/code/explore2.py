"""Check mapping structure: do S2/S3 IDs map to exactly one S1? Analyse name
rarity and address-match rates to inform the blocking strategy."""
import collections

BASE = "dataset/train"

# 1) Does any S2/S3 id appear in more than one S1's match list?
s2_owner = collections.defaultdict(int)
s3_owner = collections.defaultdict(int)
n_total = 0
with open(f"{BASE}/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, _, rest = line.rstrip("\n").partition("\t")
        n_total += 1
        if not rest.strip():
            continue
        for m in rest.split(","):
            if m.startswith("S2-"):
                s2_owner[m] += 1
            else:
                s3_owner[m] += 1

s2_multi = sum(1 for v in s2_owner.values() if v > 1)
s3_multi = sum(1 for v in s3_owner.values() if v > 1)
print(f"S2 ids referenced: {len(s2_owner)}; referenced by >1 S1: {s2_multi}")
print(f"S3 ids referenced: {len(s3_owner)}; referenced by >1 S1: {s3_multi}")
print(f"max s2 refs: {max(s2_owner.values())}, max s3 refs: {max(s3_owner.values())}")

# how many S1 have both S2 and S3 matches, only S2, only S3
both = only2 = only3 = 0
with open(f"{BASE}/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, _, rest = line.rstrip("\n").partition("\t")
        if not rest.strip():
            continue
        has2 = "S2-" in rest
        has3 = "S3-" in rest
        if has2 and has3:
            both += 1
        elif has2:
            only2 += 1
        elif has3:
            only3 += 1
print(f"S1 with both: {both}, only S2: {only2}, only S3: {only3}")
