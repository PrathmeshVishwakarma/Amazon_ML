"""Sample matched pairs to measure script mix (ASCII vs Devanagari), address
availability and similarity distributions - these drive blocking + features."""
import random
import collections

BASE = "dataset/train"
random.seed(42)

N_S1 = 100_000

# ---- sample S1 ids
with open(f"{BASE}/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    all_s1 = [line.split("\t", 1)[0] for line in f if line.strip()]
random.shuffle(all_s1)
sample_s1 = set(all_s1[:N_S1])
print("sampled S1:", len(sample_s1))

need2 = set()
need3 = set()
gt = {}
with open(f"{BASE}/train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        s1, _, rest = line.rstrip("\n").partition("\t")
        if s1 in sample_s1:
            rest = rest.strip()
            gt[s1] = rest
            if rest:
                for m in rest.split(","):
                    (need2 if m.startswith("S2-") else need3).add(m)
print("need S2:", len(need2), "need S3:", len(need3))


def is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def fetch(path, need):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            eid, _, rest = line.rstrip("\n").partition("\t")
            if eid in need:
                out[eid] = rest
    return out


s1rec = fetch(f"{BASE}/train_source1.tsv", sample_s1)
s2rec = fetch(f"{BASE}/train_source2.tsv", need2)
s3rec = fetch(f"{BASE}/train_source3.tsv", need3)
print("fetched s1", len(s1rec), "s2", len(s2rec), "s3", len(s3rec))


def addr_of(rest):
    parts = rest.split("\t")
    name = parts[0] if parts else ""
    addr = parts[1] if len(parts) > 1 else ""
    return name, addr


name_ascii = collections.Counter()
addr_empty = collections.Counter()
n_pairs = 0
for s1, mids in gt.items():
    if not mids:
        continue
    na, aa = addr_of(s1rec.get(s1, ""))
    if not is_ascii(na):
        name_ascii["s1_nonascii"] += 1
    if not aa.strip():
        addr_empty["s1_addr_empty"] += 1
    for m in mids.split(","):
        n_pairs += 1
        rec = s2rec.get(m) or s3rec.get(m) or ""
        nm, am = addr_of(rec)
        if is_ascii(nm):
            name_ascii["match_name_ascii"] += 1
        else:
            name_ascii["match_name_nonascii"] += 1
        if is_ascii(na) and is_ascii(nm):
            name_ascii["both_ascii"] += 1
        if not am.strip():
            addr_empty["match_addr_empty"] += 1

print("pairs:", n_pairs)
print(dict(name_ascii))
print(dict(addr_empty))
print(f"match name ascii rate: {name_ascii['match_name_ascii']/n_pairs:.3f}")
print(f"both ascii name rate: {name_ascii['both_ascii']/n_pairs:.3f}")
print(f"match addr empty rate: {addr_empty['match_addr_empty']/n_pairs:.3f}")
