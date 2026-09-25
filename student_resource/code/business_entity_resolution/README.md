# Business Entity Resolution

Two-stage pipeline: TF-IDF blocking (country-sharded) → LightGBM matcher → F0.5 threshold.
Country-open: no hard-coded labels; France flows through the same shard path.
No external data/APIs. Allowed licenses only (LightGBM MIT, scikit-learn BSD, RapidFuzz MIT).

## Layout

```
code/business_entity_resolution/
├── src/  text_norm.py, transliterate.py, common.py, blocking.py,
│         features.py, evaluate.py, train_matcher.py, infer.py
├── requirements.txt
└── README.md  (this file)
```

## Reproduce end-to-end (data → blocking → matching → output)

```bash
cd student_resource
pip install -r code/business_entity_resolution/requirements.txt

# 1. Train matcher + tune threshold (full pool; ~300k S1s is plenty for LGBM)
python -m code.business_entity_resolution.src.train_matcher \
  --train-dir dataset/train --out models \
  --top-k 30 --val-size 50000 --max-train-s1 300000 --seed 42
# writes models/lgbm.txt, models/threshold.txt  (prints val macro-F0.5)

# 2. Inference on test
THR=$(cat models/threshold.txt)
python -m code.business_entity_resolution.src.infer \
  --test-dir dataset/test --model models/lgbm.txt \
  --threshold $THR --out output --top-k 30 --chunk-s1 100000

# 3. Validate (must print PASS)
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

Local smoke test (fast, sampled pool, truth preserved):

```bash
python -m code.business_entity_resolution.src.train_matcher \
  --train-dir dataset/train --out /tmp/models --top-k 20 \
  --val-size 500 --max-train-s1 1500 --max-s23 20000
```

## Method notes

* Normalisation (`text_norm.py`): Devanagari→Latin, NFKD accent strip,
  `&`→`and`, legal-suffix removal for core tokens, street-abbrev expansion,
  last 4+-digit run as ZIP/PIN.
* Blocking (`blocking.py`): per-country TF-IDF word(1,2)-gram index over S2+S3,
  S1 chunks query top-K (default 30). Smoke recall@30 ≈ 0.988.
* Features (`features.py`, 14): tfidf score, name/address Jaccard + RapidFuzz
  token_set/ratio, zip/house/country matches, length diffs.
* Threshold (`evaluate.py`): sweep maximising macro F0.5 on val S1s; tuned
  threshold ships in `models/threshold.txt`.
```

Smoke results (200-S1 pool, 8k distractors): recall@30 0.988, matcher val
F0.5 0.984 vs TF-IDF-only 0.48.
