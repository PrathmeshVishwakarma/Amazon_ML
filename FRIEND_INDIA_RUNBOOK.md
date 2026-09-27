# Friend Runbook — India-only inference (8 vCPU / 64 GB box)

You are scoring **India only** (~809k businesses). The US + France shards run
on the main box; your two part files get merged there. Total: ~40 min setup
+ ~3.5–4 hrs compute.

## 0. Box

- `r6i.2xlarge` (8 vCPU, **64 GB RAM** — RAM matters more than CPUs here),
  100 GB disk, Ubuntu 22.04, region `us-east-1`.
- Keep everything under `~/`. `/tmp` is RAM-backed (~4 GB) — extracting
  anything there fakes a "disk full" error.

## 1. System + code

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libgomp1 unzip git
cd ~ && git clone https://github.com/PrathmeshVishwakarma/Amazon_ML.git
# confirm split support is present (expect 5+):
grep -c countries ~/Amazon_ML/student_resource/code/business_entity_resolution/src/infer.py
```

If that prints less than 5, run `cd ~/Amazon_ML && git pull` and check again —
do not proceed on stale code.

## 2. Dataset — TEST files only (~1.1 GB)

Train files are **not** needed for inference. Download the zip and extract
only the test TSVs:

```bash
cd ~/Amazon_ML
wget -q --show-progress https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip -O ~/sr.zip
mkdir -p ~/sr && unzip -q ~/sr.zip -d ~/sr
TDIR=$(dirname $(find ~/sr -name "test_source1.tsv" | head -1)); echo "test files in: $TDIR"
mkdir -p student_resource/dataset/test
cp $TDIR/test_source1.tsv $TDIR/test_source2.tsv $TDIR/test_source3.tsv student_resource/dataset/test/
ls -lh student_resource/dataset/test/
rm -rf ~/sr ~/sr.zip
```

Expected sizes: `test_source1.tsv` ~167M, `test_source2.tsv` ~486M,
`test_source3.tsv` ~483M. If any file is 0 bytes or much smaller, the
download was interrupted — delete it and re-run the `wget`/`cp` for that file.
Verify with `head -2 student_resource/dataset/test/test_source1.tsv`
(header row + one data row).

## 3. Python env

```bash
cd ~/Amazon_ML/student_resource
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt
```

## 4. Model weights (sent by the main box owner)

You need exactly two files in `student_resource/models_final/`:

- `lgbm.txt` (a few MB — the trained matcher)
- `threshold.txt` (one number, `0.75`)

Receive them over any channel (chat/Drive/scp) and place them:

```bash
mkdir -p ~/Amazon_ML/student_resource/models_final
# copy the two received files here, then:
ls -lh ~/Amazon_ML/student_resource/models_final/
cat ~/Amazon_ML/student_resource/models_final/threshold.txt
```

## 5. The run — India only

Flags must match **exactly** (same model, threshold, top-k, vocab) or the
merge stays coherent but the quality mix breaks. Do not improvise:

```bash
cd ~/Amazon_ML/student_resource && source .venv/bin/activate
THR=$(cat models_final/threshold.txt); echo "thr=$THR"
nohup python -u -m code.business_entity_resolution.src.infer \
  --test-dir dataset/test --model models_final/lgbm.txt \
  --threshold "$THR" --out output_india --top-k 50 \
  --jobs 6 --block-chunk 250 --countries India > infer_india.log 2>&1 &
disown
tail -f infer_india.log
```

- `--jobs 6`: 6 workers on 8 vCPUs (headroom for the OS). Never exceed vCPUs.
- `--block-chunk 250`: caps each sparse multiply — this is what keeps the
  64 GB box clear of the OOM-killer. `Ctrl+C` out of `tail` anytime (kills
  only the viewer). Closing the laptop/tab is safe (`nohup`).
- Heartbeat lines look like `[block:India/s3] chunk 12/40: ... (13 q/s ...)`
  and `[infer:India] scored (...)`. If no new lines for 45+ min while CPUs
  are idle, report it — do not restart blindly.

Expected: ~3.5–4 hrs. `free -g` should keep 10+ GB `available` throughout;
if it approaches zero, stop and report (do not lower flags on your own).

## 6. Hand back — TWO files only

When the log prints `done.`, send back (any channel):

- `output_india/.part_India_cand.tsv`
- `output_india/.part_India_match.tsv`

```bash
ls -lh ~/Amazon_ML/student_resource/output_india/.part_India_*.tsv
tail -2 ~/Amazon_ML/student_resource/infer_india.log
```

That is everything. The main box merges (`merge_parts.py`), validates, and
submits. Terminate your instance only when told — keep it until the merged
submission validates, in case a rerun is needed.

## Troubleshooting

| Symptom | Action |
|---|---|
| `EmptyDataError` on load | A TSV is 0 bytes — redo §2 for that file |
| `write error (disk full?)` on unzip | You extracted to `/tmp` — redo into `~/sr` |
| `ArrayMemoryError` in log | Stop, report the `peak XGB` value from the last chunk line |
| Log frozen 45+ min, CPUs idle | Do not restart — capture `tail -20 infer_india.log` + `ps aux \| grep infer` and report |
| `part-file config differs` | Your flags differ from §5 — relaunch with the exact command above |
