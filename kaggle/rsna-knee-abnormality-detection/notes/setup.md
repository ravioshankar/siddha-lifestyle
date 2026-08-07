# Setup checklist (Phase 0)

- [x] Python venv at `.venv/` with `requirements.txt`
- [ ] `~/.kaggle/kaggle.json` (accept competition rules, then create API token)
- [x] CSVs loadable (`scripts/verify_setup.py` passes)
  - Currently: **synthetic** sample CSVs from `scripts/make_sample_data.py`
  - Replace with: `bash scripts/download_csvs.sh`

## Commands

```bash
cd kaggle/rsna-knee-abnormality-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_sample_data.py   # temporary
# OR after token:
# bash scripts/download_csvs.sh
python scripts/verify_setup.py
```
