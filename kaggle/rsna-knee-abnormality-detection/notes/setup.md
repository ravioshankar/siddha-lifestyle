# Setup checklist (Phase 0)

- [x] `uv` environment at `.venv/` with `requirements.txt`
- [ ] `~/.kaggle/access_token` (accept competition rules, then create API token)
- [x] CSVs loadable (`scripts/verify_setup.py` passes)
  - Currently: **synthetic** sample CSVs from `scripts/make_sample_data.py`
  - Replace with: `uv run bash scripts/download_csvs.sh`

## Commands

```bash
cd kaggle/rsna-knee-abnormality-detection
uv venv
uv pip install -r requirements.txt
uv run python scripts/make_sample_data.py   # temporary
# OR after token:
# uv run bash scripts/download_csvs.sh
uv run python scripts/verify_setup.py
```
