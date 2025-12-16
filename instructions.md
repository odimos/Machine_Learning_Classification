# Local setup instructions

## Supported environment
- Python: `>=3.10,<3.13` (Python 3.10–3.11 recommended; avoid Python 3.13 for now)
- NumPy: `>=1.24,<2`
- PyTorch stack (keep within the same release generation):
  - `torch >=2.1,<2.4`
  - `torchtext >=0.16,<0.18`
  - `torchdata >=0.6,<0.8`
- Other dependencies:
  - `pandas >=1.5`
  - `nltk >=3.8`
  - `portalocker >=2.0`

### To use a different version of python from the current one you have, you can:
## Install (Windows)
```bat
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Install (macOS / Linux)
```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Post-install (required for NLTK tokenization)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Run
With the venv activated:
```bash
python set.py
```

## Troubleshooting
- If you see binary import/entry-point errors with torch/torchtext, ensure `torch`, `torchtext`, and `torchdata` are within the compatible ranges above and installed in the same virtual environment.
- If you see NumPy ABI warnings/errors, ensure NumPy is `<2` (this project pins `numpy>=1.24,<2`).
- If you see missing-module errors (`ModuleNotFoundError`), install dependencies inside the activated venv and re-run.
