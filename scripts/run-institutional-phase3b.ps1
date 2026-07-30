$ErrorActionPreference = "Stop"

python -m research.institutional_model.cli self-check
python -m research.institutional_model.cli phase3-audit @args
