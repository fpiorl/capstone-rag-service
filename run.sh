#!/bin/bash

set -e

[ -e .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m app.main
