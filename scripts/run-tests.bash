#!/usr/bin/env bash

# If any command inside script returns error, exit and return that error
set -e

# Ensure that we're always inside the root of our application,
# no matter which directory we run script: Run `./scripts/run-tests.bash`
cd "${0%/*}/.."

# Type checking
mypy .

# Auto-code formatters
isort -y
black . -l 100
git add .

# Style Checking
find . -iname "*.py" | xargs pylint

# Testing
pytest --cov=torchsummary --cov-report=html unit_test
codecov --token="84d5db20-8416-49b1-81b2-c5fe2f008210"
