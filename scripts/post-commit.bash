#!/usr/bin/env bash

echo "Running post-commit hook"

rm coverage.xml
codecov --token="84d5db20-8416-49b1-81b2-c5fe2f008210"
