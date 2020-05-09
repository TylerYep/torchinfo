#!/usr/bin/env bash
# https://rock-it.pl/automatic-code-quality-checks-with-git-hooks/
# this command creates symlink to our pre-commit script

GIT_DIR=$(git rev-parse --git-dir)

echo "Installing hooks..."
ln -s ../../scripts/pre-commit.bash $GIT_DIR/hooks/pre-commit
ln -s ../../scripts/post-commit.bash $GIT_DIR/hooks/post-commit
echo "Done!"