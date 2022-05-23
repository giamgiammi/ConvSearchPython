#!/bin/sh

set -e

DIR="$(dirname "$0")"
DIR="$(realpath "$DIR")"

cd "$DIR"
cd ..

PYTHONPATH="$(realpath ./)"
export PYTHONPATH

export NO_PRELOAD=1

pdoc3 --html --force --template-dir "$DIR/templates" --output-dir "$DIR/html" convSearchPython

cp -r "$DIR/imgs" "$DIR/html/convSearchPython/"
