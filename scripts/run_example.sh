#!/bin/sh

./scripts/build.sh || { echo '*********************** Build failed! ***********************' ; exit 1; }

./build/release/bin/example
