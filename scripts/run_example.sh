#!/bin/sh

./scripts/build.sh || { echo '*********************** Build failed! ***********************' ; exit 1; }

mkdir -p output

./build/debug/bin/cuda
