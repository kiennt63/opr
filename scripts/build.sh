#!/bin/sh

cmake -S . -Bbuild -GNinja
ninja -C build -j 8
