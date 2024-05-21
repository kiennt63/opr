#!/bin/sh

cmake -S . -Bbuild -GNinja
ninja -C build -j 8

ln -sf build/compile_commands.json .
