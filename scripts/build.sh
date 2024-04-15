cmake -S . -Bbuild -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -C build -j 8
