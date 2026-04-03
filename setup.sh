#!/bin/bash

LIB_PATH="/usr/local/lib/libkeccak.a"
HEADER_PATH="/usr/local/include/libkeccak.a.headers"

set -e

echo "Checking if libkeccak is available..."
if [[ -d "/usr/local/include/libkeccak.a.headers" && -f "/usr/local/lib/libkeccak.a" ]]; then
    echo "✔ libkeccak available system wide"
    exit
else
	echo "⚠ libkeccak not available, importing XKCP submodule"
	git submodule update --init --recursive
	cd XKCP
	echo "🔨 Build XKCP..."
	make x86-64/libXKCP.a EXTRA_CFLAGS="-march=native -mtune=native"
	echo "📦 Installation (requires sudo privileges)"
    	sudo make install
	cd bin
	sudo cp -r bin/x86-64/libXKCP.a.headers /usr/local/include/libkeccak.a.headers
	sudo cp bin/x86-64/libXKCP.a /usr/local/lib/libkeccak.a

fi
echo "Done."
