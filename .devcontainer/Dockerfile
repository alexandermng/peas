FROM mcr.microsoft.com/devcontainers/rust:1-1-bullseye

RUN apt-get update && apt-get install -y \
	build-essential git \
	clang libc++-dev libc++abi-dev lld \
	cmake ninja-build

RUN cargo install wasm-tools

# Binaryen
RUN git clone --recursive https://github.com/WebAssembly/binaryen.git /opt/binaryen && cd /opt/binaryen && \
	cmake . -DBUILD_TESTS=OFF && make && \
	make install

# Wasp
RUN git clone --recursive https://github.com/WebAssembly/wasp.git /opt/wasp && cd /opt/wasp && \
	mkdir build && cd build && cmake .. -DBUILD_TESTS=OFF && cmake --build . && \
	make install

# Wabt
RUN git clone --recursive https://github.com/WebAssembly/wabt.git /opt/wabt && cd /opt/wabt && \
	mkdir build && cd build && cmake .. -DBUILD_TESTS=OFF && cmake --build . && \
	make install

USER vscode