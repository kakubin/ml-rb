all: download path bundle

download:
	@wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.11.0%2Bcpu.zip
	@unzip libtorch-shared-with-deps-1.11.0+cpu.zip

path:
	@bundle config --local build.torch-rb --with-torch-dir=./libtorch

bundle:
	bundle install

.PHONY: all download path bundle
