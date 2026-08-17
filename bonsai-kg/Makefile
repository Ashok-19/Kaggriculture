ROOT := ..
BUILD := $(ROOT)/scratch/bonsai_kg
CXX ?= g++
NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic
NVFLAGS := -O3 -std=c++17 -arch=sm_86 --compiler-options=-O3,-Wall

.PHONY: all build test python-test binary-validation differential game-parity feature-parity bench clean-build

all: build

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/cpp_differential_runner: tests/cpp_differential_runner.cpp include/bonsai_kg.hpp include/bonsai_kg_abi.hpp include/bonsai_kg_io.hpp tests/synthetic_io.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/cuda_differential_runner: tests/cuda_differential_runner.cu include/bonsai_kg.hpp include/bonsai_kg_abi.hpp include/bonsai_kg_io.hpp tests/synthetic_io.hpp | $(BUILD)
	$(NVCC) -std=c++17 -O2 $< -o $@

$(BUILD)/cpp_game_runner: tests/cpp_game_runner.cpp include/bonsai_kg.hpp include/bonsai_kg_abi.hpp include/bonsai_kg_io.hpp cpp/kag_adapter.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/cuda_game_runner: tests/cuda_game_runner.cu include/bonsai_kg.hpp include/bonsai_kg_abi.hpp include/bonsai_kg_io.hpp cuda/kag_adapter.cuh | $(BUILD)
	$(NVCC) $(NVFLAGS) -lineinfo $< -o $@

$(BUILD)/cpp_feature_parity: tests/cpp_feature_parity.cpp tests/feature_hash.hpp cpp/kag_adapter.hpp include/bonsai_kg_abi.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/cuda_feature_parity: tests/cuda_feature_parity.cu tests/feature_hash.hpp cuda/kag_adapter.cuh include/bonsai_kg_abi.hpp | $(BUILD)
	$(NVCC) $(NVFLAGS) $< -o $@

$(BUILD)/validate_compiled: tests/validate_compiled.cpp include/bonsai_kg_abi.hpp include/bonsai_kg_io.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/size_report: tests/size_report.cpp include/bonsai_kg.hpp include/bonsai_kg_abi.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/benchmark: tests/benchmark.cu include/bonsai_kg.hpp include/bonsai_kg_abi.hpp | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=sm_86 $< -o $@

$(BUILD)/cuda_policy_smoke: tests/cuda_policy_smoke.cu cuda/policy.cuh cuda/kag_adapter.cuh include/bonsai_kg.hpp include/bonsai_kg_abi.hpp | $(BUILD)
	$(NVCC) $(NVFLAGS) -lineinfo $< -o $@

$(BUILD)/reactive_move.bkgc: examples/reactive_move.json python/bonsai_kg.py | $(BUILD)
	python3 python/bonsai_kg.py $< --out $@

build: $(BUILD)/cpp_differential_runner $(BUILD)/cuda_differential_runner $(BUILD)/cpp_game_runner $(BUILD)/cuda_game_runner $(BUILD)/cpp_feature_parity $(BUILD)/cuda_feature_parity $(BUILD)/validate_compiled $(BUILD)/size_report $(BUILD)/benchmark $(BUILD)/cuda_policy_smoke $(BUILD)/reactive_move.bkgc

python-test:
	python3 tests/test_semantics.py

binary-validation: $(BUILD)/validate_compiled
	python3 tests/test_binary_validation.py

differential: $(BUILD)/cpp_differential_runner $(BUILD)/cuda_differential_runner
	python3 tests/differential.py --cases 256 --ticks 16

game-parity: $(BUILD)/cpp_game_runner $(BUILD)/cuda_game_runner $(BUILD)/reactive_move.bkgc
	python3 tests/game_parity.py --steps 80 --seeds 12345 54321 62000 62001

feature-parity: $(BUILD)/cpp_feature_parity $(BUILD)/cuda_feature_parity
	python3 tests/test_feature_parity.py --steps 80 --seeds 12345 54321

test: python-test binary-validation differential game-parity feature-parity

bench: $(BUILD)/size_report $(BUILD)/benchmark $(BUILD)/cuda_policy_smoke
	$(BUILD)/size_report
	$(BUILD)/benchmark 5000000 65536 64
	$(BUILD)/cuda_policy_smoke 4096 96

clean-build:
	rm -rf $(BUILD)
