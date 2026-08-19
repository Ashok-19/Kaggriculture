ROOT := ..
BUILD := $(ROOT)/scratch/bonsai_kg
CXX ?= g++
NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic
NVFLAGS := -O3 -std=c++17 -arch=sm_86 --compiler-options=-O3,-Wall
V6BUILD := $(ROOT)/scratch/bonsai_kg_v6_translation
V6INT := integrations/stateful_v6
V6ASSETS := $(V6INT)/assets

.PHONY: all build test python-test schema-test binary-validation differential game-parity feature-parity bench clean-build v6-assets v6-build v6-test v6-canary v6-official-canary

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

$(BUILD)/cpp_observation_boundary: tests/cpp_observation_boundary.cpp cpp/kag_adapter.hpp include/bonsai_kg_abi.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/cuda_observation_boundary: tests/cuda_observation_boundary.cu cuda/kag_adapter.cuh include/bonsai_kg_abi.hpp | $(BUILD)
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

build: $(BUILD)/cpp_differential_runner $(BUILD)/cuda_differential_runner $(BUILD)/cpp_game_runner $(BUILD)/cuda_game_runner $(BUILD)/cpp_feature_parity $(BUILD)/cuda_feature_parity $(BUILD)/cpp_observation_boundary $(BUILD)/cuda_observation_boundary $(BUILD)/validate_compiled $(BUILD)/size_report $(BUILD)/benchmark $(BUILD)/cuda_policy_smoke $(BUILD)/reactive_move.bkgc

python-test: $(BUILD)/validate_compiled
	python3 -m unittest -v tests/test_semantics.py tests/test_v2_semantics.py tests/test_v2_limits.py tests/test_v3_finalization.py

schema-test:
	python3 -m json.tool schema/policy.schema.json >/dev/null
	python3 -m json.tool schema/search_grammar_v1.json >/dev/null
	python3 -m json.tool schema/search_grammar_v2.json >/dev/null
	python3 -m json.tool schema/search_grammar_v3.json >/dev/null
	python3 -c 'import json; p=json.load(open("schema/policy.schema.json")); g=json.load(open("schema/search_grammar_v3.json")); assert p["properties"]["schema_version"]["const"]==3 and g["compiled_abi_version"]==3 and g["required_board_size"]==10'

binary-validation: $(BUILD)/validate_compiled
	python3 tests/test_binary_validation.py

differential: $(BUILD)/cpp_differential_runner $(BUILD)/cuda_differential_runner
	python3 tests/differential.py --cases 512 --ticks 16

game-parity: $(BUILD)/cpp_game_runner $(BUILD)/cuda_game_runner $(BUILD)/reactive_move.bkgc
	python3 tests/game_parity.py --steps 80 --seeds 12345 54321 62000 62001

feature-parity: $(BUILD)/cpp_feature_parity $(BUILD)/cuda_feature_parity
	python3 tests/test_feature_parity.py --steps 80 --seeds 12345 54321

test: schema-test python-test binary-validation differential game-parity feature-parity

bench: $(BUILD)/size_report $(BUILD)/benchmark $(BUILD)/cuda_policy_smoke
	$(BUILD)/size_report
	$(BUILD)/benchmark 5000000 65536 64
	$(BUILD)/cuda_policy_smoke 4096 96

v6-assets:
	python3 $(V6INT)/build_assets.py

$(V6BUILD):
	mkdir -p $(V6BUILD)

$(V6BUILD)/stateful_v6_pack_cpp: tests/stateful_v6_pack_cpp.cpp $(V6INT)/cpp_runtime.hpp $(V6INT)/route_pack.hpp $(V6ASSETS)/identity.hpp | $(V6BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(V6BUILD)/stateful_v6_pack_cuda: tests/stateful_v6_pack_cuda.cu $(V6INT)/cuda_runtime.cuh $(V6INT)/route_pack.hpp $(V6ASSETS)/identity.hpp | $(V6BUILD)
	$(NVCC) $(NVFLAGS) $< -o $@

$(V6BUILD)/stateful_v6_game_cpp: tests/stateful_v6_game_cpp.cpp $(V6INT)/cpp_runtime.hpp $(V6INT)/route_pack.hpp $(V6ASSETS)/identity.hpp | $(V6BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(V6BUILD)/stateful_v6_game_cuda: tests/stateful_v6_game_cuda.cu $(V6INT)/cuda_runtime.cuh $(V6INT)/route_pack.hpp $(V6ASSETS)/identity.hpp | $(V6BUILD)
	$(NVCC) $(NVFLAGS) $< -o $@

$(V6BUILD)/stateful_v6_observation_cpp: tests/stateful_v6_observation_cpp.cpp $(V6INT)/cpp_runtime.hpp $(V6INT)/route_pack.hpp $(V6ASSETS)/identity.hpp | $(V6BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(V6BUILD)/stateful_v6_engine_exploit_cpp: tests/stateful_v6_engine_exploit_cpp.cpp $(V6INT)/engine_exploits.hpp | $(V6BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(V6BUILD)/stateful_v6_engine_exploit_cuda: tests/stateful_v6_engine_exploit_cuda.cu $(V6INT)/engine_exploits.cuh | $(V6BUILD)
	$(NVCC) $(NVFLAGS) $< -o $@

v6-build: v6-assets $(BUILD)/cpp_differential_runner $(BUILD)/cuda_differential_runner $(V6BUILD)/stateful_v6_pack_cpp $(V6BUILD)/stateful_v6_pack_cuda $(V6BUILD)/stateful_v6_game_cpp $(V6BUILD)/stateful_v6_game_cuda $(V6BUILD)/stateful_v6_observation_cpp $(V6BUILD)/stateful_v6_engine_exploit_cpp $(V6BUILD)/stateful_v6_engine_exploit_cuda

v6-test: v6-build
	python3 -m unittest -v tests/test_stateful_v6_translation.py tests/test_stateful_v6_engine_exploits.py
	python3 tests/stateful_v6_controller_parity.py
	python3 tests/stateful_v6_pack_parity.py
	python3 tests/stateful_v6_engine_exploit_native_parity.py

v6-canary: v6-build
	python3 tests/stateful_v6_translation_parity.py --games-per-seat 32 --steps 720

v6-official-canary: v6-build
	uv run --with kaggle-environments==1.32.7 python tests/stateful_v6_game_parity.py --games-per-seat 32 --steps 720

clean-build:
	rm -rf $(BUILD)
