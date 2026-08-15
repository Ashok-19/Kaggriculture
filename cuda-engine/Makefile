NVCC ?= /usr/bin/nvcc
CXX ?= g++
ARCH ?= sm_86
BUILD_DIR ?= ../scratch/cuda-engine-build
INCLUDES := -Iinclude
# Exact market parity forbids --use_fast_math. Keep IEEE double operations intact.
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo --compiler-options=-O3,-Wall
PTXAS ?= -Xptxas=-v
BINS := $(BUILD_DIR)/validate $(BUILD_DIR)/bench $(BUILD_DIR)/price_probe

.PHONY: all check validate-parity state-parity snapshot-parity price-parity policy-smoke bench branch-bench tape-bench tape-branch bonsai-search bonsai-eval-one hybrid-grid clean
all: $(BINS)

check: price-parity validate-parity state-parity snapshot-parity policy-smoke

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/validate: src/validate.cu include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/bench: src/bench.cu include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/price_probe: src/price_probe.cu include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/snapshot_validate: src/snapshot_validate.cu include/sim.cuh include/pyrandom.cuh include/runtime.cuh include/logical_hash.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/branch_bench: src/branch_bench.cu include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/tape_bench: src/tape_bench.cu include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/tape_branch: src/tape_branch.cu include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/policy_smoke: src/policy_smoke.cu include/policy_runner.cuh include/observation.cuh include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/bonsai_search: src/bonsai_search.cu include/bonsai_gp.cuh include/observation.cuh include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/hybrid_grid: src/hybrid_grid.cu include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/bonsai_eval_one: src/bonsai_eval_one.cu include/bonsai_gp.cuh include/observation.cuh include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/cuda_state_hash: src/state_parity.cu include/sim.cuh include/pyrandom.cuh include/runtime.cuh include/logical_hash.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/cpu_state_hash: src/cpu_state_hash.cpp include/logical_hash.hpp ../cpp-engine/include/sim.hpp ../cpp-engine/include/pyrandom.hpp | $(BUILD_DIR)
	$(CXX) -O3 -std=c++17 -Iinclude -I../cpp-engine/include -o $@ $<

validate-parity: all
	$(BUILD_DIR)/validate ../scratch/cpp-engine-build/traces/*.txt

state-parity: $(BUILD_DIR)/cuda_state_hash $(BUILD_DIR)/cpu_state_hash
	sh tools/run_state_parity.sh $(BUILD_DIR) ../scratch/cpp-engine-build/traces/*.txt

snapshot-parity: $(BUILD_DIR)/snapshot_validate
	$(BUILD_DIR)/snapshot_validate ../scratch/cpp-engine-build/traces/*.txt

policy-smoke: $(BUILD_DIR)/policy_smoke
	$(BUILD_DIR)/policy_smoke

price-parity: all
	uv run --with kaggle-environments==1.32.7 python ../cpp-engine/tools/check_price_parity.py $(BUILD_DIR)/price_probe

bench: all
	$(BUILD_DIR)/bench ../scratch/cpp-engine-build/traces/replay_random_vs_random_11.txt 4096 128

branch-bench: $(BUILD_DIR)/branch_bench
	$(BUILD_DIR)/branch_bench ../scratch/cpp-engine-build/traces/replay_starter_vs_starter_11.txt 168 28 128 128

tape-bench: $(BUILD_DIR)/tape_bench
	$(BUILD_DIR)/tape_bench ../scratch/live_top10/current_best_aug14/tapes/kawashigi_s0_sub55425101_ep92923168.json ../scratch/live_top10/v32_live_losses/tapes/hax2456_93023456.json 62000 4096 64

tape-branch: $(BUILD_DIR)/tape_branch
	$(BUILD_DIR)/tape_branch ../scratch/live_top10/current_best_aug14/tapes/kawashigi_s0_sub55425101_ep92923168.json ../scratch/live_top10/v32_live_losses/tapes/hax2456_93023456.json 0 168 62000 1 ../scratch/live_top10/current_best_aug14/tapes/kawashigi_s0_sub55425101_ep92923168.json

bonsai-search: $(BUILD_DIR)/bonsai_search
	@echo "usage: $(BUILD_DIR)/bonsai_search --routes ROUTE_LIST [search options]"

hybrid-grid: $(BUILD_DIR)/hybrid_grid
	@echo "usage: $(BUILD_DIR)/hybrid_grid --routes ROUTE_LIST [grid options]"

bonsai-eval-one: $(BUILD_DIR)/bonsai_eval_one
	@echo "usage: $(BUILD_DIR)/bonsai_eval_one --routes LIST --program FILE.bin --opponent I --seat S --seed N"

clean:
	rm -rf $(BUILD_DIR)
