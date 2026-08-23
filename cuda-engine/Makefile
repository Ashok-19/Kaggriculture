NVCC ?= /usr/bin/nvcc
CXX ?= g++
ARCH ?= sm_86
BUILD_DIR ?= ../scratch/cuda-engine-build
INCLUDES := -Iinclude
# Exact market parity forbids --use_fast_math. Keep IEEE double operations intact.
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo --compiler-options=-O3,-Wall
PTXAS ?= -Xptxas=-v
B4EC_NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
B4EC_HOST_CXX ?= g++-9
B4EC_HOST_FLAG := $(if $(strip $(B4EC_HOST_CXX)),-ccbin $(B4EC_HOST_CXX),)
BINS := $(BUILD_DIR)/validate $(BUILD_DIR)/bench $(BUILD_DIR)/price_probe

.PHONY: all check validate-parity state-parity snapshot-parity price-parity policy-smoke bench branch-bench tape-bench tape-branch bonsai-search bonsai-eval-one bonsai-branch-cases hybrid-grid clean opening-option-beam bkg4-edit-corpus-eval bkg4-invariant-edit-eval
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

$(BUILD_DIR)/bkg4_edit_corpus_eval: src/bkg4_edit_corpus_eval.cu include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh ../bonsai-kg/include/bonsai_kg_v4.hpp ../bonsai-kg/include/bonsai_kg_v4_io.hpp ../bonsai-kg/cuda/kag_adapter_v4.cuh ../bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh | $(BUILD_DIR)
	$(NVCC) $(B4EC_NVCCFLAGS) $(B4EC_HOST_FLAG) -o $@ $<

$(BUILD_DIR)/bkg4_invariant_edit_eval: src/bkg4_invariant_edit_eval.cu include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh ../bonsai-kg/include/bonsai_kg_v4.hpp ../bonsai-kg/include/bonsai_kg_v4_io.hpp ../bonsai-kg/include/bkg4_invariant_module.hpp ../bonsai-kg/cuda/kag_adapter_v4.cuh ../bonsai-kg/integrations/bkg4_champion/cuda_compat.cuh | $(BUILD_DIR)
	$(NVCC) $(B4EC_NVCCFLAGS) $(B4EC_HOST_FLAG) -o $@ $<

$(BUILD_DIR)/opening_crossover: src/opening_crossover.cu include/bonsai_gp.cuh include/observation.cuh include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/opening_geometry_oracle: src/opening_geometry_oracle.cu include/bonsai_gp.cuh include/observation.cuh include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/wheat_timing_search: src/wheat_timing_search.cu include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/turn0_option_search: src/turn0_option_search.cu include/bonsai_gp.cuh include/bonsai_io.hpp include/observation.cuh include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/opening_option_beam: src/opening_option_beam.cu include/bonsai_gp.cuh include/observation.cuh include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(PTXAS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/bonsai_branch_cases: src/bonsai_branch_cases.cu include/bonsai_gp.cuh include/observation.cuh include/tape_io.hpp include/sim.cuh include/pyrandom.cuh include/runtime.cuh | $(BUILD_DIR)
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

bonsai-opening-crossover: $(BUILD_DIR)/opening_crossover
	@echo "usage: $(BUILD_DIR)/opening_crossover --routes LIST [--route-a I --route-b J --branch-step N --out FILE]"

bonsai-opening-geometry: $(BUILD_DIR)/opening_geometry_oracle
	@echo "usage: $(BUILD_DIR)/opening_geometry_oracle --routes LIST [--route-a I --route-b J --branch-step N --out FILE]"

wheat-timing-search: $(BUILD_DIR)/wheat_timing_search
	@echo "usage: $(BUILD_DIR)/wheat_timing_search --routes LIST [--unit-route I --total-wheat N --out CSV]"

turn0-option-search: $(BUILD_DIR)/turn0_option_search
	@echo "usage: $(BUILD_DIR)/turn0_option_search --routes LIST --router FILE.bin --program FILE.bin ... --main-opponents csv --anti-opponents csv"

opening-option-beam: $(BUILD_DIR)/opening_option_beam
	@echo "usage: $(BUILD_DIR)/opening_option_beam --routes LIST --root-ids csv --donor-routes csv --anti-opponents csv --main-opponents csv [options]"

bonsai-branch-cases: $(BUILD_DIR)/bonsai_branch_cases
	@echo "usage: $(BUILD_DIR)/bonsai_branch_cases --routes LIST --prefix FILE.bin --expert FILE.bin --branch-step N --out CSV"

bkg4-edit-corpus-eval: $(BUILD_DIR)/bkg4_edit_corpus_eval
	@echo "usage: $(BUILD_DIR)/bkg4_edit_corpus_eval --controller FILE.bkgc --routes FILE.bkv6 --legacy-routes FILE.bkv6 --cases cases.tsv --genomes states.b4gm --out results.b4ec [--device 0]"

bkg4-invariant-edit-eval: $(BUILD_DIR)/bkg4_invariant_edit_eval
	@echo "usage: $(BUILD_DIR)/bkg4_invariant_edit_eval --controller FILE.bkgc --routes FILE.bkv6 --legacy-routes FILE.bkv6 --cases cases.tsv --candidates states.b4im --out results.b4ic [--device 0]"

clean:
	rm -rf $(BUILD_DIR)
