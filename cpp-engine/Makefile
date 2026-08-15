CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wpedantic
BUILD_DIR ?= ../scratch/cpp-engine-build
INCLUDES := -Iinclude

BINS := $(BUILD_DIR)/validate $(BUILD_DIR)/bench $(BUILD_DIR)/price_probe

.PHONY: all price-parity full-parity fastpath-parity clean clean-help

all: $(BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/validate: src/validate.cpp include/sim.hpp include/pyrandom.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/bench: src/bench.cpp include/sim.hpp include/pyrandom.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BUILD_DIR)/price_probe: src/price_probe.cpp include/sim.hpp include/pyrandom.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

price-parity: all
	uv run --with kaggle-environments==1.32.7 python tools/check_price_parity.py $(BUILD_DIR)/price_probe

full-parity: all
	uv run --with kaggle-environments==1.32.7 python tools/run_full_parity.py $(BUILD_DIR)/validate $(BUILD_DIR)/traces

fastpath-parity:
	python3 tools/check_fastpaths.py

clean:
	rm -rf $(BUILD_DIR)

clean-help:
	@echo "Generated binaries/traces live in $(BUILD_DIR); remove that scratch directory when desired."
