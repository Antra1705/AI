# Host (Mac/Linux) build: unit tests, simulator, replay and training tools.
# No hardware and no extra libraries needed.
#   make test     run all tests
#   make all      build tests + sim + replay + train
#   make clean

CXX      ?= clang++
SRC_DIR  := ../adaptive_suspension/src
BUILD    := build

# Core logic must stay ESP32-friendly: C++11, no accidental double precision.
CORE_FLAGS := -std=c++11 -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -O2 -g -I$(SRC_DIR)
HOST_FLAGS := -std=c++11 -Wall -Wextra -Werror -O2 -g -I$(SRC_DIR) -Ifakes -Isim

CORE_SRC     := $(wildcard $(SRC_DIR)/core/*.cpp)
CORE_OBJ     := $(patsubst $(SRC_DIR)/core/%.cpp,$(BUILD)/core/%.o,$(CORE_SRC))
SIM_LIB_SRC  := $(filter-out sim/sim_main.cpp,$(wildcard sim/*.cpp))
SIM_LIB_OBJ  := $(patsubst sim/%.cpp,$(BUILD)/sim/%.o,$(SIM_LIB_SRC))
SIM_MAIN_OBJ := $(BUILD)/sim/sim_main.o
TEST_OBJ     := $(patsubst tests/%.cpp,$(BUILD)/tests/%.o,$(wildcard tests/*.cpp))
REPLAY_OBJ   := $(patsubst replay/%.cpp,$(BUILD)/replay/%.o,$(wildcard replay/*.cpp))
TRAIN_OBJ    := $(patsubst train/%.cpp,$(BUILD)/train/%.o,$(wildcard train/*.cpp))
HEADERS      := $(wildcard $(SRC_DIR)/*.h $(SRC_DIR)/core/*.h $(SRC_DIR)/esp32/dashboard_html.h fakes/*.h tests/*.h sim/*.h train/*.h)

BIN := $(BUILD)/bin
.PHONY: all test sim replay train clean
all: $(BIN)/run_tests $(BIN)/sim $(BIN)/replay $(if $(TRAIN_OBJ),$(BIN)/train)

test: $(BIN)/run_tests
	./$(BIN)/run_tests

sim: $(BIN)/sim
replay: $(BIN)/replay
train: $(BIN)/train

$(BUILD)/core/%.o: $(SRC_DIR)/core/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "  CXX $<"
	@$(CXX) $(CORE_FLAGS) -c $< -o $@

$(BUILD)/%.o: %.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "  CXX $<"
	@$(CXX) $(HOST_FLAGS) -c $< -o $@

$(BIN)/run_tests: $(CORE_OBJ) $(SIM_LIB_OBJ) $(TEST_OBJ)
	@mkdir -p $(dir $@)
	@echo "  LD  $@"
	@$(CXX) $^ -o $@

$(BIN)/sim: $(CORE_OBJ) $(SIM_LIB_OBJ) $(SIM_MAIN_OBJ)
	@mkdir -p $(dir $@)
	@echo "  LD  $@"
	@$(CXX) $^ -o $@ -lpthread

$(BIN)/replay: $(CORE_OBJ) $(REPLAY_OBJ)
	@mkdir -p $(dir $@)
	@echo "  LD  $@"
	@$(CXX) $^ -o $@

$(BIN)/train: $(CORE_OBJ) $(TRAIN_OBJ)
	@mkdir -p $(dir $@)
	@echo "  LD  $@"
	@$(CXX) $^ -o $@

clean:
	rm -rf $(BUILD)
