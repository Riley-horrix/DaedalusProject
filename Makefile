BUILD_DIR = ./build

NPROCS:=16

all: build/daedalus build/test

doc:
	@mkdir -p $(BUILD_DIR)/doc
	doxygen doc/Doxyfile
	@echo "Documentation available in build/doc/html/index.html"

SRC_FILES := $(shell find ./src -name \*.cpp)
INCLUDE_FILES  := $(shell find ./include -name \*.h)
TEST_FILES := $(shell find ./test -name \*.cpp)

FORMAT_FILES := $(SRC_FILES) $(TEST_FILES) $(INCLUDE_FILES)

format:
	@clang-format -Werror --dry-run --verbose $(FORMAT_FILES)

format-hard:
	@clang-format -Werror -i --verbose $(FORMAT_FILES)

test: build/test
	$(eval TESTCASE := $(word 2, $(MAKECMDGOALS)))
	@if [ -z "$(TESTCASE)" ]; then \
		echo "Running all tests..."; \
		$(BUILD_DIR)/test; \
	else \
		echo "Running test case: $(TESTCASE)"; \
		$(BUILD_DIR)/test $(TESTCASE); \
	fi

test-debug: build/test-debug
	$(eval TESTCASE := $(word 2, $(MAKECMDGOALS)))
	@if [ -z "$(TESTCASE)" ]; then \
		echo "Running all tests..."; \
		$(BUILD_DIR)/test; \
	else \
		echo "Running test case: $(TESTCASE)"; \
		$(BUILD_DIR)/test $(TESTCASE); \
	fi

daedalus: build/daedalus
	$(BUILD_DIR)/daedalus

daedalus-debug: build/daedalus-debug
	$(BUILD_DIR)/daedalus

build/daedalus: build/common
	cd $(BUILD_DIR) && cmake --build . --target daedalus -j $(NPROCS)

build/daedalus-debug: build/common-debug
	cd $(BUILD_DIR) && cmake --build . --target daedalus -j $(NPROCS)

build/test:	build/common
	cd $(BUILD_DIR) && cmake --build . --target test -j $(NPROCS)

build/test-debug: build/common-debug
	cd $(BUILD_DIR) && cmake --build . --target test -j $(NPROCS)

build/common:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release ..
	cd $(BUILD_DIR) && cmake --build . --target daedalus_core -j $(NPROCS)

build/common-debug:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug ..
	cd $(BUILD_DIR) && cmake --build . --target daedalus_core -j $(NPROCS)

lines-code:
	@echo "Lines of code in the project:"
	@find src include test \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print0 | xargs -0  wc -l

clean:
	@rm -rf $(BUILD_DIR)/*

.PHONY: all test doc format clean daedalus build/daedalus daedalus-debug build/daedalus-debug build/common build/common-debug build/test build/test-debug