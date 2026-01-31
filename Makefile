# ============================================================================
# Axiom - High-Performance Cross-Platform Tensor Library
# ============================================================================
#
# Usage:
#   make              - Build release (default)
#   make debug        - Build debug
#   make test         - Run all tests
#   make format       - Format all source files
#   make help         - Show all available targets
#
# ============================================================================

# Configuration
# ----------------------------------------------------------------------------
BUILD_DIR       := build
BUILD_DIR_DEBUG := build-debug
CMAKE           := cmake
CTEST           := ctest
CLANG_FORMAT    := clang-format

# Cross-platform CPU count detection
ifeq ($(OS),Windows_NT)
    NPROC := $(NUMBER_OF_PROCESSORS)
    ifeq ($(NPROC),)
        NPROC := 4
    endif
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        NPROC := $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        NPROC := $(shell nproc 2>/dev/null || echo 4)
    endif
endif

# CMake generator selection
ifeq ($(OS),Windows_NT)
    # Use default generator on Windows (Visual Studio or Ninja if available)
    CMAKE_GENERATOR :=
else
    # Use Ninja on Unix-like systems if available
    CMAKE_GENERATOR := $(shell command -v ninja >/dev/null 2>&1 && echo "-G Ninja" || echo "")
endif

# Source files for formatting
SOURCES := $(shell find include src -name '*.hpp' -o -name '*.cpp' -o -name '*.mm' -o -name '*.h' 2>/dev/null)
TEST_SOURCES := $(shell find tests -name '*.cpp' 2>/dev/null)
EXAMPLE_SOURCES := $(shell find examples -name '*.cpp' 2>/dev/null)

# Colors for terminal output
CYAN    := \033[36m
GREEN   := \033[32m
YELLOW  := \033[33m
RED     := \033[31m
BOLD    := \033[1m
RESET   := \033[0m

# ============================================================================
# Default Target
# ============================================================================

.PHONY: all
all: release

# ============================================================================
# Build Targets
# ============================================================================

.PHONY: release
release: $(BUILD_DIR)/CMakeCache.txt  ## Build release version (default)
	@echo "$(CYAN)Building release...$(RESET)"
	@$(CMAKE) --build $(BUILD_DIR) -j$(NPROC)
	@echo "$(GREEN)✓ Release build complete$(RESET)"

.PHONY: debug
debug: $(BUILD_DIR_DEBUG)/CMakeCache.txt  ## Build debug version
	@echo "$(CYAN)Building debug...$(RESET)"
	@$(CMAKE) --build $(BUILD_DIR_DEBUG) -j$(NPROC)
	@echo "$(GREEN)✓ Debug build complete$(RESET)"

.PHONY: lib
lib: $(BUILD_DIR)/CMakeCache.txt  ## Build only the library (no tests/examples)
	@echo "$(CYAN)Building library...$(RESET)"
	@$(CMAKE) --build $(BUILD_DIR) --target axiom -j$(NPROC)
	@echo "$(GREEN)✓ Library build complete$(RESET)"

# Build configuration marker file (works with both Make and Ninja generators)
$(BUILD_DIR)/CMakeCache.txt:
	@echo "$(CYAN)Configuring release build...$(RESET)"
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Release

$(BUILD_DIR_DEBUG)/CMakeCache.txt:
	@echo "$(CYAN)Configuring debug build...$(RESET)"
	@$(CMAKE) -B $(BUILD_DIR_DEBUG) $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Debug

.PHONY: configure
configure:  ## Reconfigure CMake (release)
	@echo "$(CYAN)Reconfiguring release build...$(RESET)"
	@$(CMAKE) -B $(BUILD_DIR) $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Release
	@echo "$(GREEN)✓ Configuration complete$(RESET)"

.PHONY: configure-debug
configure-debug:  ## Reconfigure CMake (debug)
	@echo "$(CYAN)Reconfiguring debug build...$(RESET)"
	@$(CMAKE) -B $(BUILD_DIR_DEBUG) $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=Debug
	@echo "$(GREEN)✓ Configuration complete$(RESET)"

.PHONY: rebuild
rebuild: clean all  ## Clean and rebuild release

.PHONY: rebuild-debug
rebuild-debug: clean-debug debug  ## Clean and rebuild debug

# ============================================================================
# Test Targets
# ============================================================================

.PHONY: test
test: release  ## Run all tests
	@echo "$(CYAN)Running tests...$(RESET)"
	@cd $(BUILD_DIR) && $(CTEST) --output-on-failure
	@echo "$(GREEN)✓ Tests complete$(RESET)"

.PHONY: test-verbose
test-verbose: release  ## Run all tests with verbose output
	@echo "$(CYAN)Running tests (verbose)...$(RESET)"
	@cd $(BUILD_DIR) && $(CTEST) --output-on-failure --verbose

.PHONY: test-debug
test-debug: debug  ## Run all tests (debug build)
	@echo "$(CYAN)Running tests (debug)...$(RESET)"
	@cd $(BUILD_DIR_DEBUG) && $(CTEST) --output-on-failure

.PHONY: test-failed
test-failed: release  ## Rerun only failed tests
	@echo "$(CYAN)Rerunning failed tests...$(RESET)"
	@cd $(BUILD_DIR) && $(CTEST) --rerun-failed --output-on-failure

# Run a single test: make test-single TEST=tensor_basic
.PHONY: test-single
test-single: release  ## Run single test (TEST=name)
	@if [ -z "$(TEST)" ]; then \
		echo "$(RED)Error: specify test with TEST=name$(RESET)"; \
		echo "Available tests:"; \
		cd $(BUILD_DIR) && $(CTEST) -N | grep "Test #" | sed 's/.*: //'; \
		exit 1; \
	fi
	@echo "$(CYAN)Running test: $(TEST)$(RESET)"
	@cd $(BUILD_DIR) && $(CTEST) -R "$(TEST)" --output-on-failure

.PHONY: test-list
test-list: release  ## List all available tests
	@echo "$(CYAN)Available tests:$(RESET)"
	@cd $(BUILD_DIR) && $(CTEST) -N

# ============================================================================
# Code Formatting
# ============================================================================

.PHONY: format
format:  ## Format all source files with clang-format
	@echo "$(CYAN)Formatting source files...$(RESET)"
	@$(CLANG_FORMAT) -i $(SOURCES) $(TEST_SOURCES) $(EXAMPLE_SOURCES) 2>/dev/null || true
	@echo "$(GREEN)✓ Formatting complete$(RESET)"

.PHONY: format-check
format-check:  ## Check formatting without modifying files
	@echo "$(CYAN)Checking code formatting...$(RESET)"
	@FAILED=0; \
	for file in $(SOURCES) $(TEST_SOURCES) $(EXAMPLE_SOURCES); do \
		if [ -f "$$file" ]; then \
			$(CLANG_FORMAT) --dry-run --Werror "$$file" 2>/dev/null || FAILED=1; \
		fi; \
	done; \
	if [ $$FAILED -eq 1 ]; then \
		echo "$(RED)✗ Some files need formatting. Run 'make format' to fix.$(RESET)"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ All files properly formatted$(RESET)"; \
	fi

.PHONY: format-diff
format-diff:  ## Show formatting changes without applying
	@echo "$(CYAN)Showing format diff...$(RESET)"
	@for file in $(SOURCES) $(TEST_SOURCES) $(EXAMPLE_SOURCES); do \
		if [ -f "$$file" ]; then \
			diff -u "$$file" <($(CLANG_FORMAT) "$$file") 2>/dev/null || true; \
		fi; \
	done

# ============================================================================
# Static Analysis
# ============================================================================

# macOS SDK path for clang-tidy
MACOS_SDK := $(shell xcrun --show-sdk-path 2>/dev/null)
CLANG_TIDY_ARGS := --extra-arg=-isysroot$(MACOS_SDK) --extra-arg=-std=c++20

.PHONY: lint
lint: $(BUILD_DIR)/CMakeCache.txt  ## Run clang-tidy static analysis
	@echo "$(CYAN)Running clang-tidy...$(RESET)"
	@if ! command -v clang-tidy &> /dev/null; then \
		echo "$(YELLOW)clang-tidy not found. Install with: brew install llvm$(RESET)"; \
		exit 0; \
	fi
	@find src/tensor -name '*.cpp' | while read file; do \
		echo "  Checking $$file..."; \
		clang-tidy -p $(BUILD_DIR) $(CLANG_TIDY_ARGS) "$$file" --quiet 2>/dev/null || true; \
	done
	@echo "$(GREEN)✓ Static analysis complete$(RESET)"

.PHONY: lint-fix
lint-fix: $(BUILD_DIR)/CMakeCache.txt  ## Run clang-tidy and apply fixes
	@echo "$(CYAN)Running clang-tidy with fixes...$(RESET)"
	@if ! command -v clang-tidy &> /dev/null; then \
		echo "$(RED)clang-tidy not found. Install with: brew install llvm$(RESET)"; \
		exit 1; \
	fi
	@find src/tensor -name '*.cpp' | while read file; do \
		echo "  Fixing $$file..."; \
		clang-tidy -p $(BUILD_DIR) $(CLANG_TIDY_ARGS) "$$file" --fix --quiet 2>/dev/null || true; \
	done
	@echo "$(GREEN)✓ Fixes applied$(RESET)"

# ============================================================================
# Installation
# ============================================================================

.PHONY: install
install: release  ## Install to system (may require sudo)
	@echo "$(CYAN)Installing Axiom...$(RESET)"
	@$(CMAKE) --install $(BUILD_DIR)
	@echo "$(GREEN)✓ Installation complete$(RESET)"

.PHONY: install-local
install-local: release  ## Install to ./install directory
	@echo "$(CYAN)Installing to local directory...$(RESET)"
	@$(CMAKE) --install $(BUILD_DIR) --prefix ./install
	@echo "$(GREEN)✓ Installed to ./install$(RESET)"

.PHONY: uninstall
uninstall:  ## Uninstall from system
	@echo "$(YELLOW)Uninstalling Axiom...$(RESET)"
	@if [ -f $(BUILD_DIR)/install_manifest.txt ]; then \
		xargs rm -f < $(BUILD_DIR)/install_manifest.txt; \
		echo "$(GREEN)✓ Uninstall complete$(RESET)"; \
	else \
		echo "$(RED)No install manifest found$(RESET)"; \
	fi

# ============================================================================
# Cleaning
# ============================================================================

.PHONY: clean
clean:  ## Clean release build artifacts
	@echo "$(CYAN)Cleaning release build...$(RESET)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✓ Clean complete$(RESET)"

.PHONY: clean-debug
clean-debug:  ## Clean debug build artifacts
	@echo "$(CYAN)Cleaning debug build...$(RESET)"
	@rm -rf $(BUILD_DIR_DEBUG)
	@echo "$(GREEN)✓ Clean complete$(RESET)"

.PHONY: clean-all
clean-all: clean clean-debug  ## Clean all build artifacts
	@rm -rf install
	@echo "$(GREEN)✓ All builds cleaned$(RESET)"

.PHONY: distclean
distclean: clean-all  ## Clean everything including generated files
	@rm -rf compile_commands.json
	@echo "$(GREEN)✓ Distribution clean complete$(RESET)"

# ============================================================================
# Development Helpers
# ============================================================================

.PHONY: compile-commands
compile-commands: $(BUILD_DIR)/CMakeCache.txt  ## Generate compile_commands.json
	@echo "$(CYAN)Generating compile_commands.json...$(RESET)"
	@ln -sf $(BUILD_DIR)/compile_commands.json .
	@echo "$(GREEN)✓ compile_commands.json linked$(RESET)"

.PHONY: watch
watch:  ## Watch for changes and rebuild (requires fswatch)
	@if ! command -v fswatch &> /dev/null; then \
		echo "$(RED)Error: fswatch not installed. Install with: brew install fswatch$(RESET)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Watching for changes... (Ctrl+C to stop)$(RESET)"
	@fswatch -o include src tests examples | xargs -n1 -I{} make release

.PHONY: info
info: $(BUILD_DIR)/CMakeCache.txt  ## Show build configuration
	@echo "$(BOLD)Axiom Build Information$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "$(CYAN)Build directory:$(RESET)  $(BUILD_DIR)"
	@echo "$(CYAN)Parallel jobs:$(RESET)    $(NPROC)"
	@echo "$(CYAN)CMake version:$(RESET)    $$($(CMAKE) --version | head -1)"
	@echo "$(CYAN)Compiler:$(RESET)         $$($(CMAKE) -B $(BUILD_DIR) -N 2>/dev/null | grep -m1 'CXX compiler' || echo 'Run configure first')"
	@echo ""
	@$(CMAKE) -B $(BUILD_DIR) -N 2>/dev/null | grep -E "Metal support|Build type|Version" || true

.PHONY: loc
loc:  ## Count lines of code
	@echo "$(BOLD)Lines of Code$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "$(CYAN)Headers:$(RESET)   $$(find include -name '*.hpp' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}')"
	@echo "$(CYAN)Sources:$(RESET)   $$(find src -name '*.cpp' -o -name '*.mm' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}')"
	@echo "$(CYAN)Tests:$(RESET)     $$(find tests -name '*.cpp' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}')"
	@echo "$(CYAN)Examples:$(RESET)  $$(find examples -name '*.cpp' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}')"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "$(BOLD)Total:$(RESET)     $$(find include src tests examples -name '*.cpp' -o -name '*.hpp' -o -name '*.mm' -o -name '*.h' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}')"

# ============================================================================
# Examples
# ============================================================================

.PHONY: examples
examples: release  ## Build and list examples
	@echo "$(CYAN)Examples built. Run with:$(RESET)"
	@ls -1 $(BUILD_DIR)/examples/ 2>/dev/null | grep -v CMake | while read f; do \
		echo "  ./$(BUILD_DIR)/examples/$$f"; \
	done

.PHONY: run-example
run-example: release  ## Run an example (EXAMPLE=name)
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "$(RED)Error: specify example with EXAMPLE=name$(RESET)"; \
		echo "Available examples:"; \
		ls -1 $(BUILD_DIR)/examples/ 2>/dev/null | grep -v CMake; \
		exit 1; \
	fi
	@echo "$(CYAN)Running example: $(EXAMPLE)$(RESET)"
	@./$(BUILD_DIR)/examples/$(EXAMPLE)

# ============================================================================
# CI/CD Helpers
# ============================================================================

.PHONY: ci
ci: format-check release test  ## Run full CI pipeline

.PHONY: ci-quick
ci-quick: release test  ## Quick CI (skip format check)

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "$(BOLD)Axiom Makefile$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "$(BOLD)Build Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E '(release|debug|lib|configure|rebuild)' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Test Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E 'test' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Format Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E 'format' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Install Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E 'install' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Clean Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -E 'clean' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Other Targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -vE '(release|debug|lib|configure|rebuild|test|format|install|clean)' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  make                    # Build release"
	@echo "  make test               # Run all tests"
	@echo "  make test-single TEST=tensor_basic"
	@echo "  make format             # Format all code"
	@echo "  make ci                 # Full CI pipeline"
	@echo ""
