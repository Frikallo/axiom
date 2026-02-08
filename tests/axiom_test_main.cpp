#include "axiom_test_utils.hpp"

// Register AxiomEnvironment so ops are initialized before any test runs.
// gtest_main provides main(), so we use a static-init trick.
static auto *const kAxiomEnv =
    ::testing::AddGlobalTestEnvironment(new axiom::testing::AxiomEnvironment);
