# Developer Guide

Build from source, add new operations, understand the architecture, and contribute to Axiom.

```{toctree}
:hidden:

building
testing
adding-operations
backend-architecture
code-style
contributing
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} Building
:link: building
:link-type: doc

CMake options, platform prerequisites, Make targets, and BLAS configuration.
:::

:::{grid-item-card} Testing
:link: testing
:link-type: doc

Test framework, ASSERT/RUN_TEST macros, writing tests, and GPU test skipping.
:::

:::{grid-item-card} Adding Operations
:link: adding-operations
:link-type: doc

Step-by-step guide: OpType, CPU kernel, GPU kernel, API, and tests.
:::

:::{grid-item-card} Backend Architecture
:link: backend-architecture
:link-type: doc

Storage abstraction, CPU/Metal backends, memory model, and dispatch system.
:::

:::{grid-item-card} Code Style
:link: code-style
:link-type: doc

clang-format, naming conventions, C++20 patterns, and static analysis.
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

Contribution workflow, PR checklist, and development setup.
:::
::::
