# Repository Guidelines

## Project Structure & Module Organization
`moe_infinity/` contains the Python package: model wrappers, runtime hooks, memory/offload logic, policies, and OpenAI-style entrypoints. `core/` and `extensions/` hold the C++/CUDA sources compiled into the native `_store` and `_engine` modules. Tests are split by layer: `tests/python/{unit,integration,benchmark}`, `tests/docker`, `tests/cpp/unittest`, and `tests/cuda`. Use `examples/` for runnable demos and `benchmarks/` for reproducible benchmark drivers and trace files.

## Build, Test, and Development Commands
- `pip install -e .` builds and installs the package from source, including CUDA extensions when PyTorch CUDA is available.
- `pip install -r requirements-lint.txt` installs repo formatting, lint, and typing tools.
- `pre-commit run --all-files` runs Ruff, `ruff-format`, `clang-format`, `codespell`, and scoped `mypy`.
- `python -m pytest tests/python/unit -q` runs fast Python unit tests.
- `python tests/docker/run_tests.py` runs Tier 1 CPU tests and Tier 2 CUDA tests when a GPU is visible.
- `cmake -S tests/cpp/unittest/utils -B build/utils && cmake --build build/utils && ctest --test-dir build/utils` runs the C++ unit suite.
- `cmake -S tests/cuda -B build/cuda -DCUTLASS_DIR=$CUTLASS_DIR && cmake --build build/cuda` builds CUDA test binaries.

## Coding Style & Naming Conventions
Use 4-space indentation in Python. Keep lines near the configured 80-column Ruff limit and prefer double quotes. Name Python modules and test files in `snake_case`, classes in `CamelCase`, and constants in `SCREAMING_SNAKE_CASE`. Follow existing style in `core/` and `extensions/`; native code is formatted with `clang-format` and targets C++17/CUDA 17.

## Testing Guidelines
Add tests beside the affected layer: Python behavior in `tests/python/unit`, API/server behavior in `tests/python/integration`, kernels in `tests/cuda`, and allocator or queue behavior in `tests/cpp/unittest`. Name Python tests `test_<feature>.py`. Mark GPU-only pytest cases with `@pytest.mark.cuda` so they auto-skip on CPU-only hosts.

## Commit & Pull Request Guidelines
Use conventional commits with a scope, such as `fix(parallel): ...` or `feat(policy): ...`. Keep commits focused; separate Python API, kernel, and benchmark work when practical. PRs should include the problem being solved, a short change summary, test commands run, and any CUDA, CUTLASS, or model assumptions. Include benchmark output when changing offloading, scheduling, or kernel execution paths.

## Configuration Tips
Set `CUTLASS_DIR` before source builds or CUDA test builds. Use a unique `offload_path` per model to avoid cross-model cache contamination.
