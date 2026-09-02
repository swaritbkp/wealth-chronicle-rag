# Contributing to WealthChronicle AI

Thank you for your interest in contributing to **WealthChronicle AI** We welcome bug fixes, documentation improvements, architectural optimizations, and new feature contributions.

---

## 1. Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors. Please be respectful and constructive in all interactions.

---

## 2. Development Environment Setup

WealthChronicle AI is built on **Python 3.11**. Ensure you have Python 3.11 installed before proceeding.

### Local Setup Steps:

```bash
# 1. Clone the repository
git clone https://github.com/swaritbkp/wealth-chronicle-rag.git
cd wealth-chronicle-rag

# 2. Create Python 3.11 virtual environment
py -3.11 -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-mock ruff mypy
```

---

## 3. Code Standards & Static Analysis

Before submitting any code changes, ensure all linters, formatters, and type checkers pass without errors.

### Formatting & Linting
We use [Ruff](https://github.com/astral-sh/ruff) for lightning-fast linting and code formatting:

```bash
# Format code
ruff format .

# Check and auto-fix linting issues)
ruff check . --fix
```

### Static Type Checking
We enforce strict type annotations across all core modules using [MyPy](https://mypy-lang.org/):

```bash
mypy schemas.py engine.py ingest.py app.py --ignore-missing-imports
```
Ensure zero type errors before opening a pull request.

---

## 4. Test Verification Requirement

Every PR must maintain 100% test passing status. Our test suite includes unit tests, integration tests, concurrency stress tests, rate-limiter validations, and RAGAS faithfulness regression tests.

```bash
# Run the complete test battery (all 117 tests must pass)
pytest tests/ -v
```

---

## 5. Branching & PR Guidelines

### Branch Naming Conventions
- `feat/<feature-name>`: New capabilities or architectural improvements
- `fix/<bug-name>`: Bug fixes and error resolutions
- `chore/<task-name>`: Maintenance, dependency updates, and governance
- `docs/<doc-name>`: Documentation additions and updates
- `test/<test-name>`: Test harness improvements and golden set updates

### Pull Request (PR) Checklist
1. Branch created from latest `main`.
2. All 117 tests pass locally (`pytest tests/ -v`).
3. Linter checks pass (`ruff check . --fix`).
4. Type checks pass(`mypy schemas.py engine.py ingest.py app.py --ignore-missing-imports`).
5. Clear PR description explaining the rationale, changes made, and verification steps.
6. Secrets or proprietary data are NEVER committed.
