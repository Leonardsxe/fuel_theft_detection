# Test Suite for Fuel Theft Detection System

Comprehensive test suite covering unit tests, integration tests, and end-to-end workflows.

## 📁 Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                     # Pytest configuration
├── README.md                      # This file
│
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_thresholds.py        # Threshold calculation tests
│   ├── test_features.py          # Feature engineering tests
│   ├── test_models.py            # Model training/evaluation tests
│   ├── test_preprocessor.py      # Data preprocessing tests
│   ├── test_validator.py         # Data validation tests
│   ├── test_parser.py            # Data parsing tests
│   └── test_loader.py            # Data loading tests
│
├── integration/                   # Integration tests (components working together)
│   ├── test_pipeline.py          # Pipeline integration tests
│   └── test_end_to_end.py        # Complete workflow tests
│
└── data/                          # Test data and fixtures
    ├── sample_data.csv            # Sample telemetry data
    └── generate_sample_data.py   # Script to regenerate sample data
```

## 🚀 Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-timeout
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/ -m unit

# Integration tests only
pytest tests/integration/ -m integration

# Exclude slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/unit/test_thresholds.py

# Run specific test class
pytest tests/unit/test_thresholds.py::TestRobustSigmaMAD

# Run specific test
pytest tests/unit/test_thresholds.py::TestRobustSigmaMAD::test_robust_sigma_normal_distribution
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html --cov-report=term
```

Then open `htmlcov/index.html` to view detailed coverage report.

### Run with Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### Run in Parallel (faster)

```bash
pip install pytest-xdist
pytest -n auto  # Use all CPU cores
```

## 🏷️ Test Markers

Tests are marked with pytest markers for selective execution:

- **`@pytest.mark.unit`**: Fast unit tests (< 0.1s each)
- **`@pytest.mark.integration`**: Integration tests (multiple components)
- **`@pytest.mark.slow`**: Slow tests (> 1s)

Example:
```python
@pytest.mark.unit
def test_fast_unit():
    assert True

@pytest.mark.slow
@pytest.mark.integration
def test_slow_integration():
    # Complex multi-component test
    pass
```

## 🧪 Test Categories

### Unit Tests

**Purpose**: Test individual functions/classes in isolation

**Characteristics**:
- Fast (< 0.1s per test)
- No external dependencies
- Mock/stub external components
- High coverage of edge cases

**Files**:
- `test_thresholds.py` - Noise threshold calculation
- `test_features.py` - Feature engineering
- `test_models.py` - Model training/evaluation
- `test_preprocessor.py` - Data preprocessing
- `test_validator.py` - Data validation

### Integration Tests

**Purpose**: Test components working together

**Characteristics**:
- Moderate speed (0.1-2s per test)
- Use real components (no mocking)
- Test data flows between modules
- Verify interfaces and contracts

**Files**:
- `test_pipeline.py` - Pipeline components integration
- `test_end_to_end.py` - Complete workflows

### End-to-End Tests

**Purpose**: Test complete system workflows

**Characteristics**:
- Slower (2-30s per test)
- Use real data and models
- Test entire pipeline from input to output
- Verify production scenarios

**Files**:
- `test_end_to_end.py` - Full training and inference workflows

## 📊 Test Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Core detection logic | ≥ 90% |
| Feature engineering | ≥ 85% |
| Model training | ≥ 80% |
| Pipeline orchestration | ≥ 75% |
| Utility functions | ≥ 80% |

## 🔧 Writing New Tests

### Test Structure Template

```python
"""
Description of what this test file covers.
"""

import pytest
from src.module import function_to_test


class TestFeatureName:
    """Tests for specific feature/class."""
    
    def test_happy_path(self):
        """Test normal expected behavior."""
        result = function_to_test(valid_input)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test boundary conditions."""
        result = function_to_test(edge_case_input)
        assert result is handled correctly
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Using Fixtures

```python
def test_with_sample_data(sample_telemetry_df):
    """Fixtures are automatically provided by pytest."""
    assert len(sample_telemetry_df) > 0
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubles(input, expected):
    assert double(input) == expected
```

## 🐛 Debugging Tests

### Run with Debug Output

```bash
pytest --tb=long  # Full traceback
pytest -s         # Show print statements
pytest --pdb      # Drop to debugger on failure
```

### Run Single Test with Debug

```bash
pytest tests/unit/test_thresholds.py::test_specific -vv -s
```

### Check Test Collection

```bash
pytest --collect-only  # Show all tests that would run
```

## 📝 Test Data

### Sample Data Generation

Generate fresh test data:

```bash
python tests/data/generate_sample_data.py
```

This creates `tests/data/sample_data.csv` with:
- 3 vehicles
- 7 days of telemetry
- Synthetic theft events (labeled)
- ~30,000 data points

### Custom Test Data

Create custom fixtures in `conftest.py`:

```python
@pytest.fixture
def custom_test_data():
    """Your custom test data."""
    return pd.DataFrame({...})
```

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## 📈 Test Metrics

Run tests and generate metrics:

```bash
# Coverage report
pytest --cov=src --cov-report=term-missing

# Execution time report
pytest --durations=10  # Show 10 slowest tests

# Test report in XML (for CI)
pytest --junitxml=report.xml
```

## ❓ Troubleshooting

### Tests Failing After Code Changes

1. Check if test assumptions are still valid
2. Update fixtures if interfaces changed
3. Run single test with verbose output: `pytest path/to/test.py -vv -s`

### Import Errors

```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### Fixture Not Found

- Check `conftest.py` in test directory or parent
- Ensure fixture name matches exactly
- Check fixture scope (session, module, function)

### Tests Pass Locally But Fail in CI

- Check environment differences (Python version, dependencies)
- Verify test data exists in CI environment
- Check for hardcoded paths (use `Path(__file__).parent` instead)

## 📚 Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## ✅ Checklist for New Features

When adding a new feature:

- [ ] Write unit tests for core functionality
- [ ] Write integration tests for component interactions
- [ ] Add fixtures for common test data
- [ ] Update this README if test structure changes
- [ ] Ensure tests pass: `pytest`
- [ ] Check coverage: `pytest --cov=src`
- [ ] Mark slow tests with `@pytest.mark.slow`