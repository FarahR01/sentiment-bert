# Integration Testing Guide - Sentiment BERT

## Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures and sample data
├── test_pipeline_integration.py     # Main integration test suite
└── README.md                        
```



## Running Tests

### Prerequisites
```bash
pip install pytest pytest-timeout pandas scikit-learn torch transformers
```

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Class
```bash
# Data pipeline tests only
pytest tests/test_pipeline_integration.py::TestDataPipeline -v

# End-to-end tests only
pytest tests/test_pipeline_integration.py::TestEndToEndPipeline -v
```

### Run Tests with Output
```bash
# Show print statements and detailed output
pytest tests/ -v -s
```

### Run Tests by Marker
```bash
# Mark tests in pytest.ini (requires @pytest.mark.integration)
pytest tests/ -m integration -v
```

### Run Single Test
```bash
pytest tests/test_pipeline_integration.py::TestDataPipeline::test_data_ingestion_and_validation -v
```

### Run with Timeout Protection
```bash
# Prevents hanging tests (timeout=300 seconds in pytest.ini)
pytest tests/ -v --timeout=300
```

---

## Key Design Principles

### 1. **Representative Sample Data**
- Uses 15-sample dataset balanced with positive/negative sentiments
- Avoids expensive operations (full IMDB dataset, pretrained BERT download)
- Validates realistic scenarios (sarcasm, sentiment markers)

### 2. **Component Interaction Testing**
Tests verify communication between modules:
- CSV → DataFrame → Dataset → DataLoader tensors
- Data format consistency at each stage
- Checkpoint save/restore cycle

### 3. **Explicit Failure Modes**
Each test documents what it catches:
- Data schema mismatches
- Tensor shape incompatibilities
- Gradient computation failures
- Checkpoint serialization bugs

### 4. **Fast Execution**
- Complete suite runs in <3 seconds
- Uses mock models instead of real pretrained weights
- Minimal temporary file I/O

---


## Extending the Tests

### Adding New Test Cases

1. **Create test method** in appropriate class:
```python
def test_new_component(self, fixture_name):
    """
    Test: Clear description of what is being tested.
    
    Validates:
    - Specific behavior 1
    - Specific behavior 2
    """
    # Implementation
    assert condition, "Failure message"
```

2. **Add appropriate pytest marker**:
```python
@pytest.mark.integration
@pytest.mark.slow
def test_expensive_operation(self):
    ...
```

3. **Run tests to verify**:
```bash
pytest tests/test_pipeline_integration.py::YourTestClass::test_new_component -v -s
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'app'"
**Solution**: Ensure tests run from project root:
```bash
cd c:\Users\Lenovo\Desktop\sentiment-bert
pytest tests/ -v
```

### Issue: CUDA out of memory (if running on GPU)
**Solution**: Mock model uses CPU by default—this shouldn't occur. If needed:
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/ -v
```

### Issue: Tests timeout
**Solution**: Check pytest.ini timeout setting (default 300s). Increase if needed:
```bash
pytest tests/ --timeout=600 -v
```

---

## Future Test Enhancements

- [ ] Add tests for W&B logging integration
- [ ] Add tests for ONNX export functionality
- [ ] Add performance benchmarking tests
- [ ] Add tests for FastAPI endpoints
- [ ] Add data augmentation validation tests
- [ ] Add tests for edge cases (empty reviews, very long reviews)
- [ ] Add parametrized tests for different hyperparameters

---

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Towards Data Science: Testing in ML](https://towardsdatascience.com/)
- [PyTorch Testing Guidelines](https://pytorch.org/docs/stable/testing.html)

---

**Last Updated**: March 20, 2026  
**Author**: Farah  

