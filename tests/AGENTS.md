# Agents — tests/

## Conventions

- Mirror `src/multi_time/` subpackage structure in test subfolders
- Shared fixtures in top-level `conftest.py`
- Each test subfolder has `__init__.py` for pytest discovery
- Test function names always start with `test_`
- Import source functions under aliases if they start with `test_` to prevent pytest collection conflicts
- All tests use real methods — no mocks or fakes
