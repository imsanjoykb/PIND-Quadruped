# Contributing to PIND

Thank you for your interest in contributing to PIND! This document provides guidelines for contributions.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/PIND-Quadruped.git
cd PIND-Quadruped
```

2. Create a virtual environment and install in development mode:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting (line length: 100)
- Use `isort` for import sorting
- Add type hints where appropriate
- Write docstrings for all public functions/classes (Google style)

Format your code before committing:
```bash
black pind/ scripts/ tests/
isort pind/ scripts/ tests/
flake8 pind/ scripts/ tests/
```

## Testing

Write tests for new features:
```bash
pytest tests/ -v
pytest tests/ --cov=pind --cov-report=html
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes with clear, atomic commits
3. Add tests for new functionality
4. Update documentation as needed
5. Run tests and linting
6. Push and create a pull request

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive

## Reporting Issues

When reporting bugs, please include:
- OS and Python version
- PyTorch and CUDA versions (if using GPU)
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces

## Feature Requests

We welcome feature requests! Please:
- Check if the feature already exists or is planned
- Clearly describe the use case
- Explain why it would benefit the project

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

Open an issue or discussion on GitHub if you have questions about contributing.

Thank you for helping improve PIND!
