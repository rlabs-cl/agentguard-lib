# Contributing to AgentGuard

We welcome contributions! This guide will help you get set up and submit your work.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

### Set Up Your Dev Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/rlabs-cl/agentguard-lib.git
   cd agentguard-lib
   ```

2. Install dependencies in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Verify your setup by running tests:
   ```bash
   pytest tests/
   ```

### Running Tests

Run the full test suite:
```bash
pytest tests/
```

Run tests for a specific module:
```bash
pytest tests/test_skeleton.py
```

Run with coverage:
```bash
pytest tests/ --cov=agentguard
```

### Code Quality Checks

Check code style and linting:
```bash
ruff check agentguard/
```

Format code:
```bash
ruff format agentguard/
```

## Creating a Custom Archetype

Custom archetypes extend AgentGuard with project-specific patterns. Here's how to create one:

### 1. Create the Archetype Directory

```bash
mkdir -p ~/.agentguard/archetypes/my_archetype
```

### 2. Create `archetype.yaml`

Define your archetype configuration:

```yaml
id: my_archetype
name: My Custom Archetype
description: A custom archetype for my use case
stack: [python, fastapi, postgres]

skeleton:
  description: "Create a project structure for X"
  files:
    src/app.py: "Application entry point"
    src/config.py: "Configuration module"
    tests/test_app.py: "App tests"

contracts:
  description: "Generate typed stubs and interfaces"

logic:
  description: "Implement function bodies"

challenge:
  criteria:
    - "All functions have docstrings"
    - "Error handling is explicit"
    - "Type hints are complete"
```

### 3. Add Supporting Files (Optional)

Place template files or schemas in your archetype directory. These can be referenced in `archetype.yaml`.

### 4. Test Your Archetype

```bash
agentguard reload_archetypes
agentguard list_archetypes | grep my_archetype
agentguard get_archetype my_archetype
```

### 5. Publish to the Marketplace

Once tested:

1. Create a PR with your archetype in `agentguard/archetypes/`
2. Include documentation explaining your design choices
3. Ensure all tests pass
4. After merge, it will be available on the [AgentGuard Marketplace](https://agentguard.rlabs.cl/marketplace)

## Submitting Pull Requests

### Fork and Branch

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/agentguard-lib.git
   cd agentguard-lib
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Make Your Changes

1. Write or update code
2. Add tests for new functionality
3. Run tests and code checks:
   ```bash
   pytest tests/
   ruff check agentguard/
   ```

### Commit and Push

```bash
git add .
git commit -m "feat: clear description of your changes"
git push origin feature/your-feature-name
```

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` tests
- `refactor:` code restructuring
- `chore:` maintenance

### Open a Pull Request

1. Go to GitHub and open a PR from your branch to `main`
2. Include:
   - Clear description of what you're changing and why
   - Reference any related issues (e.g., "Closes #42")
   - Evidence that tests pass
3. Wait for review and address feedback

## Need Help?

- Check out [README.md](README.md) for overview
- Browse [issues labeled `good first issue`](https://github.com/rlabs-cl/agentguard-lib/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- Open a discussion if you have questions

Thank you for contributing!
