# Releasing AgentGuard

This document describes how to publish new versions of AgentGuard to **PyPI** (Python) and **npm** (TypeScript SDK).

---

## Prerequisites (one-time setup)

### PyPI — Trusted Publisher (recommended, no tokens)

1. Go to [pypi.org](https://pypi.org) → Create an account or log in
2. Navigate to **Publishing** → **Add a new pending publisher**
3. Fill in the form:
   | Field | Value |
   |---|---|
   | PyPI project name | `rlabs-agentguard` |
   | Owner | `rlabs-cl` |
   | Repository | `AgentGuard` |
   | Workflow name | `publish-pypi.yml` |
   | Environment name | `pypi` |
4. Save — GitHub Actions will authenticate via OIDC (no API token needed)

> **Optional**: repeat for Test PyPI at [test.pypi.org](https://test.pypi.org) with environment name `testpypi`.

### npm — Access Token

1. Go to [npmjs.com](https://www.npmjs.com) → Create an account or log in
2. Profile → **Access Tokens** → **Generate New Token** → "Granular Access Token"
3. Set permissions: **Read and Write** for packages
4. Copy the token
5. In GitHub: repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
   - Name: `NPM_TOKEN`
   - Value: *(paste your token)*

### GitHub Environments (recommended)

Create these environments in GitHub (repo → Settings → Environments) to get deploy protection rules and a clean audit log:

- `pypi` — for PyPI publishes
- `testpypi` — for Test PyPI publishes  
- `npm` — for npm publishes

You can add manual approval requirements on each environment for extra safety.

---

## How to Release

### Python package → PyPI

```bash
# 1. Bump the version in TWO places (keep them in sync):
#    - pyproject.toml   →  version = "0.2.0"
#    - agentguard/_version.py  →  __version__ = "0.2.0"

# 2. Commit the version bump
git add pyproject.toml agentguard/_version.py
git commit -m "release: v0.2.0"

# 3. Tag it (must match pattern v*)
git tag v0.2.0

# 4. Push commit + tag
git push origin main --tags

# 5. Go to GitHub → Releases → "Create a new release"
#    - Choose tag: v0.2.0
#    - Title: v0.2.0
#    - Auto-generate release notes or write your own
#    - Click "Publish release"
#    → The publish-pypi.yml workflow runs automatically
```

After ~2 minutes, your package is live at:
- https://pypi.org/project/rlabs-agentguard/
- Install: `pip install rlabs-agentguard`

### TypeScript SDK → npm

```bash
# 1. Bump the version in sdks/typescript/package.json
#    "version": "0.2.0"

# 2. Commit
git add sdks/typescript/package.json
git commit -m "release: sdk-v0.2.0"

# 3. Tag it (must start with sdk-v to trigger the npm workflow)
git tag sdk-v0.2.0

# 4. Push
git push origin main --tags

# 5. Create a GitHub Release from the sdk-v0.2.0 tag
#    → The publish-npm.yml workflow runs automatically
```

After ~1 minute, your package is live at:
- https://www.npmjs.com/package/@agentguard/sdk
- Install: `npm install @agentguard/sdk`

---

## Releasing both at the same time

If you're bumping both Python and TypeScript:

```bash
# Bump both versions, commit together
git add pyproject.toml agentguard/_version.py sdks/typescript/package.json
git commit -m "release: v0.2.0 + sdk-v0.2.0"

# Create both tags
git tag v0.2.0
git tag sdk-v0.2.0

# Push everything
git push origin main --tags

# Create TWO GitHub Releases: one for v0.2.0, one for sdk-v0.2.0
```

---

## CI Pipeline

Every push to `main` and every pull request runs the **CI workflow** ([ci.yml](.github/workflows/ci.yml)):

- **Python**: lint (ruff) → type check (mypy) → test (pytest) across Python 3.11/3.12/3.13
- **TypeScript**: type check (tsc) → build

---

## Testing a release locally

### Python
```bash
# Build locally
pip install build
python -m build
# Produces dist/rlabs_agentguard-0.1.0.tar.gz + rlabs_agentguard-0.1.0-py3-none-any.whl

# Test install from the wheel
pip install dist/rlabs_agentguard-0.1.0-py3-none-any.whl
agentguard --version
```

### TypeScript
```bash
cd sdks/typescript
npm install
npm run build
# Check dist/ has .js + .d.ts files

# Dry-run publish (does NOT actually publish)
npm publish --dry-run
```

---

## Version scheme

| Component | Tag pattern | Example | Trigger |
|---|---|---|---|
| Python package | `v*` | `v0.2.0` | `publish-pypi.yml` |
| TypeScript SDK | `sdk-v*` | `sdk-v0.2.0` | `publish-npm.yml` |

Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **PATCH**: bug fixes, no API changes
- **MINOR**: new features, backwards compatible
- **MAJOR**: breaking API changes
