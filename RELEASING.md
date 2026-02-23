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
   | Repository | `agentguard-lib` |
   | Workflow name | `publish-pypi.yml` |
   | Environment name | `pypi` |
4. Save — GitHub Actions will authenticate via OIDC (no API token needed)

> **Note**: The trusted publisher is configured on the **public mirror** repo (`agentguard-lib`), not the monorepo. The `Sync Library` workflow pushes library changes to the mirror automatically.

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
#    - pyproject.toml   →  version = "0.3.0"
#    - agentguard/_version.py  →  __version__ = "0.3.0"

# 2. Commit the version bump
git add pyproject.toml agentguard/_version.py
git commit -m "release: bump library to v0.3.0"

# 3. Push to main (do NOT create tag on the monorepo)
git push origin main

# 4. Wait for the Sync Library workflow to push changes to agentguard-lib
#    This happens automatically on every push to main.

# 5. Create a GitHub Release on the PUBLIC MIRROR (agentguard-lib)
#    The Sync Library workflow does NOT push tags. You must create them
#    on the mirror repo directly.
gh release create v0.3.0 \
  --repo rlabs-cl/agentguard-lib \
  --title "v0.3.0" \
  --generate-notes

#    → The publish-pypi.yml workflow on agentguard-lib triggers automatically
#      (it fires on the `release: [published]` event, not on tag push)
```

After ~2 minutes, your package is live at:
- https://pypi.org/project/rlabs-agentguard/
- Install: `pip install rlabs-agentguard`

> **Important**: Tags and releases must be created on `agentguard-lib` (the public mirror), not on the monorepo. The `Sync Library` workflow uses `git subtree push` which does not sync tags.

### TypeScript SDK → npm

```bash
# 1. Bump the version in sdks/typescript/package.json
#    "version": "0.3.0"

# 2. Commit
git add sdks/typescript/package.json
git commit -m "release: sdk-v0.3.0"

# 3. Push to main, wait for Sync Library
git push origin main

# 4. Create release on agentguard-lib (tag must start with sdk-v)
gh release create sdk-v0.3.0 \
  --repo rlabs-cl/agentguard-lib \
  --title "sdk-v0.3.0" \
  --generate-notes

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
git commit -m "release: v0.3.0 + sdk-v0.3.0"

# Push to main
git push origin main

# Wait for Sync Library, then create TWO releases on agentguard-lib:
gh release create v0.3.0 --repo rlabs-cl/agentguard-lib --title "v0.3.0" --generate-notes
gh release create sdk-v0.3.0 --repo rlabs-cl/agentguard-lib --title "sdk-v0.3.0" --generate-notes
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
# Produces dist/rlabs_agentguard-0.3.0.tar.gz + rlabs_agentguard-0.3.0-py3-none-any.whl

# Test install from the wheel
pip install dist/rlabs_agentguard-0.3.0-py3-none-any.whl
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

| Component | Tag pattern | Example | Trigger | Repo |
|---|---|---|---|---|
| Python package | `v*` | `v0.3.0` | `publish-pypi.yml` | `agentguard-lib` |
| TypeScript SDK | `sdk-v*` | `sdk-v0.3.0` | `publish-npm.yml` | `agentguard-lib` |

Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **PATCH**: bug fixes, no API changes
- **MINOR**: new features, backwards compatible
- **MAJOR**: breaking API changes
