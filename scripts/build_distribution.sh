#!/usr/bin/env bash
# Supported release path: locked frontend build, then Python sdist and wheel.
# Fails if Node or the frontend bundle is missing rather than packaging an empty UI.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "error: Node.js and npm are required to build the frontend into the wheel." >&2
  echo "Install the version pinned in .node-version and retry." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to build the sdist and wheel." >&2
  exit 1
fi

node --input-type=module <<'EOF'
const [major, minor] = process.versions.node.split(".").map(Number);
const supported = (major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major > 22;
if (!supported) {
  console.error(
    `error: Node ${process.versions.node} cannot build the frontend. ` +
      "Use the version pinned in .node-version (Vite requires ^20.19.0 || >=22.12.0)."
  );
  process.exit(1);
}
EOF

echo "Building frontend with the committed lockfile"
npm ci --prefix web/frontend
npm run build --prefix web/frontend

if [[ ! -s web/frontend_dist/index.html ]]; then
  echo "error: frontend build did not produce web/frontend_dist/index.html" >&2
  exit 1
fi
if ! find web/frontend_dist/assets -type f -name '*.js' -size +0c | grep -q .; then
  echo "error: frontend build did not produce a JavaScript bundle under web/frontend_dist/assets" >&2
  exit 1
fi

echo "Building Python sdist and wheel"
rm -rf dist
uv build

python3 scripts/distribution_smoke.py --archives dist
echo "Distribution archives contain the frontend bundle."
