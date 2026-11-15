# ModelForge

Hardware-aware fine-tuning utility with browser, terminal, and notebook interfaces.

## Browser UI

`modelforge run` serves the UI from `web/frontend_dist/`. That directory is not committed. Source checkouts and wheel builds create it with the locked frontend build:

```bash
python scripts/build_frontend.py
# or, from web/frontend: npm ci && npm run build
```

`pip wheel .`, `python -m build`, and `uv build` run `npm ci` and `npm run build` automatically. The build fails if Node.js/npm is missing or the frontend build fails, so a wheel is not published without a UI. Installing an already-built wheel does not require Node.

```bash
modelforge run --no-browser
```

`/`, `/app`, and SPA deep links such as `/app/history` return `index.html`. See `DEVELOPER_GUIDE.md` for the local workflow and packaging smoke test.
