# _shims/openpi_server

Vendored copy of `websocket_policy_server.py` from the `rlinf-openpi` package
(same file that ships `openpi.serving.websocket_policy_server`).

Why: `action_policy_server_robolab.py` imports the OpenPI WebSocket server via
`openpi_server.*` (first choice) or `openpi.serving.*`. Neither exists in the
`cosmos3` env, and installing full `openpi` pulls JAX + breaks on Python 3.13
(`dm-tree` fails to build). The module itself only needs stdlib +
`openpi_client` + `websockets`, so a single-file vendor is sufficient.

`scripts/start_cosmos3_policy_server.py` puts this dir on `sys.path` at launch.
Do NOT edit the vendored file; re-copy on upstream change.
