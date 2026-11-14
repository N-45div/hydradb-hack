#!/bin/bash
# Start HydraDB graph-node (private, localhost only) then the pagefault API on $PORT.
set -euo pipefail
DATA=${DATA_DIR:-/data}
mkdir -p "$DATA/store" "$DATA/cache"
# The HydraDB token never leaves the container: generate one if none is given.
if [ -z "${HYDRADB_TOKEN:-}" ]; then
  if [ ! -f "$DATA/auth-token" ]; then head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$DATA/auth-token"; fi
  export HYDRADB_TOKEN=$(cat "$DATA/auth-token")
else
  printf '%s' "$HYDRADB_TOKEN" > "$DATA/auth-token"
fi
export CLOUD_PROVIDER=${CLOUD_PROVIDER:-local} LOCAL_PATH=${LOCAL_PATH:-$DATA/store}
export GRAPH_NAMESPACE=default GRAPH_ID=default GRAPH_CELL_ID=cell-0 GRAPH_CELLS=cell-0 GRAPH_NODE_ID=node-0
export GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687
export GRAPH_DATA_CACHE_DIR=$DATA/cache GRAPH_AUTH_TOKEN_FILE=$DATA/auth-token GRAPH_ALLOW_PLAINTEXT=true
export RUST_MIN_STACK=33554432
graph-node > "$DATA/graph-node.log" 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:9090/readyz >/dev/null && break; sleep 1; done
echo "HydraDB ready"
export HYDRADB_URL=http://127.0.0.1:8443 PAGEFAULT_STORE=hydra PORT=${PORT:-8080}
exec pagefault-server
