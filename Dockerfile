# pagefault + HydraDB in one container: one service, one URL.
#   docker build -t pagefault . && docker run -p 8080:8080 -v pagefault-data:/data pagefault
# HydraDB stores its graph under /data (mount a volume). Set CLOUD_PROVIDER/S3 env
# to use object storage instead, exactly as for a standalone graph-node.

FROM ubuntu:24.04 AS build
ARG RUST_VERSION=1.99.0
ARG HYDRADB_REF=916637477f3f34766b4905b270bd4292109d77e6
ENV DEBIAN_FRONTEND=noninteractive CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates clang cmake curl git libclang-dev libcypher-parser-dev libffi-dev libgraphblas-dev pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION} && \
    rm -rf /var/lib/apt/lists/*
# HydraDB open-source engine, pinned
RUN git clone https://github.com/hydra-db/hydradb.git /hydradb && cd /hydradb && git checkout ${HYDRADB_REF} && \
    bash scripts/install_lagraph.sh /opt/lagraph && \
    LAGRAPH_LIB_DIR=/opt/lagraph/lib cargo build --locked --release --features server-runtime --bin graph-node && \
    strip --strip-debug target/release/graph-node
# pagefault
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --locked --release --bin server --bin replay && strip target/release/server target/release/replay

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive RUST_LOG=info
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libcypher-parser-dev libgraphblas-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /hydradb/target/release/graph-node /usr/local/bin/graph-node
COPY --from=build /app/target/release/server /usr/local/bin/pagefault-server
COPY --from=build /app/target/release/replay /usr/local/bin/pagefault-replay
WORKDIR /app
COPY web ./web
COPY results ./results
COPY data/demo ./data/demo
COPY deploy/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh && mkdir -p /data
EXPOSE 8080
CMD ["/usr/local/bin/start.sh"]
