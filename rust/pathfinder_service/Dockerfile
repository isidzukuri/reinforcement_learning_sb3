FROM rust:1-bookworm AS builder

WORKDIR /build/rust

COPY pathfinder/Cargo.toml pathfinder/Cargo.toml
COPY pathfinder/src/ pathfinder/src/
COPY pathfinder_service/Cargo.toml pathfinder_service/Cargo.toml
COPY pathfinder_service/Cargo.lock pathfinder_service/Cargo.lock
COPY pathfinder_service/src/ pathfinder_service/src/
COPY pathfinder_service/model.onnx pathfinder_service/model.onnx

RUN cargo build --locked --release --manifest-path pathfinder_service/Cargo.toml

FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

WORKDIR /app

COPY --from=builder /build/rust/pathfinder_service/target/release/pathfinder_service /usr/local/bin/pathfinder_service
COPY --from=builder /build/rust/pathfinder_service/model.onnx /app/model.onnx

ENV PATHFINDER_ADDR=0.0.0.0:3005 \
    PATHFINDER_MODEL_PATH=/app/model.onnx

EXPOSE 3005
USER nonroot:nonroot

ENTRYPOINT ["/usr/local/bin/pathfinder_service"]
