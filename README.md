# Decizion Router

**A local decision service for coding agents.** Decizion Router classifies a request into a tool call, an agent lane, retrieval needs, and whether clarification is required. It exposes the decision over HTTP and MCP so tools such as Codex can use one consistent routing interface.

## What it does

- Routes requests through `/v1/route` and exposes status, policy, and feedback endpoints.
- Applies confidence and risk policies. A model prediction does not grant permission to run a write or destructive action.
- Provides `/v1/systemone` for typed evaluations; the companion [Decision-Grep](https://github.com/oapache/decision-grep) model can be selected there for code relevance scoring.
- Supports local GPU inference with CUDA.

## Model availability

The source code is published first. **The trained model weights are planned for Hugging Face soon.** The checkpoints are intentionally not stored in this Git repository. Inference setup will be documented when those files are available.

## Security

This service is intended for local or trusted-network use. The current API does not authenticate administrative endpoints; do not expose it directly to the public internet.

## Status

Early source release. Model artifacts and complete installation instructions will follow on Hugging Face.
