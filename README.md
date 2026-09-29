# Decizion Router

**Our fine-tuned decision model and local routing service for coding agents.** Decizion Router classifies a request into a tool call, an agent lane, retrieval needs, and whether clarification is required. It exposes the decision over HTTP and MCP so tools such as Codex can use one consistent routing interface.

## The model

Decizion Router is not just a wrapper around a hosted API: we fine-tuned the open [Laya multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) checkpoint for coding-agent orchestration. The base is a non-autoregressive mmBERT model with **322 million parameters**. Our active fine-tuned router checkpoint is `router-v8-cal5`; fine-tuning changes its decision behavior while retaining the 322M-parameter architecture. The service runs inference locally, with CUDA and FP16 configured for GPU serving.

The fine-tuned checkpoint is approximately **1.29 GB**. We plan to publish the model weights on Hugging Face soon; they are not included in this Git repository.

## What it does

- Routes requests through `/v1/route` and exposes status, policy, and feedback endpoints.
- Applies confidence and risk policies. A model prediction does not grant permission to run a write or destructive action.
- Provides `/v1/systemone` for typed evaluations; the companion [Decision-Grep](https://github.com/oapache/decision-grep) model can be selected there for code relevance scoring.
- Supports local GPU inference with CUDA.

## Security

This service is intended for local or trusted-network use. The current API does not authenticate administrative endpoints; do not expose it directly to the public internet.

## Status

Source release is available here. The fine-tuned weights and complete installation instructions will follow on Hugging Face.
