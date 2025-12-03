#!/usr/bin/env bash

MAS_TOPOLOGY="chain"
MODEL_OPENAI="gpt-4o-mini"
MODEL_QWEN="Qwen/Qwen3-4B-Instruct-2507"
CONSENSUS_RUNS_LIST="3 5 7"
NUM_MAL_CLIENTS=3
CONSENSUS_STRATEGY="boxed"

# Consensus defense sweep: run against OpenAI-compatible API and local vLLM (Qwen3) in parallel.
# Adjust env vars OPENAI_BASE_URL / OPENAI_API_KEY as needed for the OpenAI path.

mkdir -p "./logs/hotpotqa/${MAS_TOPOLOGY}/"

# OpenAI-compatible API
for CONS_RUNS in ${CONSENSUS_RUNS_LIST}; do
python main.py \
  --mas "${MAS_TOPOLOGY}" \
  --dataset hotpotqa \
  --model "${MODEL_OPENAI}" \
  --attack mal_doc \
  --num_mal_clients "${NUM_MAL_CLIENTS}" \
  --defense consensus \
  --consensus_runs "${CONS_RUNS}" \
  --consensus_strategy "${CONSENSUS_STRATEGY}" \
  --num_samples 25 \
  --log_file "./logs/hotpotqa/${MAS_TOPOLOGY}/attack_mal_doc_${NUM_MAL_CLIENTS}_consensus_${CONSENSUS_STRATEGY}_openai_${CONS_RUNS}.txt" &
done

# Local vLLM (Qwen3) on host/port provided
for CONS_RUNS in ${CONSENSUS_RUNS_LIST}; do
python main.py \
  --mas "${MAS_TOPOLOGY}" \
  --dataset hotpotqa \
  --use_vllm_model \
  --vllm_host 0.0.0.0 \
  --vllm_port 8000 \
  --model "${MODEL_QWEN}" \
  --attack mal_doc \
  --num_mal_clients "${NUM_MAL_CLIENTS}" \
  --defense consensus \
  --consensus_runs "${CONS_RUNS}" \
  --consensus_strategy "${CONSENSUS_STRATEGY}" \
  --num_samples 25 \
  --log_file "./logs/hotpotqa/${MAS_TOPOLOGY}/attack_mal_doc_${NUM_MAL_CLIENTS}_consensus_${CONSENSUS_STRATEGY}_qwen_${CONS_RUNS}.txt" &
done

# wait
