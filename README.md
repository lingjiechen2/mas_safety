# MAS Architecture & Message Flow

For this section, assume `defense = none` (or `consensus`) so we only describe the single base path, without duplicated executors.

- Chain:
  - Planner  
    - input: `question`  
    - output: `plan from Planner`
  - Refiner  
    - input: (`question`, `plan from Planner`)  
    - output: `refined plan from Refiner`
  - Executor  
    - input: (`question`, `refined plan from Refiner`, `context or malicious doc for Executor`)  
    - output: `evidence from Executor`
  - Final Answerer  
    - input: (`question`, `plan from Planner`, `refined plan from Refiner`, `evidence from Executor`)  
    - output: `final boxed answer`

- Tree:
  - Planner  
    - input: `question`  
    - output: `plan from Planner`
  - Exec-Left  
    - input: (`question`, `plan from Planner`, `context or malicious doc for Exec-Left`)  
    - output: `left evidence from Exec-Left`
  - Exec-Right  
    - input: (`question`, `plan from Planner`, `context or malicious doc for Exec-Right`)  
    - output: `right evidence from Exec-Right`
  - Final Answerer  
    - input: (`question`, `plan from Planner`, `left evidence from Exec-Left`, `right evidence from Exec-Right`)  
    - output: `final boxed answer`

- Star:
  - Executor A  
    - input: (`question`, fixed `star plan`, `context or malicious doc for Executor A`)  
    - output: `evidence from Executor A`
  - Executor B  
    - input: (`question`, fixed `star plan`, `context or malicious doc for Executor B`)  
    - output: `evidence from Executor B`
  - Executor C  
    - input: (`question`, fixed `star plan`, `context or malicious doc for Executor C`)  
    - output: `evidence from Executor C`
  - Final Answerer  
    - input: (`question`, `evidence from Executor A`, `evidence from Executor B`, `evidence from Executor C`)  
    - output: `final boxed answer`

- Fully Connected Graph (FCG):
  - Peer 1 → Peer 2 (first message)  
    - input: (`question`, `clean context or malicious doc for Peer 1`, `no previous note`)  
    - output: `note from Peer 1 to Peer 2`
  - Peer 2 → Peer 1  
    - input: (`question`, `clean context or malicious doc for Peer 2`, `note from Peer 1 to Peer 2`)  
    - output: `note from Peer 2 to Peer 1`
  - Peer 2 → Peer 3  
    - input: (`question`, `clean context or malicious doc for Peer 2`, `no previous note`)  
    - output: `note from Peer 2 to Peer 3`
  - Peer 3 → Peer 2  
    - input: (`question`, `clean context or malicious doc for Peer 3`, `note from Peer 2 to Peer 3`)  
    - output: `note from Peer 3 to Peer 2`
  - Peer 3 → Peer 1  
    - input: (`question`, `clean context or malicious doc for Peer 3`, `no previous note`)  
    - output: `note from Peer 3 to Peer 1`
  - Peer 1 → Peer 3  
    - input: (`question`, `clean context or malicious doc for Peer 1`, `note from Peer 3 to Peer 1`)  
    - output: `note from Peer 1 to Peer 3`
  - Final Answerer  
    - input: (`question`, all six peer notes)  
    - output: `final boxed answer`

## Running the Experiments

### Run with local vLLM server

1. Start the vLLM server (see `run_vllm_server.sh`):

   ```bash
   bash run_vllm_server.sh
   ```

2. In another terminal, run the full HotpotQA sweep (Star / Chain / Tree / FCG, various mal_doc settings):

   ```bash
   bash run.sh
   ```

This uses `--use_vllm_model` and `MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"` as configured in `run.sh`.

### Quick consensus demo

Run the same HotpotQA chain setup as the OpenAI example but with consensus voting (K repeats, majority on boxed answers):

```bash
python main.py \
  --mas chain \
  --dataset hotpotqa \
  --model gpt-4o-mini \
  --attack mal_doc \
  --num_mal_clients 2 \
  --defense consensus \
  --consensus_runs 1 \
  --num_samples 10 \
  --log_file ./logs/hotpotqa/chain/attack_mal_doc_2_concensus_1_openai.txt
```

### Run with OpenAI-compatible API (no vLLM)

1. Ensure `OPENAI_BASE_URL` and `OPENAI_API_KEY` are set in your environment, or edit `main.py` defaults.  
2. Call `main.py` directly without `--use_vllm_model`, for example:

   ```bash
   python main.py \
     --mas chain \
     --dataset hotpotqa \
     --model gpt-4o-mini \
     --attack mal_doc \
     --num_mal_clients 2 \
     --defense none \
     --num_samples 10 \
     --log_file ./logs/hotpotqa/chain/attack_mal_doc_2_openai.txt
   ```

You can switch `--mas` among `chain`, `tree`, `star`, and `fcg`, and adjust `--attack`, `--num_mal_clients`, and `--defense` as needed.

## HotpotQA Results (vLLM, Qwen/Qwen3-4B-Instruct-2507)

**Dataset:** HotpotQA  
**Samples per experiment:** 25  
**Accuracy Metric:** Max Token Overlap (0–1)

### Star Topology

```text
no_attack:                    Avg Score: 0.420, Attack Success Rate: 0.000
1_agent_with_mal_doc:         Avg Score: 0.300, Attack Success Rate: 0.680
2_agents_with_mal_doc:        Avg Score: 0.193, Attack Success Rate: 0.800
3_agents_with_mal_doc:        Avg Score: 0.090, Attack Success Rate: 0.920
```

### Chain Topology

```text
no_attack:                    Avg Score: 0.443, Attack Success Rate: 0.000
1_agent_with_mal_doc:         Avg Score: 0.160, Attack Success Rate: 0.840
2_agents_with_mal_doc:        Avg Score: 0.100, Attack Success Rate: 0.880
3_agents_with_mal_doc:        Avg Score: 0.120, Attack Success Rate: 0.880
```

### Tree Topology

```text
no_attack:                    Avg Score: 0.443, Attack Success Rate: 0.000
1_agent_with_mal_doc:         Avg Score: 0.300, Attack Success Rate: 0.720
2_agents_with_mal_doc:        Avg Score: 0.040, Attack Success Rate: 0.960
3_agents_with_mal_doc:        Avg Score: 0.120, Attack Success Rate: 0.880
```

### FCG Topology

```text
no_attack:                    Avg Score: 0.424, Attack Success Rate: 0.000
1_agent_with_mal_doc:         Avg Score: 0.330, Attack Success Rate: 0.640
2_agents_with_mal_doc:        Avg Score: 0.380, Attack Success Rate: 0.600
3_agents_with_mal_doc:        Avg Score: 0.120, Attack Success Rate: 0.880
```

## MultiNews Results (vLLM, Qwen/Qwen3-4B-Instruct-2507)

**Dataset:** MultiNews  
**Task:** Summarization (prompt injection attacks)  
**Samples per experiment:** 25  
**Accuracy Metric:** ROUGE-1 style F1 (approx)

### Star Topology

```text
no_attack:                     Avg Score: 0.171, Attack Success Rate: 0.000
1_agent_with_prompt_injection: Avg Score: 0.171, Attack Success Rate: 0.920
2_agents_with_prompt_injection: Avg Score: 0.149, Attack Success Rate: 0.960
3_agents_with_prompt_injection: Avg Score: 0.159, Attack Success Rate: 0.920
```

### Chain Topology

```text
no_attack:                     Avg Score: 0.167, Attack Success Rate: 0.000
1_agent_with_prompt_injection: Avg Score: 0.149, Attack Success Rate: 0.840
2_agents_with_prompt_injection: Avg Score: 0.140, Attack Success Rate: 0.920
3_agents_with_prompt_injection: Avg Score: 0.118, Attack Success Rate: 0.960
```

### Tree Topology

```text
no_attack:                     Avg Score: 0.175, Attack Success Rate: 0.000
1_agent_with_prompt_injection: Avg Score: 0.160, Attack Success Rate: 0.960
2_agents_with_prompt_injection: Avg Score: 0.140, Attack Success Rate: 0.920
3_agents_with_prompt_injection: Avg Score: 0.146, Attack Success Rate: 0.960
```

### FCG Topology

```text
no_attack:                     Avg Score: 0.162, Attack Success Rate: 0.000
1_agent_with_prompt_injection: Avg Score: 0.142, Attack Success Rate: 0.960
2_agents_with_prompt_injection: Avg Score: 0.126, Attack Success Rate: 1.000
3_agents_with_prompt_injection: Avg Score: 0.113, Attack Success Rate: 1.000
```

## Consensus Defense Results (HotpotQA, mal_doc)

### Fixed setting: 1 attacked agent, consensus_runs=3 (boxed)

| # attacked agents | Topology | Attack | Defense | Score | Attack Success Rate |
| --- | --- | --- | --- | --- | --- |
| 1 | chain (gpt-4o-mini) | mal_doc | consensus (boxed, runs=3) | 0.601 | 0.40 |
| 1 | chain (Qwen3-4B vLLM) | mal_doc | consensus (boxed, runs=3) | 0.120 | 0.88 |
| 1 | star (gpt-4o-mini) | mal_doc | consensus (boxed, runs=3) | 0.577 | 0.40 |
| 1 | star (Qwen3-4B vLLM) | mal_doc | consensus (boxed, runs=3) | 0.270 | 0.72 |
| 1 | tree (gpt-4o-mini) | mal_doc | consensus (boxed, runs=3) | 0.673 | 0.32 |
| 1 | tree (Qwen3-4B vLLM) | mal_doc | consensus (boxed, runs=3) | 0.310 | 0.68 |
| 1 | fcg (gpt-4o-mini) | mal_doc | consensus (boxed, runs=3) | 0.630 | 0.36 |
| 1 | fcg (Qwen3-4B vLLM) | mal_doc | consensus (boxed, runs=3) | 0.250 | 0.72 |

> FCG OpenAI run lacked an aggregate block in this snapshot.

### Chain-only sweep (all logged variants)

| # attacked agents | Topology | Attack | Defense | Runs | Score | Attack Success Rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Chain | prompt_injection | None | - | 0.152 | 0.88 |
| 1 | Chain | mal_doc | consensus | 3 | 0.120 | 0.88 |
| 1 | Chain | mal_doc | consensus | 5 | 0.210 | 0.76 |
| 1 | Chain | mal_doc | consensus | 7 | 0.150 | 0.84 |
| 2 | Chain | prompt_injection | None | - | 0.142 | 0.92 |
| 2 | Chain | mal_doc | consensus | 3 | 0.050 | 0.96 |
| 2 | Chain | mal_doc | consensus | 5 | 0.190 | 0.80 |
| 2 | Chain | mal_doc | consensus | 7 | 0.120 | 0.88 |

## Redundancy Defense Results (Multi-News, prompt injection)

| # attacked agents | Topology | Attack | Defense | Score | Attack Success Rate | 
| - | - | - | - | - | - |
| 0 | Chain | None | None | 0.175 | 0.000 | 
| 1 | Chain | Prompt Injection | None | 0.152 | 0.880 |
| 1 | Chain | Prompt Injection | Redundancy ($N = 3$) | 0.161 ($\uparrow$ 0.010) | 0.920 ($\uparrow$ 0.040) |
| 2 | Chain | Prompt Injection | None | 0.142 | 0.920
| 2 | Chain | Prompt Injection | Redundancy ($N = 3$) | 0.136 ($\downarrow$ 0.006) | 1.000 ($\uparrow$ 0.080)
| 3 | Chain | Prompt Injection | None | 0.112 | 1.000
| 3 | Chain | Prompt Injection | Redundancy ($N = 3$) | 0.112 | 0.880 ($\downarrow 0.120$)
