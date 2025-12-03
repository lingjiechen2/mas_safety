
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

# Defense

## Consensus Defense Results (HotpotQA, mal_doc)

| # attacked agents | Topology | Attack | Defense | Runs | Score | Attack Success Rate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Chain | mal_doc | Consensus | 3 | 0.120 | 0.880 |
| 1 | Chain | mal_doc | Consensus | 5 | 0.210 | 0.760 |
| 1 | Chain | mal_doc | Consensus | 7 | 0.150 | 0.840 |
| 2 | Chain | mal_doc | Consensus | 3 | 0.050 | 0.960 |
| 2 | Chain | mal_doc | Consensus | 5 | 0.190 | 0.800 |
| 2 | Chain | mal_doc | Consensus | 7 | 0.120 | 0.880 |

## Redundancy Strategy experiments (Qwen3-4B)

### $N = 3$

| # attacked agents | Topology | Attack | Defense | Score | Attack Success Rate |
| - | - | - | - | - | - |
| 0 | Chain | None | None | 0.175 | 0.000 |
| 1 | Chain | Prompt Injection | None | 0.152 | 0.880 |
| 1 | Chain | Prompt Injection | Redundancy | 0.161 (↑0.010) | 0.920 (↑0.040) |
| 2 | Chain | Prompt Injection | None | 0.142 | 0.920 |
| 2 | Chain | Prompt Injection | Redundancy | 0.136 (↓0.006) | 1.000 (↑0.080) |
| 3 | Chain | Prompt Injection | None | 0.112 | 1.000 |
| 3 | Chain | Prompt Injection | Redundancy | 0.112 | 0.880 (↓0.120) |
