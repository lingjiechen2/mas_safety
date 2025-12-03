"""
Defense-enabled MAS topologies with redundancy and consensus mechanisms.

This module implements defense strategies for multi-agent systems:
- Redundancy: Generate N candidates at each node and use majority voting
- Consensus: Run the full pipeline multiple times and majority-vote the final answers

For attack-only simulations, use mas.py instead.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import pdb

from agents import (
    ANSWER_SYSTEM_BASE,
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    REFINER_SYSTEM,
    PEER_SYSTEM,
    AgentRun,
    answer_prompt,
    executor_prompt,
    peer_prompt,
    planner_prompt,
    refiner_prompt,
    run_agent,
    run_agent_with_redundancy,
)
from llm_client import load_env_vars
from voting import majority_vote

# Import shared utilities from mas.py
from mas import (
    PROMPT_INJECTION_STRINGS,
    MAL_DOC_SNIPPETS,
    BENIGN_CONTEXT,
    MASResult,
    normalize_attack,
    normalize_defense,
    choose_targets,
    sample_prompt_injection,
    sample_mal_doc,
    attack_summary_text,
)


def executor_bundle_defense(
    base_name: str,
    count: int,
    question: str,
    plan_text: str,
    model: str,
    temperature: float,
    attack_type: str,
    targeted: bool,
    prompt_injection_payload: Optional[str],
    clean_context: str,
    use_redundancy: bool = False,
    redundancy_count: int = 3,
    voting_strategy: str = "semantic",
    similarity_threshold: float = 0.7,
) -> List[AgentRun]:
    """
    Create executor agents with optional redundancy defense.

    When use_redundancy is True:
        - Each executor node generates redundancy_count candidates
        - Majority voting selects the final output at each node
        - Only the voted output is returned

    When use_redundancy is False:
        - Runs normally, count executors without voting (same as mas.py)

    Args:
        base_name: Base name for executor agents (e.g., "executor", "exec_left")
        count: Number of executor nodes to create
        question: The question being answered
        plan_text: The plan from upstream (planner/refiner)
        model: LLM model name
        temperature: Sampling temperature
        attack_type: "prompt_injection", "mal_doc", or "none"
        targeted: Whether this executor is targeted by the attack
        prompt_injection_payload: The injection string if targeted
        clean_context: Clean context/documents to use
        use_redundancy: Whether to use redundancy defense
        redundancy_count: Number of candidates per executor when using redundancy
        voting_strategy: "exact", "semantic", or "boxed"
        similarity_threshold: Similarity threshold for semantic voting

    Returns:
        List of AgentRun objects (one per executor node)
    """
    runs: List[AgentRun] = []
    context = sample_mal_doc() if attack_type == "mal_doc" and targeted else clean_context or BENIGN_CONTEXT
    attack_payload = prompt_injection_payload if attack_type == "prompt_injection" and targeted else None

    candidates = None

    for idx in range(count):
        suffix = f"{base_name}_{idx+1}" if count > 1 else base_name

        if use_redundancy:
            # Redundancy defense: generate N candidates and vote
            run, run_candidates = run_agent_with_redundancy(
                name=suffix,
                role="executor",
                system_prompt=EXECUTOR_SYSTEM,
                user_prompt=executor_prompt(question, plan_text, context),
                model=model,
                temperature=temperature,
                attack_payload=attack_payload,
                redundancy_count=redundancy_count,
                voting_strategy=voting_strategy,
                similarity_threshold=similarity_threshold,
            )
            candidates = run_candidates
        
        else:
            # No defense: normal execution
            run = run_agent(
                name=suffix,
                role="executor",
                system_prompt=EXECUTOR_SYSTEM,
                user_prompt=executor_prompt(question, plan_text, context),
                model=model,
                temperature=temperature,
                attack_payload=attack_payload,
            )
        runs.append(run)

    return runs


def run_chain_topology_defense(
    question: str,
    model: str,
    temperature: float,
    attack_type: str,
    num_mal: int,
    defense: str,
    context_text: str,
    redundancy_count: int = 3,
    voting_strategy: str = "semantic",
    similarity_threshold: float = 0.7,
) -> MASResult:
    """
    Chain topology with defense: planner -> refiner -> executor -> answerer

    Redundancy defense strategy:
        - Each agent node (planner, refiner, executor) generates N candidates
        - Majority voting selects the final output at each node
        - Only the voted output is passed to the next agent

    This prevents attacked outputs from propagating through the chain.

    Args:
        question: The question to answer
        model: LLM model name
        temperature: Sampling temperature
        attack_type: "prompt_injection", "mal_doc", or "none"
        num_mal: Number of agents to attack
        defense: "redundancy", "consensus", or "none"
        context_text: Clean context/documents
        redundancy_count: Number of candidates per agent (default: 3)
        voting_strategy: Voting strategy (default: "semantic")
        similarity_threshold: Similarity threshold for semantic voting (default: 0.7)

    Returns:
        MASResult with final answer and logs
    """
    logs: List[AgentRun] = []
    pool = ["executor"] if attack_type == "mal_doc" else ["planner", "refiner", "executor"]
    attack_targets = choose_targets(pool, num_mal)

    planner_attack = sample_prompt_injection() if "planner" in attack_targets and attack_type == "prompt_injection" else None
    refiner_attack = sample_prompt_injection() if "refiner" in attack_targets and attack_type == "prompt_injection" else None
    executor_prompt_attack = sample_prompt_injection() if "executor" in attack_targets and attack_type == "prompt_injection" else None
    executor_targeted = "executor" in attack_targets

    use_redundancy = (defense == "redundancy")
    
    # Planner with optional redundancy
    if use_redundancy:
        planner_run = run_agent_with_redundancy(
            name="planner",
            role="planner",
            system_prompt=PLANNER_SYSTEM,
            user_prompt=planner_prompt(question),
            model=model,
            temperature=temperature,
            attack_payload=planner_attack,
            redundancy_count=redundancy_count,
            voting_strategy=voting_strategy,
            similarity_threshold=similarity_threshold,
        )
        
    else:
        planner_run = run_agent(
            name="planner",
            role="planner",
            system_prompt=PLANNER_SYSTEM,
            user_prompt=planner_prompt(question),
            model=model,
            temperature=temperature,
            attack_payload=planner_attack,
        )
    logs.append(planner_run)
    
    # Refiner with optional redundancy
    if use_redundancy:
        refiner_run = run_agent_with_redundancy(
            name="refiner",
            role="refiner",
            system_prompt=REFINER_SYSTEM,
            user_prompt=refiner_prompt(question, planner_run.content),
            model=model,
            temperature=temperature,
            attack_payload=refiner_attack,
            redundancy_count=redundancy_count,
            voting_strategy=voting_strategy,
            similarity_threshold=similarity_threshold,
        )
        
    else:
        refiner_run = run_agent(
            name="refiner",
            role="refiner",
            system_prompt=REFINER_SYSTEM,
            user_prompt=refiner_prompt(question, planner_run.content),
            model=model,
            temperature=temperature,
            attack_payload=refiner_attack,
        )
    logs.append(refiner_run)

    # Executor with optional redundancy
    # Note: Always use 1 executor node; redundancy happens inside the node
    exec_count = 1
    executor_runs = executor_bundle_defense(
        base_name="executor",
        count=exec_count,
        question=question,
        plan_text=refiner_run.content,
        model=model,
        temperature=temperature,
        attack_type=attack_type,
        targeted=executor_targeted,
        prompt_injection_payload=executor_prompt_attack,
        clean_context=context_text,
        use_redundancy=use_redundancy,
        redundancy_count=redundancy_count,
        voting_strategy=voting_strategy,
        similarity_threshold=similarity_threshold,
    )
    logs.extend(executor_runs)

    # Final answerer (no redundancy applied here)
    upstream = [planner_run.content, refiner_run.content] + [e.content for e in executor_runs]
    final_run = run_agent(
        name="answerer",
        role="answerer",
        system_prompt=ANSWER_SYSTEM_BASE,
        user_prompt=answer_prompt(
            question=question,
            upstream_messages=upstream,
            defense_mode=defense,
            attack_summary=attack_summary_text(attack_type, attack_targets),
        ),
        model=model,
        temperature=temperature,
    )
    logs.append(final_run)

    return MASResult(
        final_answer=final_run.content,
        logs=logs,
        attack_targets=attack_targets,
        attack_type=attack_type,
        topology="chain",
        defense=defense,
    )


def run_tree_topology_defense(
    question: str,
    model: str,
    temperature: float,
    attack_type: str,
    num_mal: int,
    defense: str,
    context_text: str,
    redundancy_count: int = 3,
    voting_strategy: str = "semantic",
    similarity_threshold: float = 0.7,
) -> MASResult:
    """
    Tree topology with defense: planner -> [exec_left, exec_right] -> answerer

    Redundancy defense strategy:
        - Planner generates N candidates and votes
        - Each executor (left/right) generates N candidates and votes
        - Only voted outputs are passed to answerer
    """
    logs: List[AgentRun] = []
    pool = ["exec_left", "exec_right"] if attack_type == "mal_doc" else ["planner", "exec_left", "exec_right"]
    attack_targets = choose_targets(pool, num_mal)

    planner_attack = sample_prompt_injection() if "planner" in attack_targets and attack_type == "prompt_injection" else None
    left_attack = sample_prompt_injection() if "exec_left" in attack_targets and attack_type == "prompt_injection" else None
    right_attack = sample_prompt_injection() if "exec_right" in attack_targets and attack_type == "prompt_injection" else None

    use_redundancy = (defense == "redundancy")

    # Planner with optional redundancy
    if use_redundancy:
        planner_run = run_agent_with_redundancy(
            name="planner",
            role="planner",
            system_prompt=PLANNER_SYSTEM,
            user_prompt=planner_prompt(question),
            model=model,
            temperature=temperature,
            attack_payload=planner_attack,
            redundancy_count=redundancy_count,
            voting_strategy=voting_strategy,
            similarity_threshold=similarity_threshold,
        )
    else:
        planner_run = run_agent(
            name="planner",
            role="planner",
            system_prompt=PLANNER_SYSTEM,
            user_prompt=planner_prompt(question),
            model=model,
            temperature=temperature,
            attack_payload=planner_attack,
        )
    logs.append(planner_run)

    # Executors with optional redundancy (1 node each, not multiplied)
    exec_count = 1
    left_runs = executor_bundle_defense(
        base_name="exec_left",
        count=exec_count,
        question=question,
        plan_text=planner_run.content,
        model=model,
        temperature=temperature,
        attack_type=attack_type,
        targeted="exec_left" in attack_targets,
        prompt_injection_payload=left_attack,
        clean_context=context_text,
        use_redundancy=use_redundancy,
        redundancy_count=redundancy_count,
        voting_strategy=voting_strategy,
        similarity_threshold=similarity_threshold,
    )
    right_runs = executor_bundle_defense(
        base_name="exec_right",
        count=exec_count,
        question=question,
        plan_text=planner_run.content,
        model=model,
        temperature=temperature,
        attack_type=attack_type,
        targeted="exec_right" in attack_targets,
        prompt_injection_payload=right_attack,
        clean_context=context_text,
        use_redundancy=use_redundancy,
        redundancy_count=redundancy_count,
        voting_strategy=voting_strategy,
        similarity_threshold=similarity_threshold,
    )
    logs.extend(left_runs + right_runs)

    # Final answerer
    upstream = [planner_run.content] + [r.content for r in left_runs + right_runs]
    final_run = run_agent(
        name="answerer",
        role="answerer",
        system_prompt=ANSWER_SYSTEM_BASE,
        user_prompt=answer_prompt(
            question=question,
            upstream_messages=upstream,
            defense_mode=defense,
            attack_summary=attack_summary_text(attack_type, attack_targets),
        ),
        model=model,
        temperature=temperature,
    )
    logs.append(final_run)

    return MASResult(
        final_answer=final_run.content,
        logs=logs,
        attack_targets=attack_targets,
        attack_type=attack_type,
        topology="tree",
        defense=defense,
    )


def run_star_topology_defense(
    question: str,
    model: str,
    temperature: float,
    attack_type: str,
    num_mal: int,
    defense: str,
    context_text: str,
    redundancy_count: int = 3,
    voting_strategy: str = "semantic",
    similarity_threshold: float = 0.7,
) -> MASResult:
    """
    Star topology with defense: [exec_a, exec_b, exec_c] -> answerer

    Redundancy defense strategy:
        - Each executor generates N candidates and votes
        - Only voted outputs are passed to answerer
    """
    logs: List[AgentRun] = []
    attack_targets = choose_targets(["exec_a", "exec_b", "exec_c"], num_mal)
    attack_map: Dict[str, Optional[str]] = {
        name: sample_prompt_injection() if name in attack_targets and attack_type == "prompt_injection" else None
        for name in ["exec_a", "exec_b", "exec_c"]
    }

    use_redundancy = (defense == "redundancy")
    exec_count = 1

    all_runs: List[AgentRun] = []
    for base_name in ["exec_a", "exec_b", "exec_c"]:
        runs = executor_bundle_defense(
            base_name=base_name,
            count=exec_count,
            question=question,
            plan_text="Independent star executor: focus on complementary angles.",
            model=model,
            temperature=temperature,
            attack_type=attack_type,
            targeted=base_name in attack_targets,
            prompt_injection_payload=attack_map[base_name],
            clean_context=context_text,
            use_redundancy=use_redundancy,
            redundancy_count=redundancy_count,
            voting_strategy=voting_strategy,
            similarity_threshold=similarity_threshold,
        )
        all_runs.extend(runs)
    logs.extend(all_runs)

    # Final answerer
    upstream = [r.content for r in all_runs]
    final_run = run_agent(
        name="answerer",
        role="answerer",
        system_prompt=ANSWER_SYSTEM_BASE,
        user_prompt=answer_prompt(
            question=question,
            upstream_messages=upstream,
            defense_mode=defense,
            attack_summary=attack_summary_text(attack_type, attack_targets),
        ),
        model=model,
        temperature=temperature,
    )
    logs.append(final_run)

    return MASResult(
        final_answer=final_run.content,
        logs=logs,
        attack_targets=attack_targets,
        attack_type=attack_type,
        topology="star",
        defense=defense,
    )


def run_fcg_topology_defense(
    question: str,
    model: str,
    temperature: float,
    attack_type: str,
    num_mal: int,
    defense: str,
    context_text: str,
    redundancy_count: int = 3,
    voting_strategy: str = "semantic",
    similarity_threshold: float = 0.7,
) -> MASResult:
    """
    Fully connected graph topology with defense: peer-to-peer messaging

    Redundancy defense strategy:
        - Each peer message generates N candidates and votes
        - Only voted outputs are sent to other peers
    """
    logs: List[AgentRun] = []
    peer_names = ["peer1", "peer2", "peer3"]
    attack_targets = choose_targets(peer_names, num_mal)
    attack_map: Dict[str, Optional[str]] = {
        name: sample_prompt_injection() if name in attack_targets and attack_type == "prompt_injection" else None
        for name in peer_names
    }
    mal_doc_map: Dict[str, str] = {
        name: sample_mal_doc() if name in attack_targets and attack_type == "mal_doc" else ""
        for name in peer_names
    }

    use_redundancy = (defense == "redundancy")

    def peer_exchange(sender: str, receiver: str, prior_note: str) -> AgentRun:
        """Execute a single peer-to-peer message exchange with optional redundancy."""
        doc_note = mal_doc_map[sender]
        base_context = "" if doc_note else context_text
        prompt = peer_prompt(receiver, question, prior_note, base_context=base_context)
        if doc_note:
            prompt = f"{prompt}\n\nDocument you must rely on:\n{doc_note}"

        if use_redundancy:
            return run_agent_with_redundancy(
                name=f"{sender}_to_{receiver}",
                role="peer",
                system_prompt=PEER_SYSTEM,
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                attack_payload=attack_map[sender] if attack_type == "prompt_injection" else None,
                redundancy_count=redundancy_count,
                voting_strategy=voting_strategy,
                similarity_threshold=similarity_threshold,
            )
        else:
            return run_agent(
                name=f"{sender}_to_{receiver}",
                role="peer",
                system_prompt=PEER_SYSTEM,
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                attack_payload=attack_map[sender] if attack_type == "prompt_injection" else None,
            )

    # Peer-to-peer message exchanges: 1->2, 2->1, 2->3, 3->2, 3->1, 1->3 (6 chats)
    chat12 = peer_exchange("peer1", "peer2", "")
    logs.append(chat12)
    chat21 = peer_exchange("peer2", "peer1", chat12.content)
    logs.append(chat21)

    chat23 = peer_exchange("peer2", "peer3", "")
    logs.append(chat23)
    chat32 = peer_exchange("peer3", "peer2", chat23.content)
    logs.append(chat32)

    chat31 = peer_exchange("peer3", "peer1", "")
    logs.append(chat31)
    chat13 = peer_exchange("peer1", "peer3", chat31.content)
    logs.append(chat13)

    # Final answerer
    upstream = [c.content for c in logs]
    final_run = run_agent(
        name="answerer",
        role="answerer",
        system_prompt=ANSWER_SYSTEM_BASE,
        user_prompt=answer_prompt(
            question=question,
            upstream_messages=upstream,
            defense_mode=defense,
            attack_summary=attack_summary_text(attack_type, attack_targets),
        ),
        model=model,
        temperature=temperature,
    )
    logs.append(final_run)

    return MASResult(
        final_answer=final_run.content,
        logs=logs,
        attack_targets=attack_targets,
        attack_type=attack_type,
        topology="fcg",
        defense=defense,
    )


def run_trial_defense(
    topology: str,
    question: str,
    model: str,
    temperature: float,
    attack: str,
    num_mal_clients: int,
    defense: str,
    context_text: str,
    seed: Optional[int] = None,
    redundancy_count: int = 3,
    voting_strategy: str = "semantic",
    similarity_threshold: float = 0.7,
    consensus_runs: int = 3,
    consensus_strategy: str = "boxed",
) -> MASResult:
    """
    Run a single trial with defense mechanisms enabled.

    This function routes to the appropriate defense-enabled topology function.
    For attack-only simulations (no defense), use run_trial() from mas.py instead.

    Defense modes:
        - "redundancy": Generate N candidates at each node and use majority voting
        - "consensus": Detect conflicting evidence and refuse if tampering detected
        - "none": No defense (equivalent to mas.py but with defense infrastructure)

    Args:
        topology: "chain", "tree", "star", or "fcg"
        question: The question to answer
        model: LLM model name
        temperature: Sampling temperature
        attack: Attack type ("prompt_injection", "mal_doc", or "none")
        num_mal_clients: Number of agents to attack
        defense: Defense mode ("redundancy", "consensus", or "none")
        context_text: Clean context/documents
        seed: Random seed for reproducibility
        redundancy_count: Number of candidates per agent (for redundancy defense)
        voting_strategy: Voting strategy ("exact", "semantic", or "boxed")
        similarity_threshold: Similarity threshold for semantic voting (0.0-1.0)
        consensus_runs: Number of full pipeline runs when using consensus defense
        consensus_strategy: Voting strategy for consensus aggregation ("boxed", "semantic", "exact")

    Returns:
        MASResult with final answer and execution logs
    """
    load_env_vars()
    if seed is not None:
        random.seed(seed)

    norm_attack = normalize_attack(attack)
    norm_defense = normalize_defense(defense)
    topo = topology.lower()

    kwargs = {
        "question": question,
        "model": model,
        "temperature": temperature,
        "attack_type": norm_attack,
        "num_mal": num_mal_clients,
        "defense": norm_defense,
        "context_text": context_text,
        "redundancy_count": redundancy_count,
        "voting_strategy": voting_strategy,
        "similarity_threshold": similarity_threshold,
    }

    def run_once(seed_override: Optional[int] = None) -> MASResult:
        if seed_override is not None:
            random.seed(seed_override)

        if topo == "chain":
            return run_chain_topology_defense(**kwargs)
        if topo == "tree":
            return run_tree_topology_defense(**kwargs)
        if topo == "star":
            return run_star_topology_defense(**kwargs)
        if topo in {"graph", "fcg", "fully_connected", "fully-connected"}:
            return run_fcg_topology_defense(**kwargs)

        raise ValueError(f"Unknown topology: {topology}")

    if norm_defense == "consensus":
        runs: List[MASResult] = []
        for idx in range(consensus_runs):
            run_seed = seed + idx if seed is not None else None
            runs.append(run_once(run_seed))

        if not runs:
            return run_once()

        answers = [r.final_answer for r in runs]
        winning_answer, _ = majority_vote(answers, strategy=consensus_strategy)
        try:
            winner_idx = answers.index(winning_answer)
        except ValueError:
            winner_idx = 0
        winner = runs[winner_idx]
        return MASResult(
            final_answer=winning_answer,
            logs=winner.logs,
            attack_targets=winner.attack_targets,
            attack_type=winner.attack_type,
            topology=winner.topology,
            defense=winner.defense,
        )

    return run_once()
