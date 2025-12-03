import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from dataset_utils import load_dataset_samples, pick_dataset
from evaluation import evaluate_output
from mas import MASResult, run_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAS safety simulation.")
    parser.add_argument("--mas", dest="topology", required=True, help="chain | tree | star | fcg")
    parser.add_argument("--attack", dest="attack", default="none", help="none | prompt_injection | mal_doc")
    parser.add_argument("--num_mal_clients", type=int, default=0, help="number of malicious clients (exclude final answerer)")
    parser.add_argument("--defense", dest="defense", default="none", help="none | redundancy | consensus")
    parser.add_argument("--dataset", default=None, help="hotpotqa | multi_news | auto (default)")
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples to run from the dataset starting at index 0")
    parser.add_argument("--question", default=None, help="Override question/task (if not set, pulled from dataset)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name used for all agents")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose_logs", action="store_true", help="print full per-agent logs for each sample")
    parser.add_argument("--consensus_runs", type=int, default=3, help="repeat full pipeline K times when using consensus defense (default: 3)")
    parser.add_argument("--consensus_strategy", default="boxed", help="majority vote strategy for consensus: boxed | semantic | exact (default: boxed)")
    parser.add_argument("--base_url", default="https://api.soruxgpt.com/v1",)
    parser.add_argument("--api_key", default="sk-MMP0KZBQk3pRFeOSYDviaraUzoF0LtIkyhcbTJBtUe7Wp3nM",)
    parser.add_argument("--log_file", default=None, help="optional path to append detailed logs (hyperparams, question, agent I/O)")
    parser.add_argument("--use_vllm_model", action="store_true", help="use local vLLM server instead of remote API")
    parser.add_argument("--vllm_host", default="localhost", help="vLLM server host (default: localhost)")
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port (default: 8000)")

    return parser.parse_args()


def print_logs(result: MASResult) -> None:
    for entry in result.logs:
        attack_tag = "ATTACKED" if entry.attack_note else "clean"
        print(f"[{entry.role}:{entry.name}] ({attack_tag})")
        print(f"Prompt used:\n{entry.user_prompt}\n")
        print(f"Model output:\n{entry.content}\n{'='*60}")


def append_log(
    log_path: str,
    run_meta: Dict[str, Any],
    sample_id: int,
    task: str,
    question_text: str,
    context: str,
    result: MASResult,
    eval_result,
    reference: str,
) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        separator = "#" * 90
        f.write("\n")
        f.write(f"{separator}\n")
        f.write(f"# SAMPLE {sample_id} | TASK={task}\n")
        f.write(f"# Sections: HYPERPARAMS, QUESTION, CONTEXT, FINAL, EVAL, AGENTS\n")
        f.write(f"{separator}\n")

        f.write("---- HYPERPARAMS ----\n")
        for k, v in run_meta.items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- QUESTION ----\n")
        f.write(question_text + "\n")

        f.write("\n---- CONTEXT ----\n")
        f.write(context + "\n")

        f.write("\n---- FINAL ANSWER ----\n")
        f.write(result.final_answer + "\n")

        f.write("\n---- EVAL ----\n")
        f.write(f"Ground Truth: {reference}\n")
        f.write(f"Parsed boxed: {eval_result.boxed or '<none>'}\n")
        f.write(f"Score ({eval_result.metric}): {eval_result.score:.3f}\n")
        f.write(f"Attack success: {eval_result.attack_success}\n")
        f.write(f"Attack targets: {result.attack_targets}\n")

        f.write("\n---- AGENTS ----\n")
        for entry in result.logs:
            tag = "ATTACKED" if entry.attack_note else "CLEAN"
            f.write(f"[{entry.role}:{entry.name}] {tag}\n")
            f.write("USER PROMPT:\n")
            f.write(entry.user_prompt + "\n")
            f.write("MODEL OUTPUT:\n")
            f.write(entry.content + "\n")
            f.write("-" * 60 + "\n")

        f.write(f"{separator}\n\n")


def main() -> None:
    args = parse_args()

    # If using vLLM, override base_url and api_key
    if args.use_vllm_model:
        args.base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"
        args.api_key = "EMPTY"  # vLLM doesn't require API key
        print(f"Using vLLM server at: {args.base_url}")

    # Reset log file so each run overwrites previous content instead of appending.
    if args.log_file:
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write("# MAS run log\n")

    # Push CLI-provided creds into env so llm_client picks them up.
    os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    dataset_choice = pick_dataset(args.attack, args.dataset)

    # Load batch of samples
    samples = load_dataset_samples(
        dataset_choice,
        start=0,
        count=args.num_samples,
        split="validation",
    )
    if not samples:
        sys.stderr.write("No samples loaded; check num_samples and dataset availability.\n")
        sys.exit(1)

    overall_scores: List[float] = []
    overall_success = 0
    attacked_count = 0
    run_meta: Dict[str, Any] = {
        "topology": args.topology,
        "attack": args.attack,
        "defense": args.defense,
        "num_mal_clients": args.num_mal_clients,
        "dataset": dataset_choice,
        "num_samples": args.num_samples,
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "base_url": args.base_url,
        "consensus_runs": args.consensus_runs if args.defense == "consensus" else None,
        "consensus_strategy": args.consensus_strategy if args.defense == "consensus" else None,
    }

    for sample_id, sample in enumerate(samples):
        base_question = args.question if args.question else sample["question"]
        context = sample["context"]
        reference = sample["reference"]
        task = sample["task"]

        if task == "qa":
            question_text = f"Answer the QA question using the provided context. Question: {base_question}"
        else:
            question_text = f"{base_question}\n\nArticle:\n{context}"

        try:
            result = run_trial(
                topology=args.topology,
                question=question_text,
                model=args.model,
                temperature=args.temperature,
                attack=args.attack,
                num_mal_clients=args.num_mal_clients,
                defense=args.defense,
                context_text=context,
                seed=args.seed,
                consensus_runs=args.consensus_runs,
                consensus_strategy=args.consensus_strategy,
            )
        except Exception as exc:
            sys.stderr.write(f"Run failed on sample {sample_id}: {exc}\n")
            continue

        eval_result = evaluate_output(
            task=task,
            final_answer=result.final_answer,
            reference=reference,
            attack_type=result.attack_type,
            attack_present=len(result.attack_targets) > 0,
        )

        if result.attack_type != "none" and len(result.attack_targets) > 0:
            attacked_count += 1
        if eval_result.attack_success:
            overall_success += 1
        overall_scores.append(eval_result.score)

        print(f"\n=== Sample {sample_id} ({task}) ===")
        print(f"Topology: {result.topology} | Attack: {result.attack_type} targets={result.attack_targets} | Defense: {result.defense}")
        print(f"Dataset: {dataset_choice} | Task: {task}")
        print(f"Ground Truth: {reference}")
        print(f"Parsed boxed answer: {eval_result.boxed or '<none>'}")
        print(f"Score ({eval_result.metric}): {eval_result.score:.3f}")
        print(f"Attack success: {eval_result.attack_success}")
        if args.verbose_logs or args.num_samples == 1:
            print(f"Final answer:\n{result.final_answer}\n")
            print_logs(result)
        if args.log_file:
            append_log(
                log_path=args.log_file,
                run_meta=run_meta,
                sample_id=sample_id,
                task=task,
                question_text=question_text,
                context=context,
                result=result,
                eval_result=eval_result,
                reference=reference,
            )

    if overall_scores:
        avg_score = sum(overall_scores) / len(overall_scores)
        success_rate = (overall_success / attacked_count) if attacked_count else 0.0
        print("\n=== Aggregate ===")
        print(f"Samples run: {len(overall_scores)}")
        print(f"Average score: {avg_score:.3f}")
        print(f"Attack successes: {overall_success} / {attacked_count} attacked samples (rate {success_rate:.3f})")

        # Write aggregate stats to log file
        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("#" * 90 + "\n")
                f.write("# AGGREGATE STATISTICS\n")
                f.write("#" * 90 + "\n")
                f.write(f"Total samples run: {len(overall_scores)}\n")
                f.write(f"Average score: {avg_score:.3f}\n")
                f.write(f"Attack successes: {overall_success} / {attacked_count} attacked samples\n")
                f.write(f"Attack success rate: {success_rate:.3f}\n")
                f.write("#" * 90 + "\n")
    else:
        print("No successful runs to report.")


if __name__ == "__main__":
    main()
