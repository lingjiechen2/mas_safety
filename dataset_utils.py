import random
from typing import Dict, List, Optional, Tuple


def _try_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required. Install with `pip install datasets`."
        ) from exc
    return load_dataset


def truncate_text(text: str, max_chars: int = 2000) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


HOTPOT_CONFIG = "distractor"


def _hotpotqa_to_dict(example) -> Dict[str, str]:
    question = example["question"]
    answer = example["answer"]
    context_blocks = []

    # HotpotQA context is a dict with 'title' and 'sentences' keys
    ctx = example["context"]
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        titles = ctx["title"]
        sentences = ctx["sentences"]
        # Iterate through parallel lists of titles and sentences
        for title, sents in zip(titles, sentences):
            if isinstance(sents, (list, tuple)):
                sent_text = " ".join(sents)
            else:
                sent_text = str(sents)
            context_blocks.append(f"{title}: {sent_text}")
    else:
        # Fallback for other formats (if any)
        for entry in ctx:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                title = entry[0]
                sents = entry[1] if len(entry) == 2 else entry[1:]
                if isinstance(sents, (list, tuple)):
                    sent_text = " ".join(sents)
                else:
                    sent_text = str(sents)
                context_blocks.append(f"{title}: {sent_text}")
            else:
                context_blocks.append(str(entry))

    context_text = truncate_text("\n".join(context_blocks), max_chars=3500)
    return {
        "question": question,
        "context": context_text,
        "reference": answer,
        "task": "qa",
    }


def load_hotpotqa_sample(index: int, split: str = "validation") -> Dict[str, str]:
    load_dataset = _try_import_datasets()
    ds = load_dataset("hotpot_qa", HOTPOT_CONFIG, split=split)
    example = ds[int(index)]
    return _hotpotqa_to_dict(example)


def _multinews_to_dict(example) -> Dict[str, str]:
    document = truncate_text(example["document"], max_chars=5000)
    summary = example["summary"]
    return {
        "question": "Summarize the following news article in 3-5 sentences.",
        "context": document,
        "reference": summary,
        "task": "summarization",
    }


def load_multinews_sample(index: int, split: str = "validation") -> Dict[str, str]:
    load_dataset = _try_import_datasets()
    ds = load_dataset("alexfabbri/multi_news", split=split)
    example = ds[int(index)]
    return _multinews_to_dict(example)


def pick_dataset(attack: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    # default rule: prompt injection -> multi_news; otherwise hotpotqa
    return "multi_news" if attack == "prompt_injection" else "hotpotqa"


def load_dataset_sample(
    dataset_name: str,
    index: int,
    split: str = "validation",
) -> Dict[str, str]:
    name = dataset_name.lower()
    if name in {"hotpot", "hotpotqa", "hotpot_qa"}:
        return load_hotpotqa_sample(index, split)
    if name in {"multinews", "multi_news", "alexfabbri/multi_news"}:
        return load_multinews_sample(index, split)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset_samples(
    dataset_name: str,
    start: int,
    count: int,
    split: str = "validation",
) -> List[Dict[str, str]]:
    load_dataset = _try_import_datasets()
    name = dataset_name.lower()
    if name in {"hotpot", "hotpotqa", "hotpot_qa"}:
        ds = load_dataset("hotpot_qa", HOTPOT_CONFIG, split=split)
        end = min(len(ds), start + count)
        return [_hotpotqa_to_dict(ds[i]) for i in range(start, end)]
    if name in {"multinews", "multi_news", "alexfabbri/multi_news"}:
        ds = load_dataset("alexfabbri/multi_news", split=split, trust_remote_code=True)
        end = min(len(ds), start + count)
        return [_multinews_to_dict(ds[i]) for i in range(start, end)]
    raise ValueError(f"Unsupported dataset: {dataset_name}")
