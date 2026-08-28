import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# ---------------- Configuration ----------------

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "CMB": {
        "data_path": "benchmark/checked_merged_test_CMB.json",
        "language": "zh",
    },
    "CMExam": {
        "data_path": "benchmark/checked_merged_CMExam_test.json",
        "language": "zh",
    },
    "MedMCQA": {
        "data_path": "benchmark/checked_converted_medmcqa_test.json",
        "language": "en",
    },
    "MedQA": {
        "data_path": "benchmark/checked_converted_medqa_test.json",
        "language": "en",
    },
}

MODEL_GROUPS: Dict[str, List[str]] = {
    "Base": [
        "DeepSeek-R1-8B",
        "Qwen3-8B",
        "DeepSeek-R1-14B",
        "Qwen3-14B",
    ],
    "EPS": [
        "EPS-DeepSeek-R1-8B",
        "EPS-Qwen3-8B",
        "EPS-DeepSeek-R1-14B",
        "EPS-Qwen3-14B",
    ],
    "Flagship": [
        "ChatGPT-5",
        "DeepSeek-R1",
        "Gemini-2.5-Flash",
        "Grok-4-Fast",
    ],
}

LOCAL_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "DeepSeek-R1-8B": {
        "base_model": "PATH_OR_HF_ID_TO_DEEPSEEK_R1_8B",
        "adapter_paths": [],
    },
    "Qwen3-8B": {
        "base_model": "PATH_OR_HF_ID_TO_QWEN3_8B",
        "adapter_paths": [],
    },
    "DeepSeek-R1-14B": {
        "base_model": "PATH_OR_HF_ID_TO_DEEPSEEK_R1_14B",
        "adapter_paths": [],
    },
    "Qwen3-14B": {
        "base_model": "PATH_OR_HF_ID_TO_QWEN3_14B",
        "adapter_paths": [],
    },
    "EPS-DeepSeek-R1-8B": {
        "base_model": "PATH_OR_HF_ID_TO_DEEPSEEK_R1_8B",
        "adapter_paths": [
            "PATH_TO_SFT_LORA_FOR_EPS_DEEPSEEK_R1_8B",
            "PATH_TO_KTO_LORA_FOR_EPS_DEEPSEEK_R1_8B",
        ],
    },
    "EPS-Qwen3-8B": {
        "base_model": "PATH_OR_HF_ID_TO_QWEN3_8B",
        "adapter_paths": [
            "PATH_TO_SFT_LORA_FOR_EPS_QWEN3_8B",
            "PATH_TO_KTO_LORA_FOR_EPS_QWEN3_8B",
        ],
    },
    "EPS-DeepSeek-R1-14B": {
        "base_model": "PATH_OR_HF_ID_TO_DEEPSEEK_R1_14B",
        "adapter_paths": [
            "PATH_TO_SFT_LORA_FOR_EPS_DEEPSEEK_R1_14B",
            "PATH_TO_KTO_LORA_FOR_EPS_DEEPSEEK_R1_14B",
        ],
    },
    "EPS-Qwen3-14B": {
        "base_model": "PATH_OR_HF_ID_TO_QWEN3_14B",
        "adapter_paths": [
            "PATH_TO_SFT_LORA_FOR_EPS_QWEN3_14B",
            "PATH_TO_KTO_LORA_FOR_EPS_QWEN3_14B",
        ],
    },
}

FLAGSHIP_SPECS: Dict[str, Dict[str, Any]] = {
    "ChatGPT-5": {
        "base_url": "https://api.openai.com",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5",
        "supports_seed": True,
    },
    "DeepSeek-R1": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-reasoner",
        "supports_seed": False,
    },
    "Gemini-2.5-Flash": {
        "base_url": "OPENAI_COMPATIBLE_BASE_URL_FOR_GEMINI",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "supports_seed": False,
    },
    "Grok-4-Fast": {
        "base_url": "https://api.x.ai",
        "api_key_env": "XAI_API_KEY",
        "model": "grok-4-fast",
        "supports_seed": False,
    },
}

DEFAULT_GEN_CONFIG = GenerationConfig(
    temperature=0.35,
    top_k=40,
    top_p=0.8,
    repetition_penalty=1.15,
    max_new_tokens=512,
    do_sample=True,
)

PROMPT_FMT_ZH = (
    "你是一位专业的医学专家。请回答以下选择题，逐步分析选项后，在最后一行以 '答案：' 后跟单个选项字母（A、B、C、D 或 E）输出。\n\n"
    "问题：\n{question}\n\n"
    "选项：\n{opt_text}\n\n"
    "答案："
)

PROMPT_FMT_EN = (
    "You are a medical expert. Answer the following multiple-choice question. "
    "After reasoning, output the final answer on the last line as 'Answer:' followed by a single letter (A, B, C, D, or E).\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{opt_text}\n\n"
    "Answer:"
)

RESULTS_BASE_DIR = "data/results"
BENCHMARK_CSV_PATH = "data/benchmark_results/benchmark_accuracy.csv"
GROUP_DISPLAY_NAMES: Dict[str, str] = {
    "Base": "Base model",
    "EPS": "EPS",
    "Flagship": "Mainstream LLMs",
}
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "DeepSeek-R1-8B": "DeepSeek-R1-8B",
    "Qwen3-8B": "Qwen3-8B",
    "DeepSeek-R1-14B": "DeepSeek-R1-14B",
    "Qwen3-14B": "Qwen3-14B",
    "EPS-DeepSeek-R1-8B": "DeepSeek-R1-8B",
    "EPS-Qwen3-8B": "Qwen3-8B",
    "EPS-DeepSeek-R1-14B": "DeepSeek-R1-14B",
    "EPS-Qwen3-14B": "Qwen3-14B",
    "ChatGPT-5": "ChatGPT-5",
    "DeepSeek-R1": "DeepSeek-R1",
    "Gemini-2.5-Flash": "Gemini 2.5 Flash",
    "Grok-4-Fast": "Grok 4 Fast",
}


# ---------------- Utilities ----------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_options(option_dict: Dict[str, str]) -> str:
    items = []
    for k, v in option_dict.items():
        items.append(f"{k}. {v}")
    return "\n".join(items)


def parse_answer(text: str) -> str:
    m = re.search(r"(?:答案|Answer)\s*[:：]\s*([A-E])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    for ch in reversed(text):
        if ch in "ABCDE":
            return ch
    return ""


def t_critical_975(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
    }
    if df in table:
        return table[df]
    if df > 30:
        return 2.042
    return 2.776


def mean_ci(values: List[float]) -> Tuple[float, float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(sum(values) / n)
    if n == 1:
        return mean, mean, mean, 0.0

    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = float(math.sqrt(var))
    se = std / math.sqrt(n)
    tcrit = t_critical_975(n - 1)
    half = tcrit * se
    return mean, mean - half, mean + half, std


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return obj


def result_dir_for(category: str, model_name: str, benchmark: str) -> Path:
    root = project_root()
    return root / RESULTS_BASE_DIR / category / model_name / benchmark


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_benchmark_csv() -> Path:
    root = project_root()
    out_path = root / BENCHMARK_CSV_PATH
    ensure_dir(out_path.parent)

    rows: List[Dict[str, Union[str, int, float]]] = []
    for category, models in MODEL_GROUPS.items():
        display_group = GROUP_DISPLAY_NAMES.get(category, category)
        for model_name in models:
            display_model = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            for benchmark in BENCHMARKS:
                summary_path = result_dir_for(category, model_name, benchmark) / "result.json"
                if not summary_path.exists():
                    continue
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)

                runs = summary.get("runs", [])
                n_questions = 0
                if runs:
                    n_questions = int(runs[0].get("total", 0))

                ci95 = summary.get("ci95", [float("nan"), float("nan")])
                rows.append(
                    {
                        "group": display_group,
                        "model": display_model,
                        "benchmark": benchmark,
                        "mean_accuracy_pct": round(float(summary.get("average_accuracy", 0.0)) * 100.0, 2),
                        "ci_low_pct": round(float(ci95[0]) * 100.0, 2),
                        "ci_high_pct": round(float(ci95[1]) * 100.0, 2),
                        "n_questions": n_questions,
                    }
                )

    group_order = {name: idx for idx, name in enumerate(["Base model", "EPS", "Mainstream LLMs"])}
    model_order = {
        "Base model": {name: idx for idx, name in enumerate(["DeepSeek-R1-8B", "Qwen3-8B", "DeepSeek-R1-14B", "Qwen3-14B"])},
        "EPS": {name: idx for idx, name in enumerate(["DeepSeek-R1-8B", "Qwen3-8B", "DeepSeek-R1-14B", "Qwen3-14B"])},
        "Mainstream LLMs": {name: idx for idx, name in enumerate(["ChatGPT-5", "DeepSeek-R1", "Gemini 2.5 Flash", "Grok 4 Fast"])},
    }
    benchmark_order = {name: idx for idx, name in enumerate(BENCHMARKS)}
    rows.sort(
        key=lambda row: (
            group_order.get(str(row["group"]), 999),
            model_order.get(str(row["group"]), {}).get(str(row["model"]), 999),
            benchmark_order.get(str(row["benchmark"]), 999),
        )
    )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "model",
                "benchmark",
                "mean_accuracy_pct",
                "ci_low_pct",
                "ci_high_pct",
                "n_questions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


# ---------------- Model runners ----------------

@dataclass
class RunOutput:
    raw_text: str
    pred: str


class BaseRunner:
    def generate_answer(self, prompt: str, seed: int, gen_cfg: GenerationConfig) -> RunOutput:
        raise NotImplementedError

    def close(self) -> None:
        return


class LocalHFRunner(BaseRunner):
    def __init__(self, base_model: str, adapter_paths: List[str]):
        if adapter_paths and PeftModel is None:
            raise RuntimeError("peft is required to load adapters, but it is not available.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        for idx, a_path in enumerate(adapter_paths, start=1):
            peft_model = PeftModel.from_pretrained(
                self.model,
                a_path,
                adapter_name=f"a{idx}",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
            )
            peft_model.set_adapter(f"a{idx}")
            self.model = peft_model.merge_and_unload()

        self.model.eval()

    def generate_answer(self, prompt: str, seed: int, gen_cfg: GenerationConfig) -> RunOutput:
        set_seed(seed)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_cfg)

        prompt_len = inputs["input_ids"].shape[1]
        out_text = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        pred = parse_answer(out_text)
        return RunOutput(raw_text=out_text.strip(), pred=pred)

    def close(self) -> None:
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OpenAICompatibleRunner(BaseRunner):
    def __init__(self, base_url: str, api_key: str, model: str, supports_seed: bool):
        if requests is None:
            raise RuntimeError("requests is required for flagship inference, but it is not available.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.supports_seed = supports_seed

    def _post(self, payload: Dict[str, Any], max_retries: int = 6) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        backoff = 1.0
        last_err: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 20.0)
                    continue
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
        raise RuntimeError(f"Request failed after retries: {last_err}")

    def generate_answer(self, prompt: str, seed: int, gen_cfg: GenerationConfig) -> RunOutput:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(gen_cfg.temperature) if gen_cfg.temperature is not None else 0.0,
            "top_p": float(gen_cfg.top_p) if gen_cfg.top_p is not None else 1.0,
            "max_tokens": int(gen_cfg.max_new_tokens) if gen_cfg.max_new_tokens is not None else 512,
        }
        if self.supports_seed:
            payload["seed"] = int(seed)

        data = self._post(payload)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        pred = parse_answer(content or "")
        return RunOutput(raw_text=(content or "").strip(), pred=pred)


def build_runner(model_name: str) -> Tuple[str, BaseRunner, Dict[str, Any]]:
    if model_name in LOCAL_MODEL_SPECS:
        spec = LOCAL_MODEL_SPECS[model_name]
        base_model = str(spec["base_model"])
        adapter_paths = list(spec.get("adapter_paths", []))
        runner = LocalHFRunner(base_model=base_model, adapter_paths=adapter_paths)
        meta = {"backend": "local_hf", "base_model": base_model, "adapter_paths": adapter_paths}
        return "local", runner, meta

    if model_name in FLAGSHIP_SPECS:
        spec = FLAGSHIP_SPECS[model_name]
        api_key = os.environ.get(spec["api_key_env"], "")
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {spec['api_key_env']}")
        runner = OpenAICompatibleRunner(
            base_url=str(spec["base_url"]),
            api_key=api_key,
            model=str(spec["model"]),
            supports_seed=bool(spec.get("supports_seed", False)),
        )
        meta = {
            "backend": "openai_compatible",
            "base_url": str(spec["base_url"]),
            "model": str(spec["model"]),
            "api_key_env": str(spec["api_key_env"]),
            "supports_seed": bool(spec.get("supports_seed", False)),
        }
        return "remote", runner, meta

    raise KeyError(f"Unknown model: {model_name}")


# ---------------- Evaluation ----------------

def evaluate_one(
    runner: BaseRunner,
    dataset: List[Dict[str, Any]],
    prompt_fmt: str,
    gen_cfg: GenerationConfig,
    seed: int,
    save_details: bool,
    limit_n: Optional[int],
) -> Tuple[float, int, int, List[Dict[str, Any]]]:
    if limit_n is not None:
        dataset = dataset[: int(limit_n)]

    total = len(dataset)
    correct = 0
    details: List[Dict[str, Any]] = []

    for idx, item in enumerate(tqdm(dataset, ncols=80)):
        question = str(item["question"])
        option_dict = dict(item["option"])
        gt = str(item["answer"]).strip().upper()

        prompt = prompt_fmt.format(question=question, opt_text=format_options(option_dict))
        out = runner.generate_answer(prompt=prompt, seed=seed, gen_cfg=gen_cfg)

        pred = out.pred
        ok = (pred == gt)
        correct += int(ok)

        if save_details:
            details.append(
                {
                    "index": idx + 1,
                    "question": question,
                    "ground_truth": gt,
                    "prediction": pred,
                    "is_correct": bool(ok),
                    "raw_output": out.raw_text,
                }
            )

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total, details


def run_all(
    n_runs: int,
    seeds: List[int],
    overwrite: bool,
    only_missing: bool,
    save_details: bool,
    limit_n: Optional[int],
    gen_cfg: GenerationConfig,
) -> None:
    root = project_root()
    for bname, bspec in BENCHMARKS.items():
        data_path = root / str(bspec["data_path"])
        if not data_path.exists():
            raise FileNotFoundError(f"Missing dataset: {data_path}")

    for category, models in MODEL_GROUPS.items():
        for model_name in models:
            backend_type, runner, runner_meta = build_runner(model_name)
            try:
                for bname, bspec in BENCHMARKS.items():
                    out_dir = result_dir_for(category, model_name, bname)
                    summary_path = out_dir / "result.json"
                    if summary_path.exists() and only_missing:
                        continue
                    if summary_path.exists() and (not overwrite) and (not only_missing):
                        continue

                    data_path = root / str(bspec["data_path"])
                    dataset = load_dataset(data_path)

                    lang = str(bspec.get("language", "en"))
                    prompt_fmt = PROMPT_FMT_ZH if lang == "zh" else PROMPT_FMT_EN

                    run_summaries: List[Dict[str, Any]] = []
                    run_accs: List[float] = []

                    for run_idx in range(1, n_runs + 1):
                        seed = seeds[run_idx - 1] if run_idx - 1 < len(seeds) else (1000 + run_idx)
                        per_run_path = out_dir / f"result_run{run_idx}.json"
                        if per_run_path.exists() and only_missing:
                            with per_run_path.open("r", encoding="utf-8") as f:
                                per_obj = json.load(f)
                            run_accs.append(float(per_obj.get("accuracy", 0.0)))
                            run_summaries.append(
                                {
                                    "run_index": run_idx,
                                    "seed": int(per_obj.get("seed", seed)),
                                    "total": int(per_obj.get("total", 0)),
                                    "correct": int(per_obj.get("correct", 0)),
                                    "accuracy": float(per_obj.get("accuracy", 0.0)),
                                    "file": str(per_run_path),
                                }
                            )
                            continue

                        acc, correct, total, details = evaluate_one(
                            runner=runner,
                            dataset=dataset,
                            prompt_fmt=prompt_fmt,
                            gen_cfg=gen_cfg,
                            seed=seed,
                            save_details=save_details,
                            limit_n=limit_n,
                        )
                        run_accs.append(acc)

                        per_payload: Dict[str, Any] = {
                            "category": category,
                            "model": model_name,
                            "benchmark": bname,
                            "backend": runner_meta.get("backend"),
                            "run_index": run_idx,
                            "seed": seed,
                            "total": total,
                            "correct": correct,
                            "accuracy": acc,
                        }
                        if save_details:
                            per_payload["detail"] = details

                        write_json(per_run_path, per_payload)

                        run_summaries.append(
                            {
                                "run_index": run_idx,
                                "seed": seed,
                                "total": total,
                                "correct": correct,
                                "accuracy": acc,
                                "file": str(per_run_path),
                            }
                        )

                    mean_acc, ci_lo, ci_hi, std = mean_ci(run_accs)
                    summary_payload: Dict[str, Any] = {
                        "category": category,
                        "model": model_name,
                        "benchmark": bname,
                        "backend": runner_meta.get("backend"),
                        "runner_meta": runner_meta,
                        "data_path": str(data_path),
                        "n_runs": n_runs,
                        "runs": run_summaries,
                        "average_accuracy": mean_acc,
                        "std_accuracy": std,
                        "ci95": [ci_lo, ci_hi],
                        "gen_config": {
                            "temperature": gen_cfg.temperature,
                            "top_k": gen_cfg.top_k,
                            "top_p": gen_cfg.top_p,
                            "repetition_penalty": gen_cfg.repetition_penalty,
                            "max_new_tokens": gen_cfg.max_new_tokens,
                            "do_sample": gen_cfg.do_sample,
                        },
                        "prompt_language": lang,
                    }
                    write_json(summary_path, summary_payload)

                    print(
                        f"{category:<8} | {model_name:<22} | {bname:<8} | "
                        f"mean={mean_acc:.4f} | CI=[{ci_lo:.4f},{ci_hi:.4f}] | runs={n_runs} | backend={backend_type}"
                    )
            finally:
                runner.close()

    csv_path = export_benchmark_csv()
    print(f"Saved benchmark CSV: {csv_path}")


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only_missing", action="store_true")
    p.add_argument("--save_details", action="store_true")
    p.add_argument("--limit_n", type=int, default=None)
    p.add_argument("--temperature", type=float, default=float(DEFAULT_GEN_CONFIG.temperature))
    p.add_argument("--top_k", type=int, default=int(DEFAULT_GEN_CONFIG.top_k))
    p.add_argument("--top_p", type=float, default=float(DEFAULT_GEN_CONFIG.top_p))
    p.add_argument("--repetition_penalty", type=float, default=float(DEFAULT_GEN_CONFIG.repetition_penalty))
    p.add_argument("--max_new_tokens", type=int, default=int(DEFAULT_GEN_CONFIG.max_new_tokens))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    gen_cfg = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
    )

    run_all(
        n_runs=int(args.n_runs),
        seeds=seeds,
        overwrite=bool(args.overwrite),
        only_missing=bool(args.only_missing),
        save_details=bool(args.save_details),
        limit_n=args.limit_n,
        gen_cfg=gen_cfg,
    )


if __name__ == "__main__":
    main()
