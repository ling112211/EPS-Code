# Exercise-Prescription-System

This repository provides the code to reproduce the analyses, tables, and figures for the paper: **LLM-human collaboration for personalized exercise prescription in digital chronic disease management: a randomized controlled trial** (under review).

The trial was prospectively registered at the [Chinese Clinical Trial Registry (ChiCTR2600118939)](https://www.chictr.org.cn/hvshowprojectEN.html?id=295346&v=1.0).

---

## Overview

This repository contains the statistical analysis and visualization code for the Exercise Prescription System (EPS), an expert-aligned large language model (LLM)–based system designed to produce safe, personalized, and scalable exercise feedback for digital chronic disease management.

EPS was developed and evaluated through three stages:

1. **Benchmark evaluation**: Four domain-specific medical question-answering benchmarks (CMB, CMExam, MedMCQA, MedQA) assessing the medical knowledge of EPS variants and baseline models.
2. **Expert pilot study**: An ablation-style evaluation by 25 professional health managers rating exercise-feedback outputs across seven dimensions (consensus, correctness, completeness, unbiasedness, clarity, empathy, and actionability).
3. **Randomized controlled trials**: A single-blind RCT in two cohorts — 1,580 randomized weight-loss participants and 48 randomized glycemic-control participants — comparing EPS–human collaboration (EPS-generated feedback reviewed by health managers) against human coaching alone. The complete-case efficacy analyses included 1,444 and 40 participants, respectively.

Key results:

- EPS improved medical benchmark performance by 20–30 percentage points over base models across all four benchmarks, with Qwen3-14B (EPS) achieving 86.75% on CMB, 90.66% on CMExam, 89.74% on MedMCQA, and 86.30% on MedQA.
- In the weight-loss RCT, the EPS–human arm achieved significantly greater mean weight loss (1.40 kg vs 1.20 kg; *P* = 0.0004) and a higher proportion of participants achieving ≥2% weight loss (62.94% vs 54.27%; *P* = 0.0008).
- In the exploratory glycemic-control RCT, the complete-case estimate favoured EPS–human, although the manager-clustered CR2 confidence interval included zero.

## Repository Structure

```
Exercise-Prescription-System/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README_data.md                         # Data dictionary and access instructions
│   ├── benchmark_results/
│   │   └── benchmark_accuracy.csv             # Pre-computed benchmark accuracy (REAL DATA - reproduces Fig. 5a,c)
│   ├── expert_pilot/
│   │   ├── base_model.csv                     # Expert ratings for base model (REAL DATA - reproduces Fig. 5b,d)
│   │   ├── eps_without_d2.csv                 # Expert ratings for base model + D1 (REAL DATA - reproduces Fig. 5b,d)
│   │   └── eps.csv                            # Expert ratings for full EPS (REAL DATA - reproduces Fig. 5b,d)
│   └── example/                               # EXAMPLE DATA ONLY — for code verification, not paper results
│       ├── checkin/
│       │   ├── weight_loss/
│       │   │   ├── human_arm.xlsx
│       │   │   ├── eps_arm.xlsx
│       │   │   ├── human_chat_history.xlsx
│       │   │   └── eps_chat_history.xlsx
│       │   ├── glycemic/
│       │   │   ├── human_arm.xlsx
│       │   │   ├── eps_arm.xlsx
│       │   │   ├── human_chat_history.xlsx
│       │   │   └── eps_chat_history.xlsx
│       │   └── synthetic_manifest.json
│       ├── weight_loss/
│       │   ├── human_arm.xlsx                 # Anonymised example data (does NOT reproduce paper Tables/Figs)
│       │   └── eps_arm.xlsx
│       ├── glycemic/
│       │   ├── human_arm.xlsx
│       │   └── eps_arm.xlsx
│       └── questionnaire/
│           ├── human_responses.xlsx
│           └── eps_responses.xlsx
├── D1/                                        # 18 JSON datasets with 85,469 entries
├── benchmark/
│   ├── evaluate_benchmark.py                  # Benchmark model inference and 95% CI computation
│   ├── plot_benchmark.py                      # Benchmark accuracy bar chart (Fig. 5a) from pre-computed CSV
│   └── plot_eps_accuracy_gain.py              # EPS-over-base accuracy-gain heatmap (Fig. 5c)
├── expert_pilot/
│   ├── plot_expert_evaluation.py              # Expert summaries, tests, and grouped bar chart (Fig. 5b)
│   └── plot_ablation_gain_heatmap.py          # Ablation gain heatmap (Fig. 5d)
├── clinical_trial/
│   ├── baseline_characteristics.py            # Baseline demographics tables (Table 1)
│   ├── checkin_analysis/
│   │   ├── generate_synthetic_checkin_data.py
│   │   ├── build_checkin_dataset.py
│   │   ├── feedback_mediation.py
│   │   └── enhanced_feedback_mediation.py
│   ├── weight_loss_analysis.py                # Weight-loss outcomes and cumulative response (Fig. 3a)
│   ├── glycemic_control_analysis.py           # Fasting glucose outcomes and individual reductions (Fig. 3b)
│   ├── clustering_analysis.py                 # ICC and clustered analyses (Supplementary Tables 1 and 3)
│   └── operational_outcomes.py                # Delivery, latency, and draft-review outcomes (Fig. 3c)
├── questionnaire/
│   └── participant_reported.py                # Participant-reported experience panels (Fig. 4a-c)
├── sensitivity_analysis/
│   ├── ITT_weight_loss.py                     # ITT sensitivity analysis for weight-loss cohort (MI + BOCF)
│   ├── ITT_glycemic.py                        # ITT sensitivity analysis for glycemic-control cohort (MI + BOCF)
│   └── tipping_point_analysis.py              # MNAR delta-adjustment and tipping-point analysis for both cohorts
└── Subgroup Forest Plot/
    ├── weight-loss subgroup forest plot.py    # Subgroup forest plot for weight-loss cohort
    └── glycemic control subgroup forest plot.py  # Subgroup forest plot for glycemic-control cohort
```

## System Requirements

- **Python**: 3.9 or later
- **Operating system**: Tested on Ubuntu 22.04; compatible with macOS and Windows
- **Hardware**: Local model inference (benchmark evaluation) requires a CUDA-capable GPU with at least 16 GB VRAM for 14B-parameter models. All statistical analysis and plotting scripts run on CPU.
- **Dependencies**: All required packages are listed in `requirements.txt`. Key packages include `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `torch`, and `transformers`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ling112211/Exercise-Prescription-System.git
   cd Exercise-Prescription-System
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Installation should take less than 10 minutes on a standard machine with a stable internet connection.

## Data Status: Real vs. Example

> **Important**: Not all scripts in this repository can be run with real data out of the box. Please read this section carefully before running any scripts.

| Script | Output | Data in this repo |
|--------|--------|-------------------|
| `benchmark/plot_benchmark.py`, `benchmark/plot_eps_accuracy_gain.py` | Fig. 5a,c | `data/benchmark_results/benchmark_accuracy.csv` (**real**) |
| `expert_pilot/plot_expert_evaluation.py`, `expert_pilot/plot_ablation_gain_heatmap.py` | Fig. 5b,d | `data/expert_pilot/*.csv` (**real**) |
| `clinical_trial/baseline_characteristics.py` | Table 1 | `data/example/` (**example only**) |
| `clinical_trial/weight_loss_analysis.py` | Fig. 3a | `data/example/` (**example only**) |
| `clinical_trial/glycemic_control_analysis.py` | Fig. 3b | `data/example/` (**example only**) |
| `clinical_trial/clustering_analysis.py` | Supplementary Tables 1 and 3 | Controlled-access complete-case workbooks with health manager IDs (**not bundled**) |
| `clinical_trial/operational_outcomes.py` | Fig. 3c | Derived check-in summary; real trial chat data are controlled access |
| `clinical_trial/checkin_analysis/*.py` | Tagged-checkin linkage + Supplementary Table 5 frequency-control/content-audit outputs | `data/example/checkin/` (**example only**) |
| `questionnaire/participant_reported.py` | Fig. 4a-c | `data/example/` (**example only**) |
| `Subgroup Forest Plot/*.py` | Supplementary Figs. 1-2 | `data/example/` (**example only**) |
| `sensitivity_analysis/ITT_weight_loss.py` | Supplementary Table 2 (ITT weight-loss) | Controlled-access trial Excel files in `weight-loss/` and `sensitivity_analysis/weight loss missing data/` (**not bundled**) |
| `sensitivity_analysis/ITT_glycemic.py` | Supplementary Table 4 (ITT glycemic) | Controlled-access trial Excel files in `glycemic/` and `sensitivity_analysis/glycemic control missing data/` (**not bundled**) |
| `sensitivity_analysis/tipping_point_analysis.py` | Supplementary Tables 2 and 4 (tipping-point rows) | Same controlled-access inputs as the two ITT scripts above (**not bundled**) |

The example data under `data/example/` are anonymised synthetic files provided solely to verify that the code runs without errors. They do **not** reproduce the numerical results or figures reported in the paper. To obtain the real clinical trial data (weight-loss RCT, glycemic-control RCT, and questionnaire), please contact the corresponding author (see [Data Availability](#data-availability)).

For the three `sensitivity_analysis/*.py` scripts, the current implementation uses fixed input paths (no command-line data-path arguments). To run these scripts, you must place controlled-access trial Excel files in the expected `weight-loss/`, `glycemic/`, and `sensitivity_analysis/* missing data/` directories. These files are **not** bundled in this repository.

For `clinical_trial/checkin_analysis/*.py`, the repository bundles synthetic participant workbooks and chat-export workbooks under `data/example/checkin/` so the tagged-message linkage plus exploratory/enhanced mediation workflows can be run end-to-end without controlled chat exports. These files are strictly for verification and do **not** reproduce any paper result.

## D1 Corpus Inventory

The current D1 corpus contains 85,469 records across 18 JSON files. Every file is a top-level JSON array, and each record count below is the number of elements in that array. The descriptions and field lists were checked directly against the current local files in `D1/`.

| File | Records | Content | Top-level fields |
|------|--------:|---------|------------------|
| `medbooks-18-cot-filtered.json` | 11,591 | Filtered medical textbook chain-of-thought Q&A in chat-style format | `id`, `messages` |
| `medinstruct-52k-filtered.json` | 7,945 | Filtered medical instruction Q&A in chat-style format | `id`, `messages` |
| `chatdoctor-cleaned-filtered.json` | 11,731 | Cleaned and filtered ChatDoctor patient Q&A in chat-style format | `id`, `messages` |
| `medicationqa-filtered.json` | 17 | Filtered medication Q&A in chat-style format | `id`, `messages` |
| `mts-dialog-filtered.json` | 71 | Filtered medical dialogue data in chat-style format | `id`, `messages` |
| `liveqa-filtered.json` | 51 | Filtered LiveQA medical Q&A in chat-style format | `id`, `messages` |
| `huatuo_encyclopedia_qa-filtered_flat.json` | 36,977 | Filtered Huatuo encyclopedia Q&A | `questions`, `answers` |
| `huatuo_knowledge_graph_qa-filtered_flat.json` | 6,326 | Filtered Huatuo knowledge graph Q&A | `questions`, `answers` |
| `Huatuo26M-Lite-filtered_flat.json` | 6,924 | Filtered Huatuo-26M-Lite medical Q&A with disease and quality metadata | `id`, `question`, `answer`, `label`, `related_diseases`, `score` |
| `medical_o1_sft_Chinese-filtered.json` | 473 | Filtered Chinese medical reasoning data with questions, reasoning traces, and responses | `Question`, `Complex_CoT`, `Response` |
| `train_CMExam_single_sft.json` | 196 | CMExam single-choice questions in chat-style SFT format | `id`, `messages`, `kto_tag` |
| `multimedqa_sft.json` | 43 | MultiMedQA records in chat-style SFT format | `id`, `messages`, `kto_tag` |
| `medbullets_sft.json` | 47 | MedBullets Q&A in chat-style SFT format | `id`, `messages`, `kto_tag` |
| `CMB_multiple_sft.json` | 123 | CMB multiple-choice questions in chat-style SFT format | `id`, `messages`, `kto_tag` |
| `train_CMB_sin_sft.json` | 811 | CMB single-choice questions in chat-style SFT format | `id`, `messages`, `kto_tag` |
| `CMExam_multiple_sft.json` | 3 | CMExam multiple-choice questions in chat-style SFT format | `id`, `messages`, `kto_tag` |
| `medqa_train.json` | 1,072 | MedQA (USMLE) training questions in chat-style format | `id`, `messages` |
| `medmcqa_train.json` | 1,068 | MedMCQA training questions in chat-style format | `id`, `messages` |
| **Total** | **85,469** | **18 JSON datasets** | |

## How to Reproduce the Results

All scripts are run from the repository root. Data files must first be obtained (see [Data Availability](#data-availability)).

### Benchmark Evaluation (Fig. 5a and 5c)

Fig. 5a and Fig. 5c use the same pre-computed benchmark accuracy CSV.

**Step 1 — Run model inference** (requires GPU; skip if using pre-computed CSV)

Evaluates EPS variants and baseline models across four medical benchmarks. Local models are loaded via HuggingFace; flagship model APIs require environment variables for API keys.

```bash
# Set API keys for proprietary models (optional, skip if only evaluating local models)
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."

# Run benchmark evaluation (10 runs per model per benchmark by default)
python benchmark/evaluate_benchmark.py --n_runs 10 --save_details
```

Before running, update the model path placeholders in `benchmark/evaluate_benchmark.py` (`LOCAL_MODEL_SPECS`) to point to your local model checkpoints or HuggingFace model IDs.

**Step 2 — Plot the bar chart from pre-computed CSV**

A pre-computed accuracy CSV is provided at `data/benchmark_results/benchmark_accuracy.csv`. Generate Fig. 5a and Fig. 5c directly without re-running inference:

```bash
python benchmark/plot_benchmark.py \
    --input  data/benchmark_results/benchmark_accuracy.csv \
    --outdir outputs/benchmark

python benchmark/plot_eps_accuracy_gain.py \
    --input  data/benchmark_results/benchmark_accuracy.csv \
    --outdir outputs/benchmark
```

Both `--input` and `--outdir` have the above defaults and may be omitted when running from the repository root.

### Expert Pilot Evaluation (Fig. 5b and 5d)

Computes descriptive mean scores with two-sided 95% t-based confidence intervals, runs Friedman omnibus tests with Holm adjustment across the seven dimensions, runs paired Wilcoxon signed-rank tests for the three prespecified pairwise comparisons, and generates Fig. 5b and Fig. 5d.

```bash
python expert_pilot/plot_expert_evaluation.py \
    --input-dir data/expert_pilot \
    --pair-key rater_id \
    --outdir outputs/expert_pilot

python expert_pilot/plot_ablation_gain_heatmap.py \
    --input outputs/expert_pilot/expert_pilot_means_ci.csv \
    --outdir outputs/expert_pilot
```

Expected input files under `data/expert_pilot/`: `base_model.csv`, `eps_without_d2.csv`, `eps.csv`. Each CSV contains a shared rater identifier column (`rater_id` in the bundled data) and seven question columns (Q1–Q7) with A/B/C/D grades or numeric 0–3 scores from 25 health managers.

The script writes the following files to the output directory:
- `<prefix>_means_ci.csv` — descriptive means, SDs, and 95% confidence intervals.
- `<prefix>_aligned_scores.csv` — the paired analysis table after aligning raters across the three files.
- `<prefix>_friedman_tests.csv` — omnibus Friedman test results with Holm adjustment across the seven dimensions.
- `<prefix>_wilcoxon_pairwise_tests.csv` — paired Wilcoxon results with raw P values plus two Holm-adjusted columns: `p_holm_3pairs_within_dimension` (the manuscript reporting column) and `p_holm_7dims_within_comparison` (exported for transparency).
- `<prefix>_bar_mean_ci.pdf` and `<prefix>_bar_mean_ci.png` - the grouped bar chart used for Fig. 5b.
- `phase1_ablation_gain_heatmap.pdf/.png` - the incremental gain heatmap used for Fig. 5d.

### Baseline Characteristics (Table 1)

Generates the demographic comparison tables for both trial cohorts.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Table 1 in the paper. Replace the paths with your real data files once access has been granted.

```bash
python clinical_trial/baseline_characteristics.py \
    --weight_human data/example/weight_loss/human_arm.xlsx \
    --weight_eps   data/example/weight_loss/eps_arm.xlsx \
    --gly_human    data/example/glycemic/human_arm.xlsx \
    --gly_eps      data/example/glycemic/eps_arm.xlsx \
    --out_dir      outputs/clinical_trial
```

### Weight-Loss Outcomes (Fig. 3a)

Computes Welch *t*-test statistics and Clopper-Pearson confidence intervals, then generates the three bar summaries and cumulative response curve in Fig. 3a.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Fig. 3a in the paper. Replace the paths with your real data files once access has been granted.

```bash
python clinical_trial/weight_loss_analysis.py \
    --weight_human data/example/weight_loss/human_arm.xlsx \
    --weight_eps   data/example/weight_loss/eps_arm.xlsx \
    --out_dir      outputs/clinical_trial
```

If you already have the final `Results` workbook used for manuscript plotting, you can generate the publication-style figure directly:

```bash
python clinical_trial/weight_loss_analysis.py \
    --summary-xlsx path/to/EPS_vs_Human_weightloss_effects.xlsx \
    --weight_human path/to/Human_weight-loss.xlsx \
    --weight_eps   path/to/EPS-Human_weight-loss.xlsx \
    --out_dir      outputs/clinical_trial
```

The script writes both the manuscript filenames (`weight_loss_bars.pdf/.png`) and `_nm_style` aliases.

### Glycemic-Control Outcomes (Fig. 3b)

Computes fasting glucose reduction statistics and generates the two bar summaries plus individual reduction view in Fig. 3b.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Fig. 3b in the paper. Replace the paths with your real data files once access has been granted.

```bash
python clinical_trial/glycemic_control_analysis.py \
    --gly_human data/example/glycemic/human_arm.xlsx \
    --gly_eps   data/example/glycemic/eps_arm.xlsx \
    --out_dir   outputs/clinical_trial
```

If you already have the final `Results` workbook used for manuscript plotting, you can generate the publication-style figure directly:

```bash
python clinical_trial/glycemic_control_analysis.py \
    --summary-xlsx path/to/glycemic_control_summary_table_with_CI.xlsx \
    --gly_human   path/to/Human_glycemic-control.xlsx \
    --gly_eps     path/to/EPS-Human_glycemic-control.xlsx \
    --out_dir      outputs/clinical_trial
```

The script writes both the manuscript filenames (`glycemic_fpg_reduction_bars.pdf/.png`) and `_nm_style` aliases.

### ICC and Clustering Analyses (Supplementary Tables 1 and 3)

The clustering script reads the controlled-access complete-case workbooks and accepts either the original Chinese column names or the English alternatives below.

| Field | Accepted column names |
|-------|-----------------------|
| Participant ID | `序号`, `participant_id`, `id` |
| Community ID | `班级号`, `community_id`, `class_id` |
| Health manager ID | `健管师 id`, `健管师id`, `manager_id`, `health_manager_id` |
| Weight-loss baseline | `入营体重`, `baseline_weight_kg`, `entry_weight_kg` |
| Weight-loss endline | `出营体重`, `endline_weight_kg`, `exit_weight_kg` |
| Weight loss | `减重数`, `weight_loss_kg`, `weight_loss`; otherwise calculated as baseline minus endline |
| Fasting glucose baseline | `入营空腹`, `baseline_fpg_mmol_l`, `baseline_fpg` |
| Fasting glucose endline | `结营空腹`, `endline_fpg_mmol_l`, `endline_fpg` |

Community identifiers are treated as arm-specific, while health manager identifiers retain their shared meaning across arms. Run the analysis from the repository root:

```bash
python clinical_trial/clustering_analysis.py \
    --cohort weight_loss \
    --human-xlsx path/to/Human\ weight-loss.xlsx \
    --eps-xlsx path/to/EPS-Human\ weight-loss.xlsx \
    --outdir outputs/clustering/weight_loss

python clinical_trial/clustering_analysis.py \
    --cohort glycemic \
    --human-xlsx path/to/Human\ glycemic-control.xlsx \
    --eps-xlsx path/to/EPS-Human\ glycemic-control.xlsx \
    --outdir outputs/clustering/glycemic
```

Each run writes `icc_results.csv`, `model_results.csv`, `data_checks.csv`, `manager_summary_by_arm.csv`, `community_summary.csv`, and `analysis_metadata.json`.

### Operational Outcomes (Fig. 3c)

Generate Fig. 3c from the weight-loss output JSON produced by `clinical_trial/checkin_analysis/enhanced_feedback_mediation.py`. The default draft-review edit counts reproduce the manuscript audit.

```bash
python clinical_trial/operational_outcomes.py \
    --summary-json outputs/checkin_analysis/weight_loss/weight_loss_frequency_control_content_audit_summary.json \
    --outdir outputs/clinical_trial
```

### Participant-Reported Outcomes (Fig. 4a-c)

Applies the Q1 screening filter and complete-questionnaire validation by default. Questionnaires with any missing, unparseable, or out-of-range response to items 2-15 are excluded. Item-level comparisons use two-sided Welch tests with Holm adjustment across the 14 items. Optional completion-time and straight-lining filters are available with `--time-filter` and `--drop-straightliners` for sensitivity checks.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Fig. 4 in the paper. Replace the paths with your real data files once access has been granted.

```bash
python questionnaire/participant_reported.py \
    --human-xlsx data/example/questionnaire/human_responses.xlsx \
    --eps-xlsx   data/example/questionnaire/eps_responses.xlsx \
    --outdir     outputs/questionnaire
```

The panel files are `phase2_radar_mean_ci` (Fig. 4a), `phase2_item_mean_differences` (Fig. 4b), and `phase2_domain_response_distribution` (Fig. 4c). The script also writes item-level raw and Holm-adjusted P values, a questionnaire QC flow table, and an invalid-questionnaire audit.

### Subgroup Forest Plots (Supplementary Figs. 1 and 2)

Each script fits four omnibus treatment-by-subgroup interaction tests and applies Holm adjustment across those four tests within the cohort. The output workbook contains a separate `Interaction tests` sheet with raw and Holm-adjusted P values, and each forest plot displays the Holm-adjusted value on the corresponding subgroup-heading row.

> **Note**: The commands below use the example data provided in `data/example/`. The outputs will **not** match Supplementary Figs. 1 and 2 in the paper. Replace the paths with your real data files once access has been granted.

```bash
# Weight-loss subgroup analysis
python "Subgroup Forest Plot/weight-loss subgroup forest plot.py" \
    --eps   data/example/weight_loss/eps_arm.xlsx \
    --human data/example/weight_loss/human_arm.xlsx \
    --out-prefix outputs/subgroup/weightloss_subgroup

# Glycemic-control subgroup analysis
python "Subgroup Forest Plot/glycemic control subgroup forest plot.py" \
    --eps      data/example/glycemic/eps_arm.xlsx \
    --human    data/example/glycemic/human_arm.xlsx \
    --col_bmi  BMI \
    --out_table outputs/subgroup/glycemic_subgroup.xlsx \
    --out_png   outputs/subgroup/glycemic_subgroup.png \
    --out_pdf   outputs/subgroup/glycemic_subgroup.pdf
```

### ITT Sensitivity Analysis (Supplementary Tables 2 and 4)

Performs Intention-to-Treat sensitivity analyses using multiple imputation (MICE under MAR) and baseline observation carried forward (BOCF). MNAR delta-adjustment and tipping-point sensitivity are handled separately in `tipping_point_analysis.py`. Results are saved as multi-sheet Excel workbooks.

> **Note**: These three sensitivity-analysis scripts currently use fixed file paths and do not accept CLI data-path arguments. The required controlled-access input files are not bundled in this repository.

```bash
# Weight-loss ITT sensitivity analysis
python sensitivity_analysis/ITT_weight_loss.py

# Glycemic-control ITT sensitivity analysis (exploratory)
python sensitivity_analysis/ITT_glycemic.py
```

Expected input file locations:
- `weight-loss/Human weight-loss.xlsx`
- `weight-loss/EPS-Human weight-loss.xlsx`
- `glycemic/Human glycemic-control.xlsx`
- `glycemic/EPS-Human glycemic-control.xlsx`
- `sensitivity_analysis/weight loss missing data/weight loss human missing data.xlsx`
- `sensitivity_analysis/weight loss missing data/weight loss EPS-human missing data.xlsx`
- `sensitivity_analysis/glycemic control missing data/glycemic Human missing data.xlsx`
- `sensitivity_analysis/glycemic control missing data/glycemic EPS-human missing data.xlsx`

Fixed output files:
- `sensitivity_analysis/ITT_weight_loss_results.xlsx`
- `sensitivity_analysis/ITT_glycemic_results.xlsx`

### Tipping-Point Analysis (Supplementary Tables 2 and 4)

Determines how much worse missing outcomes in the EPS arm would need to be (relative to MAR imputation) before the treatment effect loses statistical significance.

> **Note**: `tipping_point_analysis.py` reads the same fixed input paths as the ITT scripts above and does not accept CLI data-path arguments.

```bash
python sensitivity_analysis/tipping_point_analysis.py
```

Fixed output file:
- `sensitivity_analysis/tipping_point_results.xlsx`

## Data Availability

This repository includes three categories of data:

**Fully available (real data, reproduces paper results):**
- `D1/` — public exercise and weight-management corpus containing 85,469 entries across 18 JSON datasets. It was used for supervised fine-tuning of EPS. See [D1 Corpus Inventory](#d1-corpus-inventory) for file-level counts, content, and fields.
- `data/benchmark_results/benchmark_accuracy.csv` - pre-computed benchmark accuracy scores used to generate Fig. 5a and Fig. 5c.
- `data/expert_pilot/` - expert ratings from the 25-person pilot study used to generate Fig. 5b and Fig. 5d.

**Example data only (does not reproduce paper results):**
- `data/example/` — anonymised synthetic datasets provided solely to verify that the analysis and plotting scripts run without errors. These files have the same format as the real data but contain different values. Outputs produced with these files will **not** match the tables and figures reported in the paper.

**Not bundled in this repository (required by the clinical-trial sensitivity analyses):**
- Controlled-access Excel files for weight-loss and glycemic-control completers and missing-participant baselines. The ICC analysis also requires the health manager and community identifiers in the complete-case workbooks. Data paths for `clinical_trial/clustering_analysis.py` are supplied through command-line arguments; the three `sensitivity_analysis/*.py` scripts use fixed paths under `weight-loss/`, `glycemic/`, and `sensitivity_analysis/* missing data/`.

The real clinical trial data (weight-loss RCT: 1,580 randomized participants, including 1,444 in the complete-case efficacy analysis; glycemic-control RCT: 48 randomized participants, including 40 in the complete-case efficacy analysis; participant questionnaire) are available under controlled access due to patient privacy regulations. Researchers who wish to access the de-identified participant data for academic purposes may contact the corresponding author. Please see `data/README_data.md` for a full description of each dataset and the required file format.

## Model Availability

EPS models are fine-tuned versions of open-source base models (DeepSeek-R1-8B, Qwen3-8B, DeepSeek-R1-14B, Qwen3-14B) using a two-stage alignment framework: supervised fine-tuning on domain-specific data (D1) followed by Kahneman–Tversky Optimization on expert-preference data (D2). Model checkpoints are not publicly released due to the use of proprietary training data. Qualified researchers may request access by contacting the corresponding author.

Base models are available from their official repositories:
- **Qwen3**: [github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) (Apache-2.0 License)
- **DeepSeek-R1**: [github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) (MIT License)

## Ethics and Trial Registration

The study protocol was approved by the Ethics Review Committees of City University of Hong Kong, Harbin Institute of Technology, and Ping An Health and Technology Co., Ltd. The trial was registered at the [Chinese Clinical Trial Registry (ChiCTR2600118939)](https://www.chictr.org.cn/hvshowprojectEN.html?id=295346&v=1.0). All participants provided written informed consent prior to enrollment.

## How to Cite

If you use this code in your research, please cite our paper (citation details will be updated upon publication):

```bibtex
@article{,
  author  = {},
  title   = {LLM-human collaboration for personalized exercise prescription in digital chronic disease management: a randomized controlled trial},
  journal = {},
  year    = {2025},
  doi     = {}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions regarding the code, data access, or model access, please contact:

- **Guangxin Jiang** (corresponding author): gxjiang@hit.edu.cn
- **Chenxi Li**: ling112358@gmail.com
- **Siyang Gao**: siyangao@cityu.edu.hk
