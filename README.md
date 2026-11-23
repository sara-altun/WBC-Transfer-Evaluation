[README.md](https://github.com/user-attachments/files/35274843/README.md)
# WBC Transfer Evaluation

**Cross-domain white blood cell classification with adaptation-free external evaluation**

Research code and reproducibility documentation for the study *Evaluating Cross-Domain Transfer and Adaptation-Free External Generalization of Retinal- and ImageNet-Pretrained Models for White Blood Cell Classification*.

- **Models:** ResNet-50 and ConvNeXt-Tiny (ImageNet-pretrained), and RETFound (retinal-pretrained, full fine-tuning).
- **Primary benchmark:** WBCBench 2026, 13-class evaluation, three random seeds (`42`, `1`, `7`).
- **External evaluation:** PBC and Raabin-WBC (Train + TestA only), using five common WBC classes **BA, EO, LY, MO, SNE**, without adaptation to external data.
- **Analysis:** Three-seed Macro-F1, class-wise metrics, BNE+SNE sensitivity, paired image-level bootstrap intervals, and manuscript tables/figures.

**Release reference:** [v1.0](https://github.com/sara-altun/WBC-Transfer-Evaluation/releases/tag/v1.0). This README documents the *corrected evaluation workflow*. The availability of any particular source file or artifact in the v1.0 release should be checked against the release assets; the release tag itself is not a guarantee that it contains every notebook listed below.

## 1. Dataset access and citations

**Download each dataset from its original publisher. Do not upload clinical images or redistribute the WBCBench files through this repository.**

| Dataset | Official access | Study use |
| --- | --- | --- |
| **WBCBench 2026** | [Xin-Tian/wbcbench2026 (Hugging Face)](https://huggingface.co/datasets/Xin-Tian/wbcbench2026) | Training (`phase2_train`), validation (`phase2_eval`), locked testing (`phase2_test`) |
| **PBC** (Acevedo et al.) | [Mendeley Data, DOI:10.17632/snkd93bnjr.1](https://data.mendeley.com/datasets/snkd93bnjr/1) | Five-class external test, from `PBC_dataset_normal_DIB` |
| **Raabin-WBC** | [Official Raabin Data access](https://raabindata.com/free-data/) | `Train` and `TestA` only; **TestB was not evaluated** |

**Access and version cautions.** WBCBench requires acceptance of the dataset's access and non-redistribution terms. The public WBCBench repository documents subsequent test-label corrections; the exact snapshot used to regenerate a reported result must be established from original local download metadata and, ideally, an immutable revision or file hashes. Do not compare scores against a different release without accounting for label changes. `patient_hash` is a blood-film/case identifier, **not a unique-person identifier**. PBC and Raabin-WBC data remain subject to their own terms and attribution requirements.

Dataset references:
- Tian et al., *WBCBench 2026: A Challenge for Robust White Blood Cell Classification Under Class Imbalance*, ISBI 2026 (see the WBCBench dataset card for the definitive citation).
- Acevedo et al., *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief (2020), DOI: [10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1).
- Mousavi Kouzehkanan et al., *Raabin-WBC: a large free access dataset of white blood cells from normal peripheral blood*, Scientific Reports (see [official source](https://raabindata.com/)).

## 2. Local project layout

The notebooks were originally executed **from the project root**. After independently obtaining and organizing the datasets, the expected local paths are:

```text
WBC-Transfer-Evaluation/              # or the local sci5-wbc project root
├── README.md
├── LICENSE
├── RETFound/
│   └── models_vit.py                 # RETFound upstream implementation (its own license)
├── wbc_project/
│   └── data/processed/
│       ├── phase2_train/<13-class-subfolders>/
│       ├── phase2_eval/<13-class-subfolders>/
│       └── phase2_test/<13-class-subfolders>/
├── data/external/
│   ├── PBC_dataset_normal_DIB/<five-class-subfolders>/
│   └── Raabin_WBC/
│       ├── Train/...
│       └── TestA/...
├── outputs/
│   ├── final/<model>/seed<seed>/<best-checkpoint>.pt
│   ├── final_test_v2/<model>/seed<seed>/
│   │   ├── true.npy
│   │   ├── pred.npy
│   │   ├── probabilities.npy
│   │   └── DONE.json
│   └── asama18_verified_v13/          # generated tables and plots
├── review_results_external_BNE_SNE_v6/
├── review_results_ret_found_only_v10/
└── review_results_external_consistency_v13/
```

For the external PBC/Raabin evaluation, the five-class names used locally are `basophil`, `eosinophil`, `lymphocyte`, `monocyte`, and `neutrophil`. Keep the original file order. The external auditing notebook checks the eligible-image manifest, excluding unreadable images using `PIL.Image.verify()` rather than arbitrarily removing files.

These paths describe the **local workflow**, not a claim that datasets or checkpoint binaries are included in the GitHub repository.

## 3. Software environment

Python 3.10, Jupyter Notebook, NumPy, pandas, scikit-learn, matplotlib, Pillow, tqdm, and PyTorch/torchvision are used across this workflow; RETFound inference additionally requires the appropriate `timm` version and RETFound model definition. Training and evaluation may require **different compatible PyTorch environments**. Install a compatible CUDA build of PyTorch for your device, or use CPU where supported.

For the **table-and-figure export only** (no GPU, training, or inference), a minimal setup is:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your operating system.
python -m pip install numpy pandas scikit-learn matplotlib pillow ipython jupyter
jupyter notebook
```

This minimal installation is **not** a complete, tested lockfile for retraining all three architectures. Reproduce training/inference using the environment and checkpoint metadata associated with each experiment.

## 4. Execution order

There are two ways to use this project. **Do not rerun model training simply to recreate manuscript tables if the original audited predictions are already available.** The notebooks below use filenames from the corrected local research workflow. Ensure each notebook is actually committed to the repository or provided in a release before advertising the corresponding step as publicly executable.

### A. Reproduce tables and figures from retained predictions (reviewer-friendly)

1. **WBCBench retained-test checks** (optional independent replay): `RETFound_WBCBench_V12_Tam_Dogrulama.ipynb` compares the three RETFound checkpoints against saved 13-class predictions. The complete replay was **not bitwise identical**: 9 class assignments differed across 98,802 predictions, with small Macro-F1 changes. Do not call this a strict `PASS_FULL`.
2. **CNN external five-class/BNE+SNE exports:** `WBC_External_BNE_SNE_V6_Tam_Analiz.ipynb` (or the validated original V6 exports), for the two CNNs.
3. **Corrected RETFound external predictions:** `WBC_RETFound_Only_V10_Dosya_Yollarina_Gore.ipynb`. It loads the final RETFound checkpoints with the **training-matched `global_pool=True`** architecture and performs the five-class / BNE+SNE external evaluations.
4. **External audit and paired intervals:** `WBC_External_V13_Duzeltilmis_Gecerli_Goruntu_Kontrolu.ipynb`. Input: the V6 and V10 saved predictions. Output: `review_results_external_consistency_v13/`. This performs a cached-record audit, not independent re-inference; it verified 18/18 runs in the reported analysis.
5. **Manuscript exports:** `Asama18_Manuscript_Tables_Figures_V2_Verified.ipynb`. Input: saved 13-class WBCBench predictions and V13-verified external records. Output: `outputs/asama18_verified_v13/`.

The last notebook creates files such as `Table1_Main_Performance.csv`, `Table2_WBCBench_PerClass_F1.csv`, `Table3_External_PairedBootstrap_CI.csv`, `Table_External_BNE_SNE_Sensitivity.csv`, `Figure2_MacroF1_Three_Datasets.png`, and `Figure3_External_BNE_SNE_Sensitivity.png`.

### B. Full workflow, when training code and data are available

1. Obtain approved WBCBench `phase2_train`, `phase2_eval`, and `phase2_test` partitions, and the external PBC/Raabin-WBC subsets; preserve the original partitions and classes.
2. Fine-tune ResNet-50, ConvNeXt-Tiny, and RETFound using the original training notebooks and validation-only checkpoint selection. The available RETFound training notebooks are `WBCBench_2026_Asama11A_v2_RETFound_Seed42_Resume_Max12_Patience3.ipynb` and `WBCBench_2026_Asama11B_RETFound_Final_FullFT_Seeds1_7.ipynb`; the seed-42 run was **resumed from a training checkpoint** rather than restarted. CNN training notebook names are not asserted here because the exact release manifest has not been verified.
3. Produce the locked 13-class predictions with `WBCBench_2026_Asama12_v2_Final_Test_Hizli_Progress.ipynb`, using fixed checkpoint files and `shuffle=False`.
4. Run the corrected external evaluation and verification in **Section 4A, steps 2–5**. Do **not** substitute the historical Aşama 15C RETFound external predictions: that older evaluation did not load the fine-tuned global-pooling architecture consistently.

The comparison concerns **trained model configurations**, not the isolated causal effect of retinal versus ImageNet pretraining. External evaluation uses no target-domain fine-tuning, calibration, or target-based model selection.

## 5. Evaluation definitions and provenance

- **WBCBench:** 13-class Macro-F1 is the primary metric; 32,934 saved test-image entries (including pristine/degraded counterparts) were reported in the local experiment. The matched five-class analysis is a *different task* and must not be presented as a 13-class score.
- **External five-class mapping:** Retain only BA, EO, LY, MO and SNE output scores for the primary external rule. For the sensitivity analysis, replace the segmented-neutrophil score with the **combined BNE+SNE probability**, leaving the remaining mapped classes unchanged. All five competing scores must be normalized/handled consistently when comparing decisions.
- **External observations:** PBC: 10,298 readable images from the selected five classes; Raabin-WBC: 14,514 images from `Train` + `TestA`. The original PBC candidate manifest included one unreadable image, which is excluded by the validated loader.
- **Sampling uncertainty:** The external 95% confidence intervals use **1,500 class-stratified, paired image-level bootstrap resamples**, with the statistic defined as the mean of three seed-specific Macro-F1 differences. These intervals **do not account for film- or patient-level clustering**. WBCBench film/case-cluster sensitivity analysis is a distinct analysis and must not be conflated with the external image bootstrap.
- **Numerical reproducibility:** Due to software, batch, and floating-point effects, repeat inference may yield tiny changes in class predictions. Record the exact environment and check the resulting metrics; do not assume bitwise identity.

### Reported corrected external scores

These are **reference values from the V13 audited local outputs**, not values guaranteed by the GitHub release without the requisite datasets/checkpoints. Arithmetic mean over seeds 42, 1, 7:

| External dataset | Model | Baseline Macro-F1 | BNE+SNE Macro-F1 |
| --- | --- | ---: | ---: |
| PBC | ConvNeXt-Tiny | 0.9650 | 0.9628 |
| PBC | ResNet-50 | 0.9255 | 0.9241 |
| PBC | RETFound | 0.8734 | 0.8729 |
| Raabin-WBC (Train + TestA) | ConvNeXt-Tiny | 0.7076 | 0.7036 |
| Raabin-WBC (Train + TestA) | ResNet-50 | 0.6209 | 0.6135 |
| Raabin-WBC (Train + TestA) | RETFound | 0.4408 | 0.4409 |

The corrected RETFound external evaluation uses the fine-tuned architecture and replaces superseded external RETFound scores; it does **not** imply that the primary 13-class WBCBench experiment was retrained.

## 6. Releasing a reproducible version

Before claiming that a tagged release reproduces the manuscript:

1. Confirm that all **named notebooks** and necessary supporting source files are actually present at the cited tag; a link to `v1.0` by itself does not establish this.
2. Include a package/environment specification tested for the actual published code, and fix the RETFound upstream commit/dependency versions.
3. Identify the **exact WBCBench data revision**, because the public dataset card documents later test-label corrections. Record external data selection rules and any stable file manifest hashes.
4. If code cannot be rerun without private checkpoints, state that limitation and what artifacts are available. Never publish confidential data, credentials, patient-level identifiers, or WBCBench images in violation of the dataset licence.
5. If the corrected V13 results were generated **after** the original v1.0 release, publish an updated tagged release (for example `v1.1`) rather than implying that the original v1.0 already contains those corrections.

## 7. Licence and third-party material

The accompanying [`LICENSE`](LICENSE) proposes the **MIT licence for author-owned code and documentation only**. Confirm that this is your intended licensing decision before publishing or replacing any existing repository licence. The licence **does not** cover WBCBench, PBC, Raabin-WBC, pretrained model weights, or external RETFound/other third-party source code; those materials remain governed by their respective publishers' terms.

**Maintainer:** Sara Altun Güven  
**Release reference:** [WBC-Transfer-Evaluation v1.0](https://github.com/sara-altun/WBC-Transfer-Evaluation/releases/tag/v1.0)
