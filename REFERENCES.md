# References & Data Sources

## Dataset

**USOVA3D** — a 3D ovarian ultrasound volume dataset built for an automated
ovary/follicle segmentation challenge. Each volume has independent
annotations from two expert raters for both the ovary (`_o_r1`, `_o_r2`) and
individual follicles (`_f_r1`, `_f_r2`).

- 16 training volumes (IDs `1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110,
  111, 115, 117, 119`)
- 19 held-out test volumes (IDs `7, 8, 9, 11, 12, 13, 14, 15, 103, 105, 107,
  108, 112, 113, 114, 116, 118, 120, 121`)
- Distributed as `Training_Set_2019.zip`, `Test_Set_2019.zip`, and
  `USOVA3D_tools.zip`
- Official submission/evaluation format documented in
  `Description_Data_Structures_For_Evaluation.pdf` (supplied with the
  dataset, not included in this repository)

**The dataset is not included in this repository.** It must be obtained
through the official USOVA3D challenge channel and placed manually in
`data/` before running `scripts/01_extract_zips.py`.

> **Licensing / usage terms — action needed:** this repository does not
> currently document the dataset's exact usage/redistribution terms (they
> were supplied alongside the download, e.g. in
> `Description_Data_Structures_For_Evaluation.pdf` or an accompanying
> agreement). Before sharing this project publicly, add the dataset's
> actual terms here (attribution requirements, permitted uses, whether
> results/derived data may be published) and confirm this project complies
> with them. In the meantime, treat the dataset as **restricted / for
> research use under the challenge's original terms only** — do not
> redistribute the raw or annotated volumes.

## Published baseline

Potočnik, B. et al. (2020). *USOVA3D ovarian ultrasound challenge* —
baseline results cited in `RESULTS_LOG.md`:

- 3D Directional Wavelet Transform baseline: ~78/100 on the official
  composite scoring metric
- CNN baseline: slightly lower than the wavelet baseline
- Human inter-rater agreement: ~83/100

The composite metric combines matching (via `czEvaluate.m` /
`AgreedBothExperts.m`) with volume-ratio and surface-distance terms for
matched follicles. **This project currently implements only the matching
step** (ported to `src/follicle_matching.py`) and reports
sensitivity/precision/F1 instead of the full composite score — see
"Known gaps" in the README. Numbers in `RESULTS_LOG.md` are therefore *not*
directly comparable to the 78/100 and 83/100 figures above until the full
metric is implemented.

> Full bibliographic details (venue, DOI) for the Potočnik et al. 2020
> paper are not recorded anywhere in this repo yet — only the author
> surname and year are known. Add the full citation here once confirmed.

## Official evaluation algorithm

The follicle ground-truth consensus and scoring logic in
`src/follicle_matching.py` is a Python port of the official USOVA3D
MATLAB evaluation scripts:

- `czEvaluate.m`
- `AgreedBothExperts.m`

Matching uses a greedy best-match-first strategy on `rho1 * rho2`, where
`rho1` is intersection-over-ground-truth-area (recall-like) and `rho2` is
intersection-over-detection-area (precision-like). This was ported
deliberately so that internal evaluation is directly comparable to the
published protocol rather than an invented criterion.

## Clinical literature (follicle maturity thresholds)

Cited in the `measure_follicles()` / classification docstrings in
`src/pipeline.py` to justify the default 16–22mm "optimal for retrieval"
window:

- Nature Communications (2025) — noted that no single universal clinical
  threshold exists; "definitive clinical consensus... is yet to be
  determined."
- Lead follicle ≥17–18mm as the standard trigger criterion; 16–22mm cited
  as the range most likely to yield a mature oocyte at retrieval
  (CNY Fertility; PMC5930292).
- Diminished ovarian reserve patients: 15–17mm optimal (J Ovarian Res,
  2025).
- Poor responders (POSEIDON group 3/4): 18–24mm optimal (PMC12406972).

> These are recorded as PubMed Central IDs and journal names only. For a
> formal write-up (thesis, paper, or clinical validation), resolve each to
> a full citation (authors, title, volume/issue, DOI) — PMC5930292 and
> PMC12406972 can be looked up directly on PubMed Central.

## No formal citation file

There is currently no `CITATION.cff`. If this project is meant to be cited
by others (e.g. a DTU course submission or thesis appendix), consider
adding one once the Potočnik et al. full citation is confirmed above.
