# Stroke SLP Analysis Index

This file separates the active manuscript pipeline from exploratory and legacy
scripts. The project root now contains build/run entry points; canonical output
builders live in `analysis/`; old one-offs live in `archive/`.

## Canonical Pipeline

Run the full build and output pipeline:

```powershell
python projects\stroke_SLP\run_pipeline.py
```

Run only the current manuscript-ready figures and tables after PSM has already
been built:

```powershell
python projects\stroke_SLP\run_analysis_outputs.py
```

## Database Build

These scripts define the current table contracts:

| Step | Script | Output |
| --- | --- | --- |
| 1 | `queries/stroke_cohort.sql` | `stroke_cohort` |
| 2 | `queries/stroke_slp.sql` | `stroke_slp` |
| 3 | `build_stroke_comorbidity.py` | `stroke_comorbidity` |
| 4 | `queries/stroke_propensity.sql` | `stroke_propensity` |
| 5 | `queries/stroke_outcomes.sql` | `stroke_outcomes` |
| 6 | `stroke_psm.py` | Updates `stroke_propensity.psm_matched_A`, `psm_match_id_A`, `prop_score_A` |

## Canonical Manuscript Outputs

These are the scripts run by `run_analysis_outputs.py`:

| Script | Output |
| --- | --- |
| `analysis/make_flowchart.py` | `output_files/fig1_cohort.png` |
| `analysis/make_table1.py` | `output_files/Table1.xlsx` |
| `analysis/make_table2.py` | `output_files/Table2.xlsx` |
| `analysis/make_results_outputs.py` | `output_files/Figure2.png`, `output_files/Table3.xlsx` |
| `analysis/make_gradient_elix.py` | `output_files/Supp_Figure1.png`, `output_files/Supp_Table1.xlsx` |
| `analysis/comp_b_sensitivity.py` | `output_files/Supp_Figure2.png`, `output_files/Supp_Table2.xlsx` |
| `analysis/comp_b_stratified.py` | `output_files/Supp_Figure3.png`, `output_files/Supp_Table3.xlsx` |
| `analysis/make_love_plot.py` | `output_files/Supp_Figure4.png` |
| `analysis/stroke_slp_common.py` | Shared DB/path, J18/J69, and time-varying Cox helpers |

## Optional / Exploratory

| Script | Status |
| --- | --- |
| `archive/exploratory/stroke_analysis.py` | Console exploration of OR, KM, cost, and TV Cox results. Useful for checking results, but not the final output generator. |
| `archive/exploratory/comp_week_bins.py` | Design exploration comparing Wk1-4 vs Wk6+ and Early vs Wk5+. Not the final supplementary week-by-week script. |
| `rerun_slp_propensity.py` | Utility to rebuild only `stroke_slp` and `stroke_propensity` after SLP definition changes. Run `stroke_psm.py` afterward. |
| `archive/exploratory/check_cols.py` | One-off table inspection utility. |

## Legacy / Stale

These scripts refer to older timing groups or older output conventions and
should not be used for primary manuscript results without revision:

| Script | Reason |
| --- | --- |
| `archive/legacy/_diag.py` | One-off matched-cohort diagnostic. |
| `archive/legacy/_diag_day0.py` | References old `0-14d` timing group that is not present in current `stroke_propensity`. |
| `archive/legacy/_sensitivity_nopsm.py` | References old `1-14d`, `15-30d`, `31-90d` timing groups. |
| `archive/legacy/make_stroke_table.py` | Legacy Table 2 workbook generator; superseded by `analysis/make_table2.py`, which keeps the comprehensive workbook shape but uses current outcome and modeling conventions. |
