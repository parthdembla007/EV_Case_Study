# EV charging + swapping operator: AI decision framework (case study)

All data are synthetic, generated from the labelled assumptions in report Section 3. These are not company results.

| Path | Content |
|---|---|
| `report/EV_Energy_Decision_System_Report.pdf` | Final report (PDF) |
| `report/EV_Energy_Decision_System_Report.docx` | Same report, editable Word file (native equations) |
| `report/report_template.md` | Report source; `{{placeholders}}` are filled from the CSVs |
| `code/sim.py` | Synthetic data, rule-based baseline, stochastic day-ahead LP, hourly MPC MILP (PuLP + HiGHS), KPIs, P&L |
| `code/run_all.py` | Runs all lever configurations, scenarios, seeds and sensitivities (4 workers, about 11 min) |
| `code/charts.py` | Regenerates every figure from the CSVs |
| `code/build_report.py` | Fills the template and builds PDF (XeLaTeX) + DOCX (pandoc) |
| `outputs/csv/` | Daily KPIs for every run, hourly dispatch (A vs B), annual P&L by configuration, scenario means, sensitivity, forecast metrics, validation checks, financials |
| `outputs/charts/` | 14 PNG figures |

To reproduce:

```
pip install pulp highspy numpy pandas matplotlib python-docx pypandoc_binary
python code/run_all.py 4
python code/charts.py
python code/build_report.py
```

Requires a LaTeX engine (XeLaTeX) for the PDF.
