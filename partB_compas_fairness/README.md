If you meet the error from running dataset, please download this to fix it. I can not add this to requirement.txt file because these command need run manually:
!mkdir -p /usr/local/lib/python3.11/dist-packages/aif360/data/raw/compas
!wget https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv -O /usr/local/lib/python3.11/dist-packages/aif360/data/raw/compas/compas-scores-two-years.csv

partB_compas_fairness/ Q4
After applying the Reweighing method, Statistical Parity Difference improved significantly from 0.1990 to 0.0136, showing reduced bias in positive classification rates between racial groups.
Equalized Odds Difference also improved (from 0.1089 to -0.0610), but to a lesser extent.

The greater improvement was in SPD.
Trade-off: Accuracy slightly decreased from 0.6688 to 0.6645, showing a mild compromise between fairness and performance.