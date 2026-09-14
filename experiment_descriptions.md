# Experiment descriptions
1. History, no text, full dataset
2. History, text, full dataset, 1B LLM
3. No history, full dataset, 1B LLM (Control 1)
4. History, no text, discharge summary subset
5. History, text, discharge summary subset, 1B LLM
6. No history, discharge summary subset
7. History, text, full dataset, 70B LLM
8. History, text, full dataset, 70B LLM, bigger encoders (not used)
9. History only, text, full dataset, 1B LLM (Control 2)

## Revision experiments

Run before the pre-admission cutoff and the merged feature set. Superseded by
20-29, and kept as the record of what that run reported.

10. In-Stay Records Only, Patients With At Least 1 Discharge Summary
11. Historical Records Only, Text Features, Patients With At Least 1 Discharge Summary
12. In-Stay + Historical Records, No Text Features, Patients With At Least 1 Discharge Summary
13. In-Stay + Historical Records, Text Features, Patients With At Least 1 Discharge Summary
14. In-Stay + Text Features Only, Patients With At Least 1 Discharge Summary
15. In-Stay Records Only, Patients With At Least 1 Historical Record
16. Historical Records Only, Text Features, Patients With At Least 1 Historical Record
17. In-Stay + Historical Records, Text Features, Patients With At Least 1 Historical Record
18. In-Stay Records Only, Patients With A Charlson Comorbidity Index

## Redo experiments

Written by `generate_revision_experiments.py`. Each inherits the tuned
hyperparameters and differs only in which records reach the model and which
patients it runs on. "Peri-stay" is the era from the pre-admission cutoff
through the end of the prediction window; "historical" is everything earlier.

20. Peri-Stay Records Only, Patients With At Least 1 Pre-Admission Text Record
21. Historical Records Only, Text Features, Patients With At Least 1 Pre-Admission Text Record
22. Peri-Stay + Historical Records, No Text Features, Patients With At Least 1 Pre-Admission Text Record
23. Peri-Stay + Historical Records, Text Features, Patients With At Least 1 Pre-Admission Text Record
24. Peri-Stay + Text Features Only, Patients With At Least 1 Pre-Admission Text Record
25. Peri-Stay Records Only, Patients With At Least 1 Historical Record
26. Historical Records Only, Text Features, Patients With At Least 1 Historical Record
27. Peri-Stay + Historical Records, Text Features, Patients With At Least 1 Historical Record
28. Peri-Stay Records Only, Patients With A Charlson Comorbidity Index

## Charlson comparison

29 is not a TransEHR2 run. It is a logistic regression on age at admission, sex and the
Charlson comorbidity index of the patient's most recent earlier hospital admission, fitted by
`run_charlson_logistic_regression.py` on the index that `compute_charlson_index.py` computes.
28 is its control: the same peri-stay-only model as 20 and 25, on the same episodes.

Their cohort is not one of the named predicates. It is the episode manifest
`compute_charlson_index.py --write_cohort` writes -- the episodes for which all three features
exist -- and both arms are given that one file, 28 through `COHORT_EPISODES` in its config.
That is what puts them on identical episodes in identical order, which the paired test needs.

29. Age, Sex And Charlson Comorbidity Index, Patients With A Charlson Comorbidity Index
