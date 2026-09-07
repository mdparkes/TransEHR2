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

Written by `generate_revision_experiments.py`. Each inherits the tuned
hyperparameters and differs only in which records reach the model and which
patients it runs on.

10. In-Stay Records Only, Patients With At Least 1 Discharge Summary
11. Historical Records Only, Text Features, Patients With At Least 1 Discharge Summary
12. In-Stay + Historical Records, No Text Features, Patients With At Least 1 Discharge Summary
13. In-Stay + Historical Records, Text Features, Patients With At Least 1 Discharge Summary
14. In-Stay + Text Features Only, Patients With At Least 1 Discharge Summary
15. In-Stay Records Only, Patients With At Least 1 Historical Record
16. Historical Records Only, Text Features, Patients With At Least 1 Historical Record
17. In-Stay + Historical Records, Text Features, Patients With At Least 1 Historical Record
18. In-Stay Records Only, Patients With A Charlson Comorbidity Index

## Charlson comparison

19 is not a TransEHR2 run. It is a logistic regression on age at admission, sex and the
Charlson comorbidity index of the patient's most recent earlier hospital admission, fitted by
`run_charlson_logistic_regression.py` on the index that `compute_charlson_index.py` computes.
18 is its control: the same in-stay-only model as 10 and 15, on the same episodes.

Their cohort is not one of the named predicates. It is the episode manifest
`compute_charlson_index.py --write_cohort` writes -- the episodes for which all three features
exist -- and both arms are given that one file, 18 through `COHORT_EPISODES` in its config.
That is what puts them on identical episodes in identical order, which the paired test needs.

19. Age, Sex And Charlson Comorbidity Index, Patients With A Charlson Comorbidity Index
