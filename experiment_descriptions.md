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
