### Overview
We explore the calibration of LLMs on Code Summarization: can an LLM's token probabilities be leveraged to produce a well-calibrated likelihood of whether the generated summary is similar to what a developer would've wrote for the same code.

This repository stores code to replicate experiments in the paper [Calibration of Large Language Models on Code
Summarization](https://arxiv.org/pdf/2404.19318) (FSE 2025).

Data and some results files are omitted due to their large size. All data is accessible at: https://zenodo.org/records/11646569

If you have difficulty working with the code or need additional details, let me know: yuvivirk344 at gmail dot com

### Where is what:
- *Thresholds on similarity metrics*: 
  - **thresholds.py**: Measuring agreement to human judgements of similarity
  - **best_metric_thresh.ipynb**: Searching for the best similarity metrics and thresholds
- *Generating code summaries*: 
  - **generate_contexts/**: Generating few-shot and ASAP contexts
  - **summarization_inference.py**: Prompting LLMs to generate code summaries
- *RQ1 & RQ2*: Raw and scaled calibration of LLMs on code summarization
  - **calibration_metrics.py**: Calculate calibration metrics and plots for raw and scaled LLM token probabilites
  - **reliability_plot.py**: Plotting function for reliability diagrams
- *RQ3*: Relationship between token position and confidence
  - **logit_vs_position_analysis.ipynb**: Plotting token position vs. distribution of token probabilties at position
  - **calibration_vs_token_cutoff.py**: Computing calibration metrics when only using first *k* tokens to measure confidence.
  - **brier_score_significant_testing.py**: Statistical test of significance for improvement in brier score 
- *Additional experiments*:
  - **reflective_logit_analysis.py**: Yes-No propmting LLMs whether the summary is similar to what a developer would write.
  - **self_reflection_prompting.py**: Prompting LLMs to generate scores or probabiltiies directly the summary is similar to what a developer would write.
  - **verbalized_confidence_analysis.py**: Analysis of self-reflection results
  - **benchmarking.py**: Measuring performance of LLMs on code summarization with different similarity metrics

### Navigating the data:
The **data** directory contains 3 subdirectories: *haque_et_al*, *Java*, *Python*. 
- The *haque_et_al* subdirectory contains all human evaluation data used (from Haque, Z.Eberhart, A.Bansal, and C.McMillan, “Semantic similarity metrics for evaluating source code summarization"). 
- Each language directory contains 3 subdirectories: *metrics_results*, *model_outputs*, and *prompting_data*.
  - *prompting_data* contains all prompts used for each prompting method and the data used to construct the prompts.
  - *model_outputs* contains the generated output per model per prompting method.
  - *metrics_results* contains the calculated summary evaluation metrics for all model outputs.

The **results** directory contains the raw results data in JSON format for benchmarks, rank correlations, raw and rescaled calibration evaluations, thresholds, and corresponding figures across multiple similarity metrics e.g. SentenceBERT(sbert).