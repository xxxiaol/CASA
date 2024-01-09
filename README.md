<img align="left" width="80" height="80" src="docs/static/images/casa.png" alt="icon">

# CASA: Causality-driven Argument Sufficiency Assessment

[<a href="https://xxxiaol.github.io/CASA/"> Project Website </a>]

![framework](docs/static/images/framework.png#pic_center)
## The CASA Framework
We first extract the premise and conclusion from a given argument, then sample contexts that meet the conditions, make interventions on the contexts, and finally estimate the probability of the conclusion for each unit.

The code is provided in the `code/` folder. 
 - `claim_extraction.py`: extract the premise and conclusion from a given argument
 - `context_sampling.py`: sample contexts that are consistent with ¬premise and ¬conclusion
 - `revision_under_intervention.py`: make interventions on the contexts to meet the premise
 - `probability_estimation.py`: estimate the probability of the conclusion for each unit

## Data
We provide the data we experiment with in the 'data/' folder.
 - `Bigbench-LFD.json`: the informal statements from the [BIG-bench logical fallacy detection](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/logical_fallacy_detection) task
 - `Climate.json`: arguments from climate change articles fact-checked by climate scientists ([the original dataset](https://github.com/Tariq60/fallacy-detection/tree/master/data/climate))
 - `AAE_sampled100.json`: randomly sampled arguments from the [Argument-Annotated Essays](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) dataset