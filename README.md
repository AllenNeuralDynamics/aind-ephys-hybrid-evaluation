# Evaluate spike sorting performance on hybrid study
## aind-ephys-hybrid-evaluation

Evaluate the performance of spike sorting options on hybrid data.

## Description

This capsule aggregates results from multiple spike sorting jobs and produces results dataframes and figures for further analysis.

## Inputs

The `data/` folder must include the session raw data, the `flattened` folder produced by the 
[aind-ephys-hybrid-generation](https://github.com/AllenNeuralDynamics/aind-ephys-hybrid-generation) capsule, 
and a list of additional folders with the spike sorting outputs from different branches to evaluate.
These additional folder names will be used as `sorter_name` (e.g., `ks25`, `ks4`, `sc2`).

### Parameters

The `code/run` script takes no parameters.

### Output

The output of this capsule includes the following folders:

* `dataframes`: aggregated dataframes in CSV format for: run times, unit counts, unit performances, and metrics.
* `motion`: the estimated motion folder for each input recording (generated by the [[aind-ephys-job-dispatch](https://github.com/AllenNeuralDynamics/aind-ephys-job-dispatch)])
* `figures`: generated figures for benchmarks, raster maps, motion, and hybrid templates
* `gt_study`: the output folder for the `spikeinterface.benchmark.SorterStudy`