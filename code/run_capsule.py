""" top level run script """
import warnings

warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import shutil

import spikeinterface as si
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.benchmark import SorterStudy
from spikeinterface.benchmark.benchmark_base import _key_separator
from spikeinterface.benchmark.benchmark_plot_tools import (
    plot_run_times,
    plot_performances_ordered,
    plot_performances_vs_snr,
    plot_unit_counts,
    plot_performances_comparison
)

data_folder = Path("../data")
results_folder = Path("../results")

FIGSIZE = (12, 7)
DEBUG_CASES = None


def create_study_folder(hybrid_folder, study_folder, verbose=True, debug_cases=None):
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    gt_sorting_paths = [p for p in hybrid_folder.iterdir() if "gt_" in p.name]

    sorters = [
        p.name for p in hybrid_folder.iterdir() 
        if "motion" not in p.name and p.is_dir() and "analyzer" not in p.name
    ]
    if verbose:
        print(f"Found {len(gt_sorting_paths)} recordings and {len(sorters)} sorter runs")

    # create datasets and cases
    datasets = {}
    cases = {}
    if debug_cases is not None:
        print(f"Only loading {debug_cases} debug cases")
        gt_sorting_paths = sorted(gt_sorting_paths)[:debug_cases]
    else:
        gt_sorting_paths = sorted(gt_sorting_paths)

    analyzers_path = {}
    for gt_sorting_path in gt_sorting_paths:
        case_name = gt_sorting_path.name
        # remove "gt_" from name
        case_name = case_name[3:]
        case_name = case_name[:case_name.find(".pkl")]
        if verbose:
            print(f"\tLoading case {case_name}")
        case_name_split = case_name.split("_")
        stream_name = "_".join(case_name_split[:-1])
        case = case_name.split("_")[-1]

        if verbose:
            print(f"\t\tLoading GT sorting")
        try:
            gt_sorting = si.load(gt_sorting_path)
        except:
            # for back-compatibility
            gt_sorting = si.load(gt_sorting_path, base_folder=data_folder)

        analyzer_folder = hybrid_folder / f"analyzer_{case_name}"
        should_add_subfolder = False

        # in this case we have to 
        if analyzer_folder.parent.resolve() == data_folder.resolve():
            should_add_subfolder = True

        if analyzer_folder.is_dir():
            print(f"\t\tLoading analyzer")
            analyzer = si.load(analyzer_folder, load_extensions=False)
            # copy analyzer to study folder
            (study_folder / "sorting_analyzer").mkdir(exist_ok=True, parents=True)

            if not analyzer.has_recording():
                print(f"\t\tAnalyzer couldn't load recording. Loading from .pkl")
                with open(hybrid_folder / f"job_{case_name}.pkl", "rb") as f:
                    dump_dict = pickle.load(f)
                    recording_dict = dump_dict["recording_dict"]
                    if verbose:
                        print(f"\t\tLoading hybrid recording")
                    try:
                        recording = si.load(recording_dict, base_folder=data_folder)
                    except Exception as e:
                        print(f"Analyzer couldn't load recording. Trying remapping:\n{e}")
                        from spikeinterface.core.core_tools import SIJsonEncoder, recursive_path_modifier
                        raw_folder_names = [
                            p.name for p in data_folder.iterdir() 
                            if "ecephys" in p.name or "behavior" in p.name
                        ]
                        assert len(raw_folder_names) == 1
                        raw_folder_name = raw_folder_names[0]
                        f = lambda x: x.replace("ecephys_session", raw_folder_name)
                        recording_dict = recursive_path_modifier(recording_dict, f)
                        recording = si.load(recording_dict, base_folder=data_folder)
                if analyzer is not None:
                    analyzer._recording = recording

            analyzer_study_folder = study_folder / "sorting_analyzer" / case_name
            analyzer.save_as(format="binary_folder", folder=analyzer_study_folder)
            # we need to add the extensions folder to avoid loading in memory
            shutil.copytree(analyzer.folder / "extensions", analyzer_study_folder / "extensions")
            # reload from results (not read-only)
            analyzers_path[case_name] = str(analyzer_study_folder)

        levels = ["sorter", "stream_name", "case"]
        sortings = {}
        for sorter in sorters:
            sorter_folder = hybrid_folder / sorter / f"spikesorted_{case_name}"
            log_file = sorter_folder / "spikeinterface_log.json"
            if log_file.is_file():
                with open(sorter_folder / "spikeinterface_log.json") as f:
                    log = json.load(f)
                    sorter_name = log["sorter_name"]
                    run_time = log["run_time"]
                datasets[case_name] = analyzer
                # only add case if sorting output is complete
                try:
                    sorting = si.load(sorter_folder)
                    case_key = (sorter, stream_name, case)
                    cases[case_key] = {
                        "label": f"{sorter_name}_{case_name}",
                        "dataset": case_name,
                        "params": {
                            "sorter_name": sorter_name,
                        }
                    }
                    # copy result
                    (study_folder / "results").mkdir(exist_ok=True, parents=True)
                    case_path_name = _key_separator.join([str(k) for k in case_key])
                    result_folder = study_folder / "results" / case_path_name
                    sorting.save(folder=result_folder / "sorting")
                    # dump run time
                    with open(result_folder / "run_time.pickle", mode="wb") as f:
                        pickle.dump(run_time, f)
                    sortings[case_key] = sorting
                    # perform gt comparison and dump
                    cmp = sc.compare_sorter_to_ground_truth(gt_sorting, sorting, exhaustive_gt=False)
                    with open(result_folder / "gt_comparison.pickle", mode="wb") as f:
                        pickle.dump(cmp, f)
                except Exception as e:
                    print(f"\t\t\tFailed to load sorter {sorter}:\n\n{e}")

                    
        # study metadata
        # analyzer path (local or external)
        (study_folder / "analyzers_path.json").write_text(json.dumps(analyzers_path, indent=4), encoding="utf8")

        info = {}
        info["levels"] = levels
        (study_folder / "info.json").write_text(json.dumps(info, indent=4), encoding="utf8")

        # cases is dumped to a pickle file, json is not possible because of the tuple key
        (study_folder / "cases.pickle").write_bytes(pickle.dumps(cases))

    if verbose:
        print(f"Creating GT study")

    study = SorterStudy(study_folder)

    return study, sorters


if __name__ == "__main__":
    si.set_global_job_kwargs(n_jobs=-1, progress_bar=False)

    # find hybrid folder
    hybrid_folder = None
    gt_files = [p for p in data_folder.iterdir() if "gt_" in p.name]
    if len(gt_files) > 0:
        hybrid_folder = data_folder

    if hybrid_folder is None:
        # look for subfolder
        subfolders = [p for p in data_folder.iterdir() if p.is_dir()]
        for subfolder in subfolders:
            gt_files = [p for p in subfolder.iterdir() if "gt_" in p.name]
            if len(gt_files) > 0:
                hybrid_folder = subfolder
                break

    assert hybrid_folder is not None, "Couldn't find hybrid folder"
    print(f"Hybrid folder: {hybrid_folder}")

    # Copy figures and motion folders
    figures_output_folder = results_folder / "figures"
    figures_output_folder.mkdir(exist_ok=True)
    fig_files = [f for f in hybrid_folder.iterdir() if f.is_file() and f.name.startswith("fig-")]
    for fig_file in fig_files:
        # file name is "fig-{type}"
        fig_file_split = fig_file.name.split("_")
        _, folder_name = fig_file_split[0].split("-")
        (figures_output_folder / folder_name).mkdir(exist_ok=True)
        fig_name = "_".join(fig_file_split[1:])
        shutil.copyfile(fig_file, figures_output_folder / folder_name / fig_name)

    motion_folders = [f for f in hybrid_folder.iterdir() if f.is_dir() and f.name.startswith("motion")]
    if len(motion_folders) > 0:
        motion_output_folder = results_folder / "motion"
        motion_output_folder.mkdir(exist_ok=True)

        for motion_folder in motion_folders:
            stream_name = "_".join(motion_folder.name.split("_")[1:])
            shutil.copytree(motion_folder, motion_output_folder / stream_name)

    # Create study
    study_folder = results_folder / "gt_study"
    study, sorter_names = create_study_folder(hybrid_folder, study_folder, debug_cases=DEBUG_CASES)

    # plotting section
    print(f"\nPlotting results")
    # motion
    case_keys = list(study.cases.keys())

    benchmark_folder = figures_output_folder / "benchmark"
    benchmark_folder.mkdir(exist_ok=True)

    levels = ["sorter"]

    colors = {}
    for i, sorter in enumerate(sorter_names):
        colors[sorter] = f"C{i}"

    study.set_colors(colors, levels_to_group_by=levels)

    fig_perf = plot_performances_ordered(study, levels_to_keep=levels, orientation="horizontal", figsize=FIGSIZE)
    fig_perf.savefig(benchmark_folder / "performances_ordered.pdf")

    fig_count = plot_unit_counts(study, levels_to_keep=levels, figsize=FIGSIZE)
    fig_count.savefig(benchmark_folder / "unit_counts.pdf")

    fig_run_times = plot_run_times(study, levels_to_keep=levels, figsize=FIGSIZE)
    fig_run_times.savefig(benchmark_folder / "run_times.pdf")

    fig_comparison = plot_performances_comparison(study, levels_to_keep=levels, figsize=FIGSIZE)
    fig_comparison.savefig(benchmark_folder / "comparison.pdf")

    study.compute_metrics(metric_names=["snr"])
    fig_snr = plot_performances_vs_snr(study, levels_to_keep=levels, orientation="horizontal", figsize=FIGSIZE)   
    fig_snr.savefig(benchmark_folder / "performance_snr.pdf")
    skip_metrics = False

    print("Copying dataframes")
    dataframes_folder = results_folder / "dataframes"
    dataframes_folder.mkdir(exist_ok=True)
    unit_counts = study.get_count_units()
    unit_counts.to_csv(dataframes_folder / "unit_counts.csv")
    performances = study.get_performance_by_unit()
    performances.to_csv(dataframes_folder / "performances.csv")
    run_times = study.get_run_times()
    run_times.to_csv(dataframes_folder / "run_times.csv")
    metrics = study.get_all_metrics()
    metrics.to_csv(dataframes_folder / "metrics.csv")
    
    print("DONE!")

