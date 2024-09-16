""" top level run script """
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
import shutil

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

data_folder = Path("../data")
results_folder = Path("../results")

FIGSIZE = (12, 7)
DEBUG_CASES = None

# Define argument parser
parser = argparse.ArgumentParser(description="Evaluate hybrid datasets")

skip_analyzer_group = parser.add_mutually_exclusive_group()
skip_analyzer_help = "Whether to skip sorting analyzer computation and related analysis"
skip_analyzer_group.add_argument("--skip-analyzer", action="store_true", help=skip_analyzer_help)
skip_analyzer_group.add_argument("static_skip_analyzer", nargs="?", default="false", help=skip_analyzer_help)


def create_study_folder(hybrid_folder, study_folder, verbose=True, debug_cases=None):
    gt_sorting_paths = [p for p in hybrid_folder.iterdir() if "gt_" in p.name]

    sorters = [
        p.name for p in hybrid_folder.iterdir() 
        if "motion" not in p.name and p.is_dir()
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
        gt_sorting = si.load_extractor(
            gt_sorting_path,
            base_folder=data_folder
        )
        with open(hybrid_folder / f"job_{case_name}.pkl", "rb") as f:
            dump_dict = pickle.load(f)
            recording_dict = dump_dict["recording_dict"]
            if verbose:
                print(f"\t\tLoading hybrid recording")
            try:
                recording = si.load_extractor(recording_dict, base_folder=data_folder)
            except:
                from spikeinterface.core.core_tools import SIJsonEncoder, recursive_path_modifier
                raw_folder_names = [
                    p.name for p in data_folder.iterdir() 
                    if "ecephys" in p.name or "behavior" in p.name
                ]
                assert len(raw_folder_names) == 1
                raw_folder_name = raw_folder_names[0]
                f = lambda x: x.replace("ecephys_session", raw_folder_name)
                recording_dict = recursive_path_modifier(recording_dict, f)
                recording = si.load_extractor(recording_dict, base_folder=data_folder)

        
        for sorter in sorters:
            sorter_folder = hybrid_folder / sorter / f"spikesorted_{case_name}"
            log_file = sorter_folder / "spikeinterface_log.json"
            if log_file.is_file():
                with open(sorter_folder / "spikeinterface_log.json") as f:
                    log = json.load(f)
                    sorter_name = log["sorter_name"]
                datasets[case_name] = (recording, gt_sorting)
                # only add case if sorting output is complete
                try:
                    sorting = si.load_extractor(sorter_folder)
                    cases[(sorter, stream_name, case)] = {
                        "label": f"{sorter_name}_{case_name}",
                        "dataset": case_name,
                        "run_sorter_params": {
                            "sorter_name": sorter_name,
                        }
                    }
                except:
                    print(f"\t\t\tFailed to load sorter {sorter}")

    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    if verbose:
        print(f"Creating GT study")
    study = sc.GroundTruthStudy.create(study_folder, datasets=datasets, cases=cases,
                                       levels=["sorter", "stream", "case"])

    # copy sortings
    if verbose:
        print(f"Copying and loading sorted data")
    for key, case_dict in cases.items():
        target_sorting_folder = study_folder / "sortings" / study.key_to_str(key)
        sorter, stream_name_abbr, case = key
        case_name = case_dict["dataset"]
        existing_sorter_folder = hybrid_folder / sorter / f"spikesorted_{case_name}"

        shutil.copytree(existing_sorter_folder, target_sorting_folder)
        log_file = study_folder / "sortings" / "run_logs" / f"{study.key_to_str(key)}.json"

        sorting = si.load_extractor(target_sorting_folder)
        study.sortings[key] = sorting

        # copy logs
        existing_log_file = existing_sorter_folder / "spikeinterface_log.json"
        if existing_log_file.is_file():
            shutil.copyfile(existing_log_file, log_file)

    return study


if __name__ == "__main__":
    args = parser.parse_args()

    SKIP_ANALYZER = args.skip_analyzer or args.static_skip_analyzer == "true"

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
    study = create_study_folder(hybrid_folder, study_folder, debug_cases=DEBUG_CASES)

    print(f"\tRunning comparisons")
    study.run_comparisons()

    if not SKIP_ANALYZER:
        print(f"\tCreating GT sorting analyzers")
        study.create_sorting_analyzer_gt()
        print(f"\tComputing metrics")
        study.compute_metrics(metric_names=["snr"])
    else:
        print(f"\tSkipping computation of sorting analyzers and metrics")

    # plotting section
    print(f"\nPlotting results")
    # motion
    case_keys = list(study.cases.keys())

    if not SKIP_ANALYZER:
        if len(motion_folders) > 0:
            print("\tRaster maps")
            rasters_folder = figures_output_folder / "rasters"
            rasters_folder.mkdir(exist_ok=True)
            
            for case_key in case_keys:
                print(f"\t\tCase: {study.key_to_str(case_key)}")
                stream_name = case_key[1]
                motion_info = spre.load_motion_info(hybrid_folder / f"motion_{stream_name}")
                analyzer_gt = study.get_sorting_analyzer(case_key)
                recording = analyzer_gt.recording
                analyzer_gt.compute("spike_locations", save=False)
                w = sw.plot_drift_raster_map(
                    peaks=motion_info["peaks"],
                    peak_locations=motion_info["peak_locations"],
                    recording=recording,
                    cmap="Greys_r",
                    scatter_decimate=10,
                )
                ax = w.ax
                analyzer = study
                _ = sw.plot_drift_raster_map(
                    sorting_analyzer=analyzer_gt,
                    color_amplitude=False,
                    color="b",
                    scatter_decimate=10,
                    ax=w.ax
                )
                ax.set_title(case_key)

                motion = motion_info["motion"]
                _ = ax.plot(
                    motion.temporal_bins_s[0],
                    motion.spatial_bins_um + motion.displacement[0],
                    color="y",
                    alpha=0.5
                )
                w.figure.savefig(rasters_folder / f"{case_key}.png", dpi=300)
        else:
            print("\tNo motion found. Skipping raster maps")
    else:
        print(f"\tSkipping drift maps plots")

    print("\tPerformances")
    benchmark_folder = figures_output_folder / "benchmark"
    benchmark_folder.mkdir(exist_ok=True)

    levels = "sorter"

    w_perf = sw.plot_study_performances(study, levels=levels, figsize=FIGSIZE)
    w_perf.figure.savefig(benchmark_folder / "performances_ordered.pdf")

    w_count = sw.plot_study_unit_counts(study, levels=levels, figsize=FIGSIZE)
    w_count.figure.savefig(benchmark_folder / "unit_counts.pdf")

    w_run_times = sw.plot_study_run_times(study, levels=levels, figsize=FIGSIZE)
    w_run_times.figure.savefig(benchmark_folder / "run_times.pdf")

    if not SKIP_ANALYZER:
        w_snr = sw.plot_study_performances(study, levels=levels, mode="snr", figsize=FIGSIZE)   
        w_snr.figure.savefig(benchmark_folder / "performance_snr.pdf")


    print("\tCopying dataframes")
    dataframes_folder = results_folder / "dataframes"
    dataframes_folder.mkdir(exist_ok=True)
    unit_counts = study.get_count_units()
    unit_counts.to_csv(dataframes_folder / "unit_counts.csv")
    performances = study.get_performance_by_unit()
    performances.to_csv(dataframes_folder / "performances.csv")
    run_times = study.get_run_times()
    run_times.to_csv(dataframes_folder / "run_times.csv")
    if not SKIP_ANALYZER:
        metrics = study.get_metrics()
        metrics.to_csv(dataframes_folder / "metrics.csv")
    
    print("DONE!")

