""" top level run script """
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

def create_study_folder(hybrid_folder, study_folder, verbose=True, debug_cases=None):
    gt_sortings_folder = hybrid_folder / "sortings"
    recordings_folder = hybrid_folder / "recordings"

    case_names = [p.stem for p in gt_sortings_folder.iterdir()]
    
    sorters = [
        p.name for p in hybrid_folder.iterdir() 
        if p.name not in ["sortings", "recordings", "figures", "motion"] and p.is_dir()
    ]
    if verbose:
        print(f"Found {len(case_names)} recordings and {len(sorters)} sorter runs")

    # create datasets and cases
    datasets = {}
    cases = {}
    if debug_cases is not None:
        print(f"Only loading {debug_cases} debug cases")
        case_names = sorted(case_names)[:debug_cases]
    else:
        case_names = sorted(case_names)
    for case_name in case_names:
        if verbose:
            print(f"\tLoading case {case_name}")
        case_name_split = case_name.split("_")
        stream_name = "_".join(case_name_split[:-2])
        complexity = case_name.split("_")[-2]
        case = case_name.split("_")[-1]

        if verbose:
            print(f"\t\tLoading GT sorting")
        sorting = si.load_extractor(
            gt_sortings_folder / f"{case_name}.pkl",
            base_folder=data_folder
        )
        with open(recordings_folder / f"job_{case_name}.pkl", "rb") as f:
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
                datasets[case_name] = (recording, sorting)
                cases[(sorter, stream_name, complexity, case)] = {
                    "label": f"{sorter_name}_{case_name}",
                    "dataset": case_name,
                    "run_sorter_params": {
                        "sorter_name": sorter_name,
                    }
                }
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    if verbose:
        print(f"Creating GT study")
    study = sc.GroundTruthStudy.create(study_folder, datasets=datasets, cases=cases,
                                       levels=["sorter", "stream", "complexity", "case"])

    # copy sortings
    if verbose:
        print(f"Copying and loading sorted data")
    for key, case_dict in cases.items():
        target_sorting_folder = study_folder / "sortings" / study.key_to_str(key)
        sorter, stream_name_abbr, complexity, case = key
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
    si.set_global_job_kwargs(n_jobs=-1, progress_bar=False)

    # find hybrid folder
    hybrid_folder = None
    if (data_folder / "sortings").is_dir():
        hybrid_folder = data_folder

    if hybrid_folder is None:
        # look for subfolder
        subfolders = [p for p in data_folder.iterdir() if p.is_dir()]
        for p in subfolders:
            if p.is_dir() and (p / "sortings").is_dir():
                hybrid_folder = p
                break

    assert hybrid_folder is not None, "Couldn't find hybrid folder"
    print(f"Hybrid folder: {hybrid_folder}")

    for folder_name in ("figures", "motion"):
        if (hybrid_folder / folder_name).is_dir():
            shutil.copytree(hybrid_folder / folder_name, results_folder / folder_name)

    study_folder = results_folder / "gt_study"
    # create study
    study = create_study_folder(hybrid_folder, study_folder, debug_cases=DEBUG_CASES)

    print(f"\tRunning comparisons")
    study.run_comparisons()

    print(f"\tCreating GT sorting analyzers")
    study.create_sorting_analyzer_gt()
    print(f"\tComputing metrics")
    study.compute_metrics(metric_names=["snr"])

    # plotting section
    print(f"\nPlotting results")
    figures_folder = results_folder / "figures"
    figures_folder.mkdir(exist_ok=True)
    # motion
    case_keys = list(study.cases.keys())

    if (hybrid_folder / "motion").is_dir():
        print("\tRaster maps")
        rasters_folder = figures_folder / "rasters"
        rasters_folder.mkdir(exist_ok=True)
        
        for case_key in case_keys:
            print(f"\t\tCase: {case_key}")
            stream_name = case_key[1]
            motion_info = spre.load_motion_info(hybrid_folder / "motion" / stream_name)
            analyzer_gt = study.get_sorting_analyzer(case_key)
            recording = analyzer_gt.recording
            analyzer_gt.compute("spike_locations", save=False)
            print(f"\t\tPlotting")
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

    print("\tPerformances")
    benchmark_folder = figures_folder / "benchmark"
    benchmark_folder.mkdir(exist_ok=True)

    w_perf = sw.plot_study_performances(study, levels=("sorter", "complexity"), figsize=FIGSIZE)
    w_snr = sw.plot_study_performances(study, levels=("sorter", "complexity"), mode="snr", figsize=FIGSIZE)   
    w_count = sw.plot_study_unit_counts(study, levels=("sorter", "complexity"), figsize=FIGSIZE)
    w_run_times = sw.plot_study_run_times(study, levels=("sorter", "complexity"), figsize=FIGSIZE)

    w_perf.figure.savefig(benchmark_folder / "performances_ordered.pdf")
    w_snr.figure.savefig(benchmark_folder / "performance_snr.pdf")
    w_count.figure.savefig(benchmark_folder / "unit_counts.pdf")
    w_run_times.figure.savefig(benchmark_folder / "run_times.pdf")

    print("\tCopying dataframes")
    dataframes_folder = results_folder / "dataframes"
    dataframes_folder.mkdir(exist_ok=True)
    unit_counts = study.get_count_units()
    unit_counts.to_csv(dataframes_folder / "unit_counts.csv")
    performances = study.get_performance_by_unit()
    performances.to_csv(dataframes_folder / "performances.csv")
    metrics = study.get_metrics()
    metrics.to_csv(dataframes_folder / "metrics.csv")
    run_times = study.get_run_times()
    metrics.to_csv(dataframes_folder / "run_times.csv")
    
    print("DONE!")

