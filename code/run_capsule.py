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
from itertools import combinations
from functools import reduce

import spikeinterface as si
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.benchmark import SorterStudy
from spikeinterface.benchmark.benchmark_base import _key_separator
from spikeinterface.benchmark.benchmark_tools import fit_sigmoid, sigmoid
from spikeinterface.benchmark.benchmark_plot_tools import (
    plot_run_times,
    plot_performances_ordered,
    plot_performances_vs_snr,
    plot_unit_counts,
    plot_performances_comparison
)

import matplotlib.pyplot as plt
import seaborn as sns

data_folder = Path("../data")
results_folder = Path("../results")

FIGSIZE = (12, 7)
DEBUG_CASES = None


def create_study_folders(hybrid_folder, study_base_folder, verbose=True, debug_cases=None):
    if study_base_folder.is_dir():
        shutil.rmtree(study_base_folder)
    gt_sorting_paths = [p for p in hybrid_folder.iterdir() if "gt_" in p.name]

    # Look for sorters
    possible_sorter_folders = [
        p for p in hybrid_folder.iterdir() 
        if "motion" not in p.name and p.is_dir() and "analyzer" not in p.name
    ]
    sorting_cases = []
    for possible_sorter_folder in possible_sorter_folders:
        spikesorted_folders = [p for p in possible_sorter_folder.iterdir() if p.name.startswith("spikesorted")]
        if len(spikesorted_folders) > 0:
            sorting_cases.append(possible_sorter_folder.name)
    print(f"Found {len(gt_sorting_paths)} hybrid GT datasets")
    print(f"Spike sorting cases: {sorting_cases}")

    # create datasets and cases
    if debug_cases is not None:
        print(f"Only loading {debug_cases} debug cases")
        gt_sorting_paths = sorted(gt_sorting_paths)[:debug_cases]
    else:
        gt_sorting_paths = sorted(gt_sorting_paths)

    data_by_session = {}
    for gt_sorting_path in gt_sorting_paths:
        # remove "gt_" from name
        full_case_name = gt_sorting_path.name
        case_name = full_case_name[3:]
        case_name = case_name[:case_name.find(".pkl")]
        full_case_name = case_name

        if verbose:
            print(f"\tLoading GT sorting for {case_name}")
        try:
            gt_sorting = si.load(gt_sorting_path)
        except:
            # for back-compatibility
            gt_sorting = si.load(gt_sorting_path, base_folder=data_folder)

        # check if session is present
        session_split = case_name.split("__")
        if len(session_split) == 2:
            session_name = session_split[0]
            case_name = session_split[1]
        else:
            session_name = "session1"
        if session_name not in data_by_session:
            data_by_session[session_name] = []
        session_info = dict(
            gt_sorting=gt_sorting,
            case_name=case_name,
            case_name_with_session=full_case_name
        )
        data_by_session[session_name].append(session_info)

    # create GT studies
    study_dict = {}
    levels = ["sorting_case", "stream_name", "case"]

    for session_name, session_info_list in data_by_session.items():
        if verbose:
            print(f"Instantiating study for session {session_name}")
        session_study_folder = study_base_folder / session_name
        analyzers_path = {}
        cases = {}

        for session_info in session_info_list:
            gt_sorting = session_info["gt_sorting"]
            case_name = session_info["case_name"]
            case_name_with_session = session_info["case_name_with_session"]
            analyzer_folder = hybrid_folder / f"analyzer_{case_name_with_session}"
            recording_file = hybrid_folder / f"job_{case_name_with_session}.pkl"

            case_name_split = case_name.split("_")
            stream_name = "_".join(case_name_split[:-1])
            case = case_name.split("_")[-1]

            print(f"\tLoading analyzer for {case_name}")
            analyzer = si.load(analyzer_folder, load_extensions=False)
            (session_study_folder / "sorting_analyzer").mkdir(exist_ok=True, parents=True)

            if not analyzer.has_recording():
                with open(recording_file, "rb") as f:
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

            analyzer_study_folder = session_study_folder / "sorting_analyzer" / case_name
            analyzer.save_as(format="binary_folder", folder=analyzer_study_folder)
            # we need to add the extensions folder to avoid loading in memory
            shutil.copytree(analyzer.folder / "extensions", analyzer_study_folder / "extensions")
            # reload from results (not read-only)
            analyzers_path[case_name] = str(analyzer_study_folder)

            sortings = {}
            for sorting_case in sorting_cases:
                sorter_folder = hybrid_folder / sorting_case / f"spikesorted_{case_name_with_session}"
                log_file = sorter_folder / "spikeinterface_log.json"
                if log_file.is_file():
                    with open(sorter_folder / "spikeinterface_log.json") as f:
                        log = json.load(f)
                        sorter_name = log["sorter_name"]
                        run_time = log["run_time"]

                    # only add case if sorting output is complete
                    try:
                        sorting = si.load(sorter_folder)
                        case_key = (sorting_case, stream_name, case)
                        cases[case_key] = {
                            "label": f"{sorting_case}_{case_name}",
                            "dataset": case_name,
                            "params": {
                                "sorter_name": sorter_name,
                            }
                        }
                        # copy result
                        (session_study_folder / "results").mkdir(exist_ok=True, parents=True)
                        case_path_name = _key_separator.join([str(k) for k in case_key])
                        result_folder = session_study_folder / "results" / case_path_name
                        sorting.save(folder=result_folder / "sorting")
                        # dump run time
                        with open(result_folder / "run_time.pickle", mode="wb") as f:
                            pickle.dump(run_time, f)
                        sortings[case_key] = sorting
                        # perform gt comparison and dump
                        cmp = sc.compare_sorter_to_ground_truth(
                            gt_sorting, 
                            sorting, 
                            exhaustive_gt=False,
                            match_score=0.2 # all matches below this are set to 0
                        )
                        with open(result_folder / "gt_comparison.pickle", mode="wb") as f:
                            pickle.dump(cmp, f)
                    except Exception as e:
                        print(f"\t\tFailed to load sorting case {sorting_case}:\n\n{e}")

                    
        # study metadata
        # analyzer path (local or external)
        (session_study_folder / "analyzers_path.json").write_text(json.dumps(analyzers_path, indent=4), encoding="utf8")

        info = {}
        info["levels"] = levels
        (session_study_folder / "info.json").write_text(json.dumps(info, indent=4), encoding="utf8")

        # cases is dumped to a pickle file, json is not possible because of the tuple key
        (session_study_folder / "cases.pickle").write_bytes(pickle.dumps(cases))

        if verbose:
            print(f"Creating GT study for session {session_name}")

        study = SorterStudy(session_study_folder)
        study_dict[session_name] = study

    return study_dict, sorting_cases


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
    motion_folders = [f for f in hybrid_folder.iterdir() if f.is_dir() and f.name.startswith("motion")]
    

    # Create study
    study_folder = results_folder / "gt_studies"
    study_dict, sorting_cases = create_study_folders(hybrid_folder, study_folder, debug_cases=DEBUG_CASES)
    session_names = list(study_dict.keys())

    colors = {}
    for i, sorting_case in enumerate(sorting_cases):
        colors[sorting_case] = f"C{i}"

    # plotting section
    dataframes = {}

    for session_name, study in study_dict.items():
        print(f"\nGenerating results for session {session_name}")

        print(f"\tPlotting results")
        # motion
        case_keys = list(study.cases.keys())

        benchmark_folder = figures_output_folder / "benchmark" / session_name
        benchmark_folder.mkdir(exist_ok=True, parents=True)

        levels = ["sorting_case"]

        study.set_colors(colors, levels_to_group_by=levels)

        fig_perf = plot_performances_ordered(study, levels_to_keep=levels, orientation="horizontal", figsize=FIGSIZE)
        fig_perf.savefig(benchmark_folder / "performances_ordered.pdf")

        fig_count = plot_unit_counts(study, levels_to_keep=levels, figsize=FIGSIZE)
        fig_count.savefig(benchmark_folder / "unit_counts.pdf")

        fig_run_times = plot_run_times(study, levels_to_keep=levels, figsize=FIGSIZE)
        fig_run_times.savefig(benchmark_folder / "run_times.pdf")

        if len(sorting_cases) > 1:
            fig_comparison = plot_performances_comparison(study, levels_to_keep=levels, figsize=FIGSIZE)
            fig_comparison.savefig(benchmark_folder / "comparison.pdf")

        study.compute_metrics(metric_names=["snr", "isi_violation", "rp_violation"])
        fig_snr = plot_performances_vs_snr(study, levels_to_keep=levels, orientation="horizontal", figsize=FIGSIZE)   
        fig_snr.savefig(benchmark_folder / "performance_snr.pdf")
        skip_metrics = False

        print("\tCopying dataframes")
        dataframes_folder = results_folder / "dataframes" / session_name
        dataframes_folder.mkdir(exist_ok=True, parents=True)
        unit_counts = study.get_count_units()
        unit_counts.to_csv(dataframes_folder / "unit_counts.csv")
        performances = study.get_performance_by_unit()
        performances.to_csv(dataframes_folder / "performances.csv")
        run_times = study.get_run_times()
        run_times.to_csv(dataframes_folder / "run_times.csv")
        metrics = study.get_all_metrics()
        metrics.to_csv(dataframes_folder / "metrics.csv")
    
        print("\tCopying motion folders and figures")
        for fig_file in fig_files:
            # file name is "fig-{type}"
            fig_file_split = fig_file.name.split("_")
            _, folder_name = fig_file_split[0].split("-")
            fig_name = "_".join(fig_file_split[1:])
            if len(session_names) > 1:
                if session_name not in fig_name:
                    continue
                fig_name = fig_name.replace(session_name, "")
            output_folder = figures_output_folder / folder_name / session_name
            output_folder.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(fig_file, output_folder / fig_name)

        if len(motion_folders) > 0:
            motion_output_folder = results_folder / "motion" / session_name
            motion_output_folder.mkdir(exist_ok=True, parents=True)

            for motion_folder in motion_folders:
                stream_name = "_".join(motion_folder.name.split("_")[1:])
                if len(session_names) > 1:
                    if session_name not in stream_name:
                        continue
                    stream_name = stream_name.replace(session_name, "")
                shutil.copytree(motion_folder, motion_output_folder / stream_name)

        # each dataset has the same duration per session
        dataset_keys = list(study.datasets.keys())
        recording, _ = study.datasets[dataset_keys[0]]
        session_duration = recording.get_total_duration()
        probes_info = recording.get_annotation("probes_info")
        if probes_info is not None and len(probes_info) == 1:
            probe_model_name = probes_info[0].get("model_name", "unknown")
        else:
            probe_model_name = "unknown"
        print(f"\tSession duration: {session_duration} - Probe model: {probe_model_name}")
        csvs = [p for p in dataframes_folder.iterdir() if p.suffix == ".csv"]
        for csv_file in csvs:
            df_name = csv_file.stem
            df = pd.read_csv(csv_file)
            df.loc[:, "session"] = session_name
            df.loc[:, "duration"] = session_duration
            df.loc[:, "probe"] = probe_model_name

            if df_name not in dataframes:
                dataframes[df_name] = df
            else:
                dataframes[df_name] = pd.concat([dataframes[df_name], df])

    # Aggregated results
    print("Aggregating results")
    performance_metrics = ["accuracy", "precision", "recall"]
    aggregated_results_folder = results_folder / "aggregated"
    df_units = pd.merge(dataframes["performances"], dataframes["metrics"])
    df_counts = dataframes["unit_counts"]
    df_run_times = dataframes["run_times"]

    dataframes_folder = aggregated_results_folder / "dataframes"
    dataframes_folder.mkdir(parents=True)
    for df_name, df in dataframes.items():
        df.to_csv(dataframes_folder / f"{df_name}.csv")

    # performance
    figures_folder = aggregated_results_folder / "figures"
    figures_folder.mkdir(parents=True)

    fig_perf, axes = plt.subplots(ncols=len(performance_metrics), figsize=(10,5), sharey=True)
    num_hybrid_units = int(len(df_units) / len(sorting_cases))

    for i, metric in enumerate(performance_metrics):
        df_units_sorted = df_units.sort_values(["sorting_case", metric], ascending=False)
        df_units_sorted.loc[:, "unit_index"] = np.arange(len(df_units_sorted)) % num_hybrid_units
        ax = axes[i]
        sns.lineplot(data=df_units_sorted, x="unit_index", y=metric, hue="sorting_case", ax=ax, palette=colors, lw=2.5)
        ax.set_title(metric.capitalize())
        if i > 0:
            ax.legend().remove()
        ax.set_xlabel("")
    axes[0].set_ylabel("Value", fontsize=15)
    axes[1].set_xlabel("Sorted units", fontsize=15)
    sns.despine(fig_perf)
    fig_perf.suptitle(f"# Units: {num_hybrid_units}", fontsize=18)
    fig_perf.savefig(figures_folder / f"performance_ordered.pdf")

    # unit counts and run times
    fig_counts_rt, axes = plt.subplots(ncols=2, figsize=(12, 6))
    ax_counts = axes[0]
    sns.boxenplot(data=df_counts, hue="sorting_case", y="num_well_detected", ax=ax_counts, palette=colors)
    ax_counts.axhline(10, color="gray", ls="--")
    ax_counts.set_ylabel("# Units > 80% accuracy")
    ax_counts.set_title("Well detected units per session")

    ax_run_times = axes[1]
    df_run_times.loc[:, "run_time_rel"] = df_run_times["run_times"] / df_run_times["duration"]
    sns.boxenplot(data=df_run_times, hue="sorting_case", y="run_time_rel", ax=ax_run_times, palette=colors)
    ax_run_times.set_ylabel("Runtime / Duration")
    ax_run_times.set_title("Speed")
    sns.despine(fig_counts_rt)

    fig_counts_rt.savefig(figures_folder / f"unit_counts_run_times.pdf")

    fig_snr, axes = plt.subplots(ncols=len(performance_metrics), figsize=(12, 5), sharey=True)
    for i, metric in enumerate(performance_metrics):
        ax = axes[i]
        sns.scatterplot(data=df_units, x="snr", y=metric, hue="sorting_case", ax=ax, alpha=0.5, palette=colors)
        ax.set_title(metric.capitalize())
        if i > 0:
            ax.legend().remove()
        ax.set_xlabel("")

        for i, sorting_case in enumerate(sorting_cases):
            df_sorter = df_units.query(f"sorting_case == '{sorting_case}'")
            xdata = df_sorter["snr"].values
            sort_indices = np.argsort(xdata)
            xdata = xdata[sort_indices]
            ydata = df_sorter[metric].values[sort_indices]
            p0 = [np.median(xdata), 1, 0]
            try:
                popt = fit_sigmoid(xdata, ydata, p0=p0)
                ax.plot(xdata, sigmoid(xdata, *popt), color=f"C{i}", lw=2)
            except Exception as e:
                print(f"\tFailed to fit sigmoid for {metric} - {sorting_case}:\n{e}")
    axes[0].set_ylabel("Value")
    axes[1].set_xlabel("SNR")
    sns.despine(fig_snr)

    fig_snr.suptitle(f"Performance VS SNR (# Units: {num_hybrid_units})")
    fig_snr.savefig(figures_folder / f"performance_vs_snr.pdf")

    # pairwise metric scatter
    if len(sorting_cases) > 1:
        pairs = combinations(sorting_cases, 2)
        on = ["stream_name", "case", "probe", "gt_unit_id"]
        for pair in pairs:
            sorting_case1, sorting_case2 = pair
            dfs_to_merge = [df_units.query(f"sorting_case == '{sorting_case}'") for sorting_case in pair]
            df_merged = reduce(lambda  left, right: pd.merge(left, right, on=on, how='outer'), dfs_to_merge)

            mapper = {}
            for col in df_merged:
                if "_x" in col:
                    mapper[col] = col.replace("_x", f"_{sorting_case1}")
                elif "_y" in col:
                    mapper[col] = col.replace("_y", f"_{sorting_case2}")
            df_merged = df_merged.rename(columns=mapper)

            fig_pair, axes = plt.subplots(ncols=len(performance_metrics), figsize=(12, 5), sharey=True)

            for i, metric in enumerate(performance_metrics):
                ax = axes[i]
                sns.scatterplot(data=df_merged, x=f"{metric}_{sorting_case1}", y=f"{metric}_{sorting_case2}", ax=ax, color=f"C{i}")
                ax.set_title(metric.capitalize())
                if i > 0:
                    ax.legend().remove()
                ax.set_xlabel("")
                ax.plot([0, 1],[0, 1], color="grey", ls="--", alpha=0.5)
            axes[0].set_ylabel(sorting_case2)
            axes[1].set_xlabel(sorting_case1)
            sns.despine(fig_pair)

            fig_pair.suptitle(f"{sorting_case1} vs {sorting_case2} (# Units: {num_hybrid_units})")
            fig_pair.savefig(figures_folder / f"{sorting_case1}_vs_{sorting_case2}.png", dpi=300)
    
    print("DONE!")

