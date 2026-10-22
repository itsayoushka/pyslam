"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import csv
import re
import shlex
import sys
import numpy as np
import math

import os
import glob
from datetime import datetime
import shutil
import subprocess
import yaml
import json
import concurrent.futures
from datetime import timezone

from pyslam.utilities.data_management import merge_dicts
from pyslam.utilities.logging import Printer
from pyslam.utilities.run_command import run_command_async, run_command_sync
import traceback
from pathlib import Path

try:
    import hjson
except ImportError:
    print(
        "hjson not installed. Please install it with: pip install hjson"
    )  # why hjson? because it allows comments in the json file and it does not complain about the trailing commas


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kSettingsFolder = kRootFolder + "/settings"
kResultsFolder = kRootFolder + "/results"
kEvaluationFolder = kRootFolder + "/pyslam/evaluation"


date_time_now = datetime.now()
date_time_now_string = date_time_now.strftime("%Y_%m_%d-%H_%M_%S")
eval_path_prefix = kResultsFolder + "/eval_" + date_time_now_string

log_folder_name = "logs"
template_entry_regex = re.compile(r"%(\w+)%")


def get_git_short_hash(repo_dir=None):
    try:
        repo_dir = repo_dir or Path(__file__).resolve().parent
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def replace_template_entries(string, vocabulary):
    try:
        # Step 1: Replace known entries
        replaced_string = re.sub(
            template_entry_regex,
            lambda match: vocabulary.get(match.group(1), match.group(0)),
            string,
        )
        # Step 2: Remove all *unmatched* placeholders
        cleaned_string = re.sub(template_entry_regex, "Not found", replaced_string)
        return cleaned_string
    except Exception as e:
        print("Error in replace_template_entries: " + str(e))
        print(traceback.format_exc())
        return string


def remove_unused_datasets(yaml_dict):
    # Step 1: Get selected dataset type (e.g., "KITTI_DATASET")
    selected_dataset = yaml_dict.get("DATASET", {}).get("type", None)

    if not selected_dataset:
        raise ValueError("No dataset type specified in DATASET.type")

    # Step 2: Remove all other *_DATASET entries
    keys_to_delete = [
        key for key in yaml_dict if key.endswith("_DATASET") and key != selected_dataset
    ]

    for key in keys_to_delete:
        del yaml_dict[key]

    return yaml_dict


def update_yaml_dict(yaml_dict, vocabulary):
    """
    Recursively update a YAML dictionary with override semantics.

    - If both existing and incoming values are dicts, merge recursively.
    - Otherwise, replace the existing value with the incoming one.
    """
    for key, value in vocabulary.items():
        if key in yaml_dict and isinstance(yaml_dict[key], dict) and isinstance(value, dict):
            update_yaml_dict(yaml_dict[key], value)
        else:
            yaml_dict[key] = value
    return yaml_dict


# A class to manage the evaluation of SLAM algorithms.
# It runs a set of SLAM presets on a set of datasets, and then collects and aggregates the results.
# A final table is created with the results.
# See test/evaluation/test_run_evaluations.py
class SlamEvaluationManager:
    RUN_METRICS_HEADERS = [
        "timestamp_utc",
        "dataset_name",
        "preset_name",
        "run_idx",
        "seed",
        "num_threads",
        "sensor_type",
        "fps",
        "n_frames_total (M)",
        "n_frames_tracked (m)",
        "coverage (C=m/M)",
        "relocalizing_frames",
        "relocalizing_ratio",
        "first_relocalizing_frame",
        "relocalization_tail_frames",
        "success (0/1)",
        "failure_reason",
        "time_to_failure_frame",
        "time_to_failure_seconds",
        "align_type",
        "ate_rmse",
        "ate_mean",
        "ate_median",
        "ate_std",
        "ate_iqr",
        "ate_p90",
        "ate_p95",
        "ate_max",
        "rpe_delta_frames",
        "rpe_trans_rmse",
        "rpe_trans_mean",
        "rpe_trans_median",
        "rpe_trans_std",
        "rpe_trans_iqr",
        "rpe_trans_p90",
        "rpe_trans_p95",
        "rpe_trans_max",
        "rpe_rot_rmse",
        "rpe_rot_mean",
        "rpe_rot_median",
        "rpe_rot_std",
        "rpe_rot_iqr",
        "rpe_rot_p90",
        "rpe_rot_p95",
        "rpe_rot_max",
        "runtime_seconds",
        "trajectories_folder",
        "plots_folder",
    ]
    SUMMARY_METRICS_HEADERS = [
        "dataset_name",
        "preset_name",
        "N (total runs)",
        "k (success count)",
        "success_rate (k/N)",
        "ci95_wilson_low",
        "ci95_wilson_high",
        "runs_with_relocalization",
        "relocalization_rate",
        "ate_rmse_median_success",
        "ate_rmse_std_success",
        "ate_rmse_iqr_success",
        "ate_rmse_cv_success",
        "rpe_trans_median_success",
        "rpe_trans_std_success",
        "rpe_trans_iqr_success",
        "rpe_rot_median_success",
        "rpe_rot_std_success",
        "rpe_rot_iqr_success",
        "coverage_median",
        "coverage_iqr",
        "time_to_failure_median_failure",
        "time_to_failure_iqr_failure",
        "hmi_ratio_median",
        "hmi_ratio_iqr",
    ]

    def __init__(self, evaluation_config_path, template_config_file_path, just_create_report=False):
        if not os.path.exists(evaluation_config_path):
            raise Exception(f"Config file {evaluation_config_path} does not exist")
        self.evaluation_config_path = evaluation_config_path
        print("Loading JSON config file: " + self.evaluation_config_path)
        with open(self.evaluation_config_path, "r") as f:
            json_data = hjson.load(f)
        self.json_data = json_data
        print("JSON evaluation config: ", json.dumps(json_data, indent=4))

        # read the template config file
        print(f"Template config file path: {template_config_file_path}")
        if not os.path.exists(template_config_file_path):
            raise Exception(f"Template config file {template_config_file_path} does not exist")
        with open(template_config_file_path, "r") as f:
            self.template_config = f.read()

        self.presets = []
        self.datasets = []
        self.metrics = {}  # metrics[preset][dataset][iteration][metric_name] -> metric_value

        self.output_path = os.path.abspath(os.path.join(eval_path_prefix, json_data["output_path"]))
        if not just_create_report:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        self.dataset_base_path = json_data["dataset_base_path"]
        self.dataset_type = json_data["dataset_type"]
        self.sensor_type = json_data["sensor_type"]
        self.number_of_runs_per_dataset = json_data["number_of_runs_per_dataset"]
        self.preset_common_parameters = (
            json_data["common_parameters"] if "common_parameters" in json_data else {}
        )
        self.saved_trajectory_format_type = json_data["saved_trajectory_format_type"]
        self.seed = json_data.get("seed", None)
        self.seed_per_run = bool(json_data.get("seed_per_run", False))
        self.max_run_seconds = self._safe_int(json_data.get("max_run_seconds"))
        self.plot_formats = self._parse_plot_formats(json_data.get("plot_formats", "png"))
        self.export_frame_logs = bool(json_data.get("export_frame_logs", False))
        self.had_run_failures = False

        if not just_create_report:
            shutil.copy(self.evaluation_config_path, self.output_path)

        self.presets = [preset for preset in json_data["presets"]]
        self.datasets = [dataset for dataset in json_data["datasets"]]

    def configure_benchmark_options(self, plot_formats="png,pdf", export_frame_logs=True):
        self.plot_formats = self._parse_plot_formats(plot_formats)
        self.export_frame_logs = bool(export_frame_logs)

    def configure_smoke_test(self, max_frames=20):
        if not self.datasets:
            raise ValueError("Cannot configure smoke test: no datasets in evaluation config")
        if not self.presets:
            raise ValueError("Cannot configure smoke test: no presets in evaluation config")
        self.datasets = [dict(self.datasets[0])]
        self.datasets[0]["max_frames"] = int(max_frames)
        self.presets = [self.presets[0]]
        self.json_data["datasets"] = self.datasets
        self.json_data["presets"] = self.presets
        self.number_of_runs_per_dataset = 1
        self.json_data["number_of_runs_per_dataset"] = 1
        Printer.yellow(
            f"Smoke-test mode: one dataset, one preset, max_frames={int(max_frames)}"
        )

    def _yaml_scalar(self, value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        # JSON string syntax is valid YAML and safely quotes wildcards such as *.png.
        return json.dumps(str(value))

    def _print_log_tail(self, log_file, max_lines=80):
        if not log_file or not os.path.exists(log_file):
            Printer.red(f"Run log is missing: {log_file}")
            return
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            tail = "".join(lines[-max_lines:]).rstrip()
            Printer.red(f"Last {min(max_lines, len(lines))} lines from {log_file}:")
            print(tail)
        except Exception as e:
            Printer.red(f"Could not read failed run log {log_file}: {e}")

    def _validate_generated_config(self, config_dict, config_file_path):
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Generated config is not valid YAML: {config_file_path}\n{e}") from e

        dataset_type = config_dict.get("DATASET", {}).get("type")
        if dataset_type != "FOLDER_DATASET":
            return
        ds = config_dict.get("FOLDER_DATASET", {})
        base_path = os.path.expanduser(str(ds.get("base_path", "")))
        name = str(ds.get("name", ""))
        settings_path = os.path.expanduser(str(ds.get("settings", "")))
        timestamps = ds.get("timestamps")
        groundtruth_file = ds.get("groundtruth_file")
        full_glob = os.path.join(base_path, name)

        errors = []
        if not os.path.isdir(base_path):
            errors.append(f"dataset base_path does not exist or is not a directory: {base_path}")
            matched_images = []
        else:
            matched_images = sorted(glob.glob(full_glob))
            if len(matched_images) == 0:
                errors.append(
                    "image glob matched no files: "
                    f"base_path={base_path}, name={name}, full_glob={full_glob}"
                )
        if not os.path.exists(settings_path):
            errors.append(f"settings file does not exist: {settings_path}")
        if timestamps not in (None, "", "auto"):
            timestamps_path = os.path.normpath(os.path.join(base_path, str(timestamps)))
            if not os.path.exists(timestamps_path):
                errors.append(f"timestamps file does not exist: {timestamps_path}")
        if groundtruth_file not in (None, "", "auto"):
            gt_path = os.path.normpath(os.path.join(base_path, str(groundtruth_file)))
            if not os.path.exists(gt_path):
                errors.append(f"groundtruth file does not exist: {gt_path}")

        if errors:
            context = [
                f"cwd: {os.getcwd()}",
                f"hostname: {os.uname().nodename}",
                f"config file: {config_file_path}",
                f"image glob tested: {full_glob}",
            ]
            raise ValueError(
                "FOLDER_DATASET preflight validation failed:\n- "
                + "\n- ".join(context + errors)
            )

        Printer.green(
            "FOLDER_DATASET preflight OK: "
            f"{len(matched_images)} images matched by {full_glob}"
        )

    def _parse_plot_formats(self, plot_formats):
        if isinstance(plot_formats, str):
            values = [x.strip().lower() for x in plot_formats.split(",") if x.strip()]
        elif isinstance(plot_formats, (list, tuple)):
            values = [str(x).strip().lower() for x in plot_formats if str(x).strip()]
        else:
            values = []
        if not values:
            values = ["png"]
        return list(dict.fromkeys(values))

    def get_main_slam_command(
        self,
        dataset,
        preset,
        iteration_idx,
        dataset_base_path,
        preset_common_parameters,
        output_path,
    ):
        print(
            "------------------------------------------------------------------------------------"
        )
        dataset_name = dataset["name"]
        preset_name = preset["name"]

        preset_output_path = os.path.abspath(os.path.join(output_path, preset_name))
        dataset_output_path = os.path.abspath(os.path.join(preset_output_path, dataset_name))
        iteration_output_path = os.path.abspath(
            os.path.join(dataset_output_path, "iteration_" + str(iteration_idx))
        )

        if not os.path.exists(preset_output_path):
            os.makedirs(preset_output_path)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        if not os.path.exists(iteration_output_path):
            os.makedirs(iteration_output_path)

        log_output_path = os.path.abspath(os.path.join(iteration_output_path, log_folder_name))
        if not os.path.exists(log_output_path):
            os.makedirs(log_output_path)
        main_log_output_file = os.path.join(log_output_path, "log.txt")

        # ------
        # Read the preset parameters and merge common with custom ones
        preset_custom_parameters = (
            preset["custom_parameters"] if "custom_parameters" in preset else None
        )

        if preset_custom_parameters:
            preset_parameters = merge_dicts(preset_common_parameters, preset_custom_parameters)
        else:
            preset_parameters = preset_common_parameters

        # ------
        # Prepare the settings file and update it with the preset parameters
        dataset_settings_path = dataset["settings_path"]
        if not os.path.exists(dataset_settings_path):
            raise Exception(f"Dataset settings file {dataset_settings_path} does not exist")
        dataset_settings = yaml.load(open(dataset_settings_path, "r"), Loader=yaml.FullLoader)
        preset_dataset_settings = (
            preset_parameters["settings"] if "settings" in preset_parameters else None
        )
        if preset_dataset_settings:
            dataset_settings = update_yaml_dict(dataset_settings, preset_dataset_settings)
        current_settings_file_path = os.path.join(iteration_output_path, "settings.yaml")
        with open(current_settings_file_path, "w") as f:
            yaml.dump(dataset_settings, f)

        # ------
        # Prepare the stereo settings file and update it with the preset parameters
        if "settings_stereo_path" in dataset:
            dataset_settings_stereo_path = dataset["settings_stereo_path"]
            if not os.path.exists(dataset_settings_stereo_path):
                raise Exception(
                    f"Dataset settings stereo file {dataset_settings_stereo_path} does not exist"
                )
            dataset_settings_stereo = yaml.load(
                open(dataset_settings_stereo_path, "r"), Loader=yaml.FullLoader
            )
            if preset_dataset_settings:
                dataset_settings_stereo = update_yaml_dict(
                    dataset_settings_stereo, preset_dataset_settings
                )
            current_settings_stereo_file_path = os.path.join(
                iteration_output_path, "settings_stereo.yaml"
            )
            with open(current_settings_stereo_file_path, "w") as f:
                yaml.dump(dataset_settings_stereo, f)

        # ------
        # Prepare the config file and update it with the preset parameters
        # Replace the template entries in the config file.
        config_vocabulary = {
            "dataset_type": self.dataset_type,
            "sensor_type": self.sensor_type,
            "output_path": iteration_output_path,
            "dataset_base_path": self._yaml_scalar(dataset_base_path),
            "dataset_name": self._yaml_scalar(dataset_name),
            "settings_path": self._yaml_scalar(current_settings_file_path),
            "saved_trajectory_format_type": self.saved_trajectory_format_type,
            "logs_path": self._yaml_scalar(log_output_path),
        }
        # Optional dataset-specific fields used by config.template.yaml placeholders
        if "images_glob_pattern" in dataset:
            config_vocabulary["dataset_images_glob_pattern"] = self._yaml_scalar(
                dataset["images_glob_pattern"]
            )
        if "fps" in dataset:
            config_vocabulary["dataset_fps"] = str(dataset["fps"])
        if "timestamps_path" in dataset:
            config_vocabulary["timestamps_path"] = self._yaml_scalar(dataset["timestamps_path"])
        if "groundtruth_file" in dataset:
            config_vocabulary["groundtruth_file_path"] = self._yaml_scalar(
                dataset["groundtruth_file"]
            )
        config_vocabulary["image_scale"] = str(dataset.get("image_scale", 1.0))
        config_vocabulary["max_frames"] = str(dataset.get("max_frames", 0))
        if "settings_stereo_path" in dataset:
            config_vocabulary["settings_stereo_path"] = self._yaml_scalar(
                current_settings_stereo_file_path
            )
        if "is_color" in dataset:
            config_vocabulary["is_color"] = str(dataset["is_color"])
        if "start_frame_id" in dataset:
            config_vocabulary["start_frame_id"] = dataset["start_frame_id"]
        else:
            config_vocabulary["start_frame_id"] = "0"

        current_config_file_str = replace_template_entries(self.template_config, config_vocabulary)
        # print(f"current_config_file_str: {current_config_file_str}")

        # Load as dictionary
        current_config_file = yaml.safe_load(current_config_file_str)
        # Clean up unused dataset sections
        current_config_file = remove_unused_datasets(current_config_file)
        # Apply config-level overrides from evaluation preset/common parameters
        # (e.g. GLOBAL_PARAMETERS, DATASET section fields, etc.)
        preset_config_settings = preset_parameters["config"] if "config" in preset_parameters else None
        if preset_config_settings:
            current_config_file = update_yaml_dict(current_config_file, preset_config_settings)
        # print(f'current_config_file: {current_config_file}')

        # Save the config file to be used
        current_config_file = yaml.safe_dump(
            current_config_file, sort_keys=False
        )  # Get it back as a string
        current_config_file_path = os.path.join(iteration_output_path, "config.yaml")
        with open(current_config_file_path, "w") as f:
            f.write(current_config_file)
        self._validate_generated_config(yaml.safe_load(current_config_file), current_config_file_path)

        Printer.bold_blue(
            f"\nRunning evaluation for preset: {preset_name}, dataset: {dataset_name}, iteration: {iteration_idx}"
        )
        Printer.bold(f"\nconfig file:")
        print(current_config_file)
        Printer.bold(f"preset parameters:")
        print(json.dumps(preset_parameters, indent=4))
        Printer.bold(f"output_path: {iteration_output_path}")
        # Printer.bold(f"\t current_config_file_path: {current_config_file_path}")
        if preset_custom_parameters:
            Printer.bold(f"custom_parameters:")
            print(f"{json.dumps(preset_custom_parameters, indent=4)}")
        # if preset_dataset_settings:
        #   Printer.bold(f"preset_dataset_settings: {json.dumps(preset_dataset_settings, indent=4)}")
        Printer.bold(f"log_output_path: {log_output_path}")
        Printer.yellow(f"To see the progress run: $ tail -f {main_log_output_file}")

        main_slam_path = os.path.join(kRootFolder, "main_slam.py")
        python_executable = shlex.quote(sys.executable)
        plot_formats_env = ",".join(self.plot_formats) if self.plot_formats else "png"
        run_seed = preset.get("seed", self.seed)
        if run_seed is None:
            run_seed = 12345
        if self.seed_per_run:
            run_seed = int(run_seed) + int(iteration_idx)
        deterministic_env = (
            f"PYSLAM_SEED={run_seed} "
            + "PYSLAM_FORCE_DETERMINISM=1 "
            + f"PYTHONHASHSEED={run_seed} "
            + "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            + "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
            + "CUBLAS_WORKSPACE_CONFIG=:4096:8 "
        )
        benchmark_env = (
            f'PYSLAM_EVAL_PLOT_FORMATS="{plot_formats_env}" '
            + f'PYSLAM_EXPORT_FRAME_LOGS={"1" if self.export_frame_logs else "0"} '
        )
        command = (
            deterministic_env
            + benchmark_env
            + python_executable
            + " "
            + main_slam_path
            + " --headless --no_output_date -c "
            + current_config_file_path
        )

        print("command: " + command)

        print("")
        return command, main_log_output_file

    def run_evaluation(self):
        commands = []
        for preset in self.json_data["presets"]:
            for dataset in self.json_data["datasets"]:
                for iteration_idx in range(self.number_of_runs_per_dataset):
                    item_run_command, item_log_output_path = self.get_main_slam_command(
                        dataset,
                        preset,
                        iteration_idx,
                        self.dataset_base_path,
                        self.preset_common_parameters,
                        self.output_path,
                    )
                    commands.append((item_run_command, item_log_output_path))
        print(
            "------------------------------------------------------------------------------------"
        )
        # Run commands in parallel
        num_threads = self.json_data["num_threads"]
        futures = []
        results = []
        future_to_command = {}
        future_to_msg = {}  # map futures to (command, label)
        future_to_log_file = {}

        num_tasks = len(commands)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks to the thread pool
            Printer.bold(
                f"Submitting {len(commands)} tasks to the thread pool with {num_threads} threads..."
            )
            num_started_tasks = 1
            for item_run_command, item_log_output_path in commands:
                msg = f"{num_started_tasks}/{num_tasks}"
                item_future = executor.submit(
                    run_command_async,
                    item_run_command,
                    item_log_output_path,
                    True,
                    msg,
                    self.max_run_seconds,
                )
                futures.append(item_future)
                future_to_command[item_future] = item_run_command
                future_to_msg[item_future] = msg
                future_to_log_file[item_future] = item_log_output_path
                num_started_tasks += 1
            # Printer.bold(f"\nSubmitted {num_tasks} tasks to the thread pool with {num_threads} threads.\n")
            # Wait for all tasks to complete
            num_completed_tasks = 0
            for future in concurrent.futures.as_completed(futures):
                num_completed_tasks += 1
                item_run_command = future_to_command[future]
                result = future.result()
                results.append(result)
                (returncode, stdout, stderr) = result
                msg = future_to_msg[future]
                if returncode == 0:
                    Printer.green(
                        f'Command {msg} "{item_run_command}" returned {returncode}, (completed {num_completed_tasks}/{num_tasks})'
                    )
                else:
                    self.had_run_failures = True
                    Printer.red(
                        f'Command {msg} "{item_run_command}" returned {returncode}, (completed {num_completed_tasks}/{num_tasks})'
                    )
                    self._print_log_tail(future_to_log_file[future])
        print(
            f"------------------------------------------------------------------------------------"
        )
        print(f"{num_completed_tasks}/{num_tasks} tasks completed")

    def collect_metrics(
        self, output_folder, preset_name, dataset_name, iteration_idx, metrics_vocabulary
    ):
        preset_results_path = os.path.join(output_folder, preset_name)
        dataset_results_path = os.path.join(preset_results_path, dataset_name)
        iteration_results_path = os.path.join(
            dataset_results_path, "iteration_" + str(iteration_idx)
        )

        metrics_folder_path = os.path.join(iteration_results_path, "plot")
        iteration_metrics_path = os.path.join(metrics_folder_path, "stats_final.json")

        other_metrics_info_path = os.path.join(iteration_results_path, "other_metrics_info.txt")
        if not os.path.exists(other_metrics_info_path):
            Printer.red(f"{dataset_name}: the file {other_metrics_info_path} is missing")
            return

        if not os.path.exists(iteration_metrics_path):
            Printer.red(f"{dataset_name}: the file {iteration_metrics_path} is missing")
            return

        Printer.bold(f"preset: {preset_name}, dataset: {dataset_name}, iteration: {iteration_idx}")

        # Open the text file in read mode and extract data.
        with open(other_metrics_info_path, "r") as f:
            text = f.read()
            lines = text.split("\n")  # Split the text into lines.
            for line in lines:  # Iterate over the lines and extract the values.
                if line == "":
                    continue
                # print(f"\t {line}")
                key, value = line.split(":", 1)
                print(f"\t {key}: {value}")
                if preset_name not in metrics_vocabulary:
                    metrics_vocabulary[preset_name] = {}
                if dataset_name not in metrics_vocabulary[preset_name]:
                    metrics_vocabulary[preset_name][dataset_name] = {}
                if iteration_idx not in metrics_vocabulary[preset_name][dataset_name]:
                    metrics_vocabulary[preset_name][dataset_name][iteration_idx] = {}
                value_clean = value.strip()
                try:
                    parsed_value = float(value_clean)
                except ValueError:
                    parsed_value = value_clean
                metrics_vocabulary[preset_name][dataset_name][iteration_idx][key] = parsed_value

        # Open the json file in read mode and extract data.
        with open(iteration_metrics_path, "r") as f:
            metrics_json_data = json.load(f)
            # Iterate over the dictionary and extract the values.
            for key, value in metrics_json_data.items():
                print(f"\t {key}: {value}")
                if preset_name not in metrics_vocabulary:
                    metrics_vocabulary[preset_name] = {}
                if dataset_name not in metrics_vocabulary[preset_name]:
                    metrics_vocabulary[preset_name][dataset_name] = {}
                if iteration_idx not in metrics_vocabulary[preset_name][dataset_name]:
                    metrics_vocabulary[preset_name][dataset_name][iteration_idx] = {}
                if isinstance(value, (int, float)):
                    metrics_vocabulary[preset_name][dataset_name][iteration_idx][key] = float(value)
                else:
                    metrics_vocabulary[preset_name][dataset_name][iteration_idx][key] = value

    def write_comparative_table(
        self,
        output_path,
        metric_name,
        presets,
        datasets,
        number_of_runs_per_dataset,
        metrics,
        precision=5,
    ):
        # Before generating the table, let's check if the provided metric is available
        is_metric_available = False
        for preset in presets:
            preset_name = preset["name"]
            if preset_name in metrics:
                for dataset in datasets:
                    dataset_name = dataset["name"]
                    if dataset_name in metrics[preset_name]:
                        for iteration_idx in range(number_of_runs_per_dataset):
                            it_metrics = metrics[preset_name][dataset_name].get(iteration_idx)
                            if it_metrics and metric_name in it_metrics:
                                is_metric_available = True
                                break
        if not is_metric_available:
            Printer.yellow(
                f'Metric name "{metric_name}" is not available. Skipping table generation for this metric name.'
            )
            return {}, None

        # Generate the table
        output_table_path = output_path + "/table_" + metric_name.replace(" ", "") + ".csv"
        csv_file = open(output_table_path, "w")
        writer = csv.writer(csv_file, delimiter=",")
        first_row = ["Dataset"]
        for preset in presets:
            first_row.append(preset["name"])
        writer.writerow(first_row)
        for dataset in datasets:
            dataset_name = dataset["name"]
            row = [dataset_name]
            for preset in presets:
                preset_name = preset["name"]
                if preset_name in metrics and dataset_name in metrics[preset_name]:
                    it_values = []
                    for iteration_idx in range(number_of_runs_per_dataset):
                        it_metrics = metrics[preset_name][dataset_name].get(iteration_idx)
                        if it_metrics and metric_name in it_metrics:
                            it_values.append(it_metrics[metric_name])
                    if len(it_values) > 0:
                        mean = np.mean(it_values)
                        row.append(round(mean, precision))
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            writer.writerow(row)
        writer.writerow([])  # add an empty row
        # add rows with the average and std
        row_average = ["Average"]
        row_std_dev = ["Std Dev"]
        map_preset_to_average = {}
        for preset in presets:
            preset_name = preset["name"]
            if preset_name in metrics:
                val_arrays = [
                    it_metrics[metric_name]
                    for dataset in datasets
                    if preset_name in metrics and dataset["name"] in metrics[preset_name]
                    for iteration_idx in range(number_of_runs_per_dataset)
                    if (it_metrics := metrics[preset_name][dataset["name"]].get(iteration_idx))
                    and metric_name in it_metrics
                    and not math.isnan(it_metrics[metric_name])
                ]
                mean = np.mean(val_arrays)
                row_average.append(round(mean, precision))
                std = np.std(val_arrays)
                row_std_dev.append(round(std, precision))
                map_preset_to_average[preset_name] = mean
            else:
                row_average.append("N/A")
                row_std_dev.append("N/A")
        writer.writerow(row_average)
        writer.writerow(row_std_dev)
        writer.writerow([])  # add an empty row
        # add final rows with best preset (with best average) and corresponding average
        row_best_preset = ["Best (Average) Preset"]
        row_best_average = ["Best (Average) Metric"]
        best_preset_name = ""
        best_average = float("inf")
        for preset_name, average in map_preset_to_average.items():
            if average < best_average:
                best_preset_name = preset_name
                best_average = average
        writer.writerow(row_best_preset + [best_preset_name])
        writer.writerow(row_best_average + [round(best_average, precision)])
        Printer.yellow(f'Generated table for metric "{metric_name}" in {output_table_path}')
        return map_preset_to_average, output_table_path

    def _safe_float(self, value):
        if value is None or value == "":
            return None
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

    def _safe_int(self, value):
        v = self._safe_float(value)
        if v is None:
            return None
        return int(v)

    def _value_or_empty(self, value):
        return "" if value is None else value

    def _quantile(self, values, q):
        if values is None or len(values) == 0:
            return None
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return None
        return float(np.quantile(arr, q))

    def _iqr(self, values):
        q25 = self._quantile(values, 0.25)
        q75 = self._quantile(values, 0.75)
        if q25 is None or q75 is None:
            return None
        return float(q75 - q25)

    def _wilson_ci_95(self, n, k):
        if n <= 0:
            return None, None
        z = 1.96
        p = float(k) / float(n)
        den = 1.0 + (z * z) / n
        center = (p + (z * z) / (2.0 * n)) / den
        halfwidth = (
            z * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n))) / den
        )
        return center - halfwidth, center + halfwidth

    def _read_key_value_file(self, file_path):
        data = {}
        if not os.path.exists(file_path):
            return data
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    value = value.strip()
                    parsed = self._safe_float(value)
                    data[key.strip()] = parsed if parsed is not None else value
        except Exception as e:
            Printer.red(f"Error while reading {file_path}: {e}")
        return data

    def _load_json_if_exists(self, file_path):
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            Printer.red(f"Error while reading {file_path}: {e}")
            return {}

    def _load_text_if_exists(self, file_path):
        if not os.path.exists(file_path):
            return ""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().lower()
        except Exception:
            return ""

    def _find_nested_key(self, data, key_candidates):
        if isinstance(data, dict):
            for key, value in data.items():
                key_l = str(key).lower()
                if key_l in key_candidates:
                    return value
                nested = self._find_nested_key(value, key_candidates)
                if nested is not None:
                    return nested
        elif isinstance(data, list):
            for value in data:
                nested = self._find_nested_key(value, key_candidates)
                if nested is not None:
                    return nested
        return None

    def _classify_failure_reason(
        self,
        log_text,
        coverage,
        trajectory_valid,
        crashed,
        n_frames_tracked,
        n_frames_total,
        relocalizing_frames=0,
        relocalization_tail_frames=0,
    ):
        if n_frames_tracked is not None and n_frames_total not in (None, 0):
            if n_frames_tracked <= max(1, int(0.05 * n_frames_total)):
                return "INIT_FAIL"
        if "eval_timeout_exceeded" in log_text:
            if "loopclosing" in log_text or "loop closing" in log_text:
                return "LOOP_CLOSURE_TIMEOUT"
            return "RUN_TIMEOUT"
        if "relocal" in log_text and ("fail" in log_text or "lost" in log_text):
            return "RELOC_FAIL"
        if relocalization_tail_frames and relocalization_tail_frames > 0:
            return "RELOCALIZATION_STUCK"
        if relocalizing_frames and relocalizing_frames > 0:
            return "RELOCALIZATION_RECOVERED"
        if (
            "loopclosing fault" in log_text
            or "loop closing fault" in log_text
            or "loop closure fault" in log_text
            or "loop closure failed" in log_text
        ):
            return "LOOP_CLOSURE_FAULT"
        if not trajectory_valid:
            return "DIVERGENCE_SILENT"
        if coverage is not None and coverage < 0.95:
            return "TRACK_DEGRADATION"
        if crashed:
            return "UNKNOWN"
        return "UNKNOWN"

    def _analyze_frame_log_states(self, frame_log_path):
        stats = {
            "relocalizing_frames": 0,
            "first_relocalizing_frame": None,
            "relocalization_tail_frames": 0,
        }
        if not frame_log_path or not os.path.exists(frame_log_path):
            return stats

        states = []
        try:
            with open(frame_log_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "tracking_state" not in reader.fieldnames:
                    return stats
                for row in reader:
                    frame_idx = self._safe_int(row.get("frame_idx"))
                    state = str(row.get("tracking_state", "")).strip().upper()
                    if state:
                        states.append((frame_idx, state))
                        if state == "RELOCALIZING":
                            stats["relocalizing_frames"] += 1
                            if stats["first_relocalizing_frame"] is None:
                                stats["first_relocalizing_frame"] = frame_idx
        except Exception as e:
            Printer.red(f"Error while parsing frame log {frame_log_path}: {e}")
            return stats

        tail_count = 0
        for _, state in reversed(states):
            if state == "RELOCALIZING":
                tail_count += 1
            else:
                break
        stats["relocalization_tail_frames"] = tail_count
        return stats

    def _build_run_metrics_rows(self):
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        rows = []
        for preset in self.presets:
            preset_name = preset["name"]
            seed_value = self._find_nested_key(
                preset, {"seed", "random_seed", "rng_seed", "seed_value"}
            )
            if seed_value is None:
                seed_value = self._find_nested_key(
                    self.json_data, {"seed", "random_seed", "rng_seed", "seed_value"}
                )
            for dataset in self.datasets:
                dataset_name = dataset["name"]
                fps = self._safe_float(dataset.get("fps"))
                for iteration_idx in range(self.number_of_runs_per_dataset):
                    iteration_results_path = os.path.join(
                        self.output_path,
                        preset_name,
                        dataset_name,
                        "iteration_" + str(iteration_idx),
                    )
                    plots_path = os.path.join(iteration_results_path, "plot")
                    stats_path = os.path.join(plots_path, "stats_final.json")
                    other_path = os.path.join(iteration_results_path, "other_metrics_info.txt")
                    log_path = os.path.join(iteration_results_path, log_folder_name, "log.txt")
                    frame_log_path = os.path.join(iteration_results_path, "frame_log.csv")

                    stats = self._load_json_if_exists(stats_path)
                    other_metrics = self._read_key_value_file(other_path)
                    log_text = self._load_text_if_exists(log_path)

                    n_frames_total = self._safe_int(other_metrics.get("num_total_frames"))
                    n_processed = self._safe_int(other_metrics.get("num_processed_frames"))
                    n_lost = self._safe_int(other_metrics.get("num_lost_frames"))
                    n_tracked = (
                        n_processed - n_lost
                        if n_processed is not None and n_lost is not None
                        else self._safe_int(other_metrics.get("num_tracked_frames"))
                    )
                    coverage = (
                        (float(n_tracked) / float(n_frames_total))
                        if n_tracked is not None and n_frames_total not in (None, 0)
                        else self._safe_float(other_metrics.get("coverage"))
                    )
                    frame_log_stats = self._analyze_frame_log_states(frame_log_path)
                    relocalizing_frames = int(frame_log_stats["relocalizing_frames"] or 0)
                    first_relocalizing_frame = frame_log_stats["first_relocalizing_frame"]
                    relocalization_tail_frames = int(
                        frame_log_stats["relocalization_tail_frames"] or 0
                    )
                    relocalizing_ratio = (
                        float(relocalizing_frames) / float(n_frames_total)
                        if n_frames_total not in (None, 0)
                        else None
                    )

                    ate_rmse = self._safe_float(stats.get("rmse"))
                    trajectory_valid = ate_rmse is not None
                    crashed = any(
                        token in log_text
                        for token in [
                            "segmentation fault",
                            "fatal error",
                            "core dumped",
                            "aborted",
                            "eval_timeout_exceeded",
                        ]
                    )
                    relocal_failed_in_log = "relocal" in log_text and (
                        "fail" in log_text or "lost" in log_text
                    )
                    success = (
                        coverage is not None
                        and coverage >= 0.95
                        and trajectory_valid
                        and not crashed
                        and not relocal_failed_in_log
                        and relocalization_tail_frames == 0
                    )
                    failure_reason = (
                        ""
                        if success
                        else self._classify_failure_reason(
                            log_text,
                            coverage,
                            trajectory_valid,
                            crashed,
                            n_tracked,
                            n_frames_total,
                            relocalizing_frames,
                            relocalization_tail_frames,
                        )
                    )

                    ttf_frame = "" if success else self._value_or_empty(n_tracked)
                    if not success and first_relocalizing_frame is not None:
                        ttf_frame = self._value_or_empty(first_relocalizing_frame)
                    ttf_seconds = ""
                    if not success and fps not in (None, 0) and isinstance(ttf_frame, int):
                        ttf_seconds = float(ttf_frame) / float(fps)

                    ate_error_series = stats.get("error_series", [])
                    if not isinstance(ate_error_series, list):
                        ate_error_series = []

                    rpe_trans_error_series = stats.get("rpe_trans_error_series", [])
                    if not isinstance(rpe_trans_error_series, list):
                        rpe_trans_error_series = []

                    rpe_rot_error_series = stats.get("rpe_rot_error_series", [])
                    if not isinstance(rpe_rot_error_series, list):
                        rpe_rot_error_series = []

                    ate_iqr_value = self._safe_float(stats.get("iqr"))
                    if ate_iqr_value is None:
                        ate_iqr_value = self._iqr(ate_error_series)
                    rpe_trans_iqr_value = self._safe_float(stats.get("rpe_trans_iqr"))
                    if rpe_trans_iqr_value is None:
                        rpe_trans_iqr_value = self._iqr(rpe_trans_error_series)
                    rpe_rot_iqr_value = self._safe_float(stats.get("rpe_rot_iqr"))
                    if rpe_rot_iqr_value is None:
                        rpe_rot_iqr_value = self._iqr(rpe_rot_error_series)
                    hmi_raw = other_metrics.get("hmi_ratio")
                    if hmi_raw is None or hmi_raw == "":
                        hmi_raw = stats.get("hmi_ratio")

                    row = {
                        "timestamp_utc": timestamp_utc,
                        "dataset_name": dataset_name,
                        "preset_name": preset_name,
                        "run_idx": iteration_idx,
                        "seed": self._value_or_empty(seed_value),
                        "num_threads": self.json_data.get("num_threads", ""),
                        "sensor_type": self.sensor_type,
                        "fps": self._value_or_empty(fps),
                        "n_frames_total (M)": self._value_or_empty(n_frames_total),
                        "n_frames_tracked (m)": self._value_or_empty(n_tracked),
                        "coverage (C=m/M)": self._value_or_empty(coverage),
                        "relocalizing_frames": self._value_or_empty(relocalizing_frames),
                        "relocalizing_ratio": self._value_or_empty(relocalizing_ratio),
                        "first_relocalizing_frame": self._value_or_empty(first_relocalizing_frame),
                        "relocalization_tail_frames": self._value_or_empty(relocalization_tail_frames),
                        "success (0/1)": 1 if success else 0,
                        "failure_reason": failure_reason,
                        "time_to_failure_frame": ttf_frame,
                        "time_to_failure_seconds": self._value_or_empty(ttf_seconds),
                        "align_type": stats.get("align_type", "SE3_UMeyama"),
                        "ate_rmse": self._value_or_empty(ate_rmse),
                        "ate_mean": self._value_or_empty(self._safe_float(stats.get("mean"))),
                        "ate_median": self._value_or_empty(self._safe_float(stats.get("median"))),
                        "ate_std": self._value_or_empty(self._safe_float(stats.get("std"))),
                        "ate_iqr": self._value_or_empty(ate_iqr_value),
                        "ate_p90": self._value_or_empty(self._safe_float(stats.get("p90"))),
                        "ate_p95": self._value_or_empty(self._safe_float(stats.get("p95"))),
                        "ate_max": self._value_or_empty(self._safe_float(stats.get("max"))),
                        "rpe_delta_frames": self._value_or_empty(
                            self._safe_int(stats.get("rpe_delta_frames"))
                        ),
                        "rpe_trans_rmse": self._value_or_empty(
                            self._safe_float(stats.get("rpe_trans_rmse"))
                        ),
                        "rpe_trans_mean": self._value_or_empty(
                            self._safe_float(stats.get("rpe_trans_mean"))
                        ),
                        "rpe_trans_median": self._value_or_empty(
                            self._safe_float(stats.get("rpe_trans_median"))
                        ),
                        "rpe_trans_std": self._value_or_empty(
                            self._safe_float(stats.get("rpe_trans_std"))
                        ),
                        "rpe_trans_iqr": self._value_or_empty(rpe_trans_iqr_value),
                        "rpe_trans_p90": self._value_or_empty(
                            self._safe_float(stats.get("rpe_trans_p90"))
                        ),
                        "rpe_trans_p95": self._value_or_empty(
                            self._safe_float(stats.get("rpe_trans_p95"))
                        ),
                        "rpe_trans_max": self._value_or_empty(
                            self._safe_float(stats.get("rpe_trans_max"))
                        ),
                        "rpe_rot_rmse": self._value_or_empty(
                            self._safe_float(stats.get("rpe_rot_rmse"))
                        ),
                        "rpe_rot_mean": self._value_or_empty(
                            self._safe_float(stats.get("rpe_rot_mean"))
                        ),
                        "rpe_rot_median": self._value_or_empty(
                            self._safe_float(stats.get("rpe_rot_median"))
                        ),
                        "rpe_rot_std": self._value_or_empty(
                            self._safe_float(stats.get("rpe_rot_std"))
                        ),
                        "rpe_rot_iqr": self._value_or_empty(rpe_rot_iqr_value),
                        "rpe_rot_p90": self._value_or_empty(self._safe_float(stats.get("rpe_rot_p90"))),
                        "rpe_rot_p95": self._value_or_empty(self._safe_float(stats.get("rpe_rot_p95"))),
                        "rpe_rot_max": self._value_or_empty(self._safe_float(stats.get("rpe_rot_max"))),
                        "runtime_seconds": self._value_or_empty(
                            self._safe_float(other_metrics.get("runtime_seconds"))
                        ),
                        "trajectories_folder": os.path.relpath(
                            iteration_results_path, self.output_path
                        ),
                        "plots_folder": os.path.relpath(plots_path, self.output_path),
                        "_hmi_ratio": self._safe_float(hmi_raw),
                    }
                    rows.append(row)
        return rows

    def _collect_numeric(self, rows, key):
        values = []
        for row in rows:
            val = self._safe_float(row.get(key))
            if val is not None:
                values.append(val)
        return values

    def _build_summary_metrics_rows(self, run_rows):
        summary_rows = []
        for preset in self.presets:
            preset_name = preset["name"]
            for dataset in self.datasets:
                dataset_name = dataset["name"]
                rows = [
                    r
                    for r in run_rows
                    if r["preset_name"] == preset_name and r["dataset_name"] == dataset_name
                ]
                n_total = len(rows)
                k_success = int(sum(int(r.get("success (0/1)", 0)) for r in rows))
                success_rate = float(k_success) / float(n_total) if n_total > 0 else None
                ci_low, ci_high = self._wilson_ci_95(n_total, k_success)

                success_rows = [r for r in rows if int(r.get("success (0/1)", 0)) == 1]
                failure_rows = [r for r in rows if int(r.get("success (0/1)", 0)) == 0]
                relocalized_rows = [
                    r for r in rows if self._safe_int(r.get("relocalizing_frames")) not in (None, 0)
                ]

                ate_success = self._collect_numeric(success_rows, "ate_rmse")
                rpe_trans_success = self._collect_numeric(success_rows, "rpe_trans_rmse")
                rpe_rot_success = self._collect_numeric(success_rows, "rpe_rot_rmse")
                coverage_all = self._collect_numeric(rows, "coverage (C=m/M)")
                ttf_fail = self._collect_numeric(failure_rows, "time_to_failure_seconds")
                hmi_values = [
                    float(r["_hmi_ratio"]) for r in rows if r.get("_hmi_ratio") is not None
                ]
                ate_rmse_cv_success = None
                if len(ate_success) > 1:
                    mean_ate = float(np.mean(ate_success))
                    if abs(mean_ate) > 1e-12:
                        ate_rmse_cv_success = float(np.std(ate_success)) / abs(mean_ate)

                summary_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "preset_name": preset_name,
                        "N (total runs)": n_total,
                        "k (success count)": k_success,
                        "success_rate (k/N)": self._value_or_empty(success_rate),
                        "ci95_wilson_low": self._value_or_empty(ci_low),
                        "ci95_wilson_high": self._value_or_empty(ci_high),
                        "runs_with_relocalization": len(relocalized_rows),
                        "relocalization_rate": self._value_or_empty(
                            (float(len(relocalized_rows)) / float(n_total)) if n_total > 0 else None
                        ),
                        "ate_rmse_median_success": self._value_or_empty(
                            self._quantile(ate_success, 0.5)
                        ),
                        "ate_rmse_std_success": self._value_or_empty(
                            float(np.std(ate_success)) if len(ate_success) > 0 else None
                        ),
                        "ate_rmse_iqr_success": self._value_or_empty(self._iqr(ate_success)),
                        "ate_rmse_cv_success": self._value_or_empty(ate_rmse_cv_success),
                        "rpe_trans_median_success": self._value_or_empty(
                            self._quantile(rpe_trans_success, 0.5)
                        ),
                        "rpe_trans_std_success": self._value_or_empty(
                            float(np.std(rpe_trans_success)) if len(rpe_trans_success) > 0 else None
                        ),
                        "rpe_trans_iqr_success": self._value_or_empty(self._iqr(rpe_trans_success)),
                        "rpe_rot_median_success": self._value_or_empty(
                            self._quantile(rpe_rot_success, 0.5)
                        ),
                        "rpe_rot_std_success": self._value_or_empty(
                            float(np.std(rpe_rot_success)) if len(rpe_rot_success) > 0 else None
                        ),
                        "rpe_rot_iqr_success": self._value_or_empty(self._iqr(rpe_rot_success)),
                        "coverage_median": self._value_or_empty(self._quantile(coverage_all, 0.5)),
                        "coverage_iqr": self._value_or_empty(self._iqr(coverage_all)),
                        "time_to_failure_median_failure": self._value_or_empty(
                            self._quantile(ttf_fail, 0.5)
                        ),
                        "time_to_failure_iqr_failure": self._value_or_empty(self._iqr(ttf_fail)),
                        "hmi_ratio_median": self._value_or_empty(self._quantile(hmi_values, 0.5)),
                        "hmi_ratio_iqr": self._value_or_empty(self._iqr(hmi_values)),
                    }
                )
        return summary_rows

    def export_benchmark_csv_artifacts(self):
        run_rows = self._build_run_metrics_rows()
        run_metrics_path = os.path.join(self.output_path, "run_metrics.csv")
        with open(run_metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.RUN_METRICS_HEADERS)
            writer.writeheader()
            for row in run_rows:
                writer.writerow({k: row.get(k, "") for k in self.RUN_METRICS_HEADERS})
        Printer.yellow(f"Generated run metrics CSV: {run_metrics_path}")

        summary_rows = self._build_summary_metrics_rows(run_rows)
        summary_metrics_path = os.path.join(self.output_path, "summary_metrics.csv")
        with open(summary_metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_METRICS_HEADERS)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k, "") for k in self.SUMMARY_METRICS_HEADERS})
        Printer.yellow(f"Generated summary metrics CSV: {summary_metrics_path}")

    def create_final_table(self, export_csv=False):
        for preset in self.presets:
            for dataset in self.datasets:
                for iteration_idx in range(self.number_of_runs_per_dataset):
                    self.collect_metrics(
                        self.output_path,
                        preset["name"],
                        dataset["name"],
                        iteration_idx,
                        self.metrics,
                    )

        map_preset_to_ATE, out_table_ATE_path = self.write_comparative_table(
            self.output_path,
            "rmse",
            self.presets,
            self.datasets,
            self.number_of_runs_per_dataset,
            self.metrics,
        )
        map_preset_to_max, out_table_max_path = self.write_comparative_table(
            self.output_path,
            "max",
            self.presets,
            self.datasets,
            self.number_of_runs_per_dataset,
            self.metrics,
        )
        map_preset_to_percent_lost, out_table_percent_lost_path = self.write_comparative_table(
            self.output_path,
            "percent_lost",
            self.presets,
            self.datasets,
            self.number_of_runs_per_dataset,
            self.metrics,
        )
        table_paths = [out_table_ATE_path, out_table_max_path, out_table_percent_lost_path]
        valid_table_paths = [p for p in table_paths if p]

        git_commit_hash = get_git_short_hash()
        print("git commit hash: ", git_commit_hash)

        if valid_table_paths:
            try:
                from pyslam.utilities.evaluation_latex import csv_list_to_pdf

                csv_list_to_pdf(
                    valid_table_paths,
                    self.output_path + "/report.pdf",
                    title="pySLAM Evaluation Report",
                    git_commit_hash=git_commit_hash,
                )
            except Exception as e:
                print("Error converting tables to pdf report: ", e)
                print(traceback.format_exc())

            try:
                from pyslam.utilities.evaluation_html import csv_list_to_html

                csv_list_to_html(
                    valid_table_paths,
                    self.output_path + "/report.html",
                    title="pySLAM Evaluation Report",
                    git_commit_hash=git_commit_hash,
                )
            except Exception as e:
                print("Error converting tables to html report: ", e)
                print(traceback.format_exc())
        else:
            Printer.yellow(
                "No comparative metric tables generated (rmse/max/percent_lost unavailable). "
                "Skipping PDF/HTML report generation."
            )

        if export_csv:
            try:
                self.export_benchmark_csv_artifacts()
            except Exception as e:
                print("Error exporting benchmark CSV artifacts: ", e)
                print(traceback.format_exc())
