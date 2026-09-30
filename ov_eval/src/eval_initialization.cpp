/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "InitializationEvaluator.h"

#include <boost/filesystem.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  std::string result_path;
  std::string ground_truth_path;
  std::string output_path;
  ov_eval::InitializationAlignment alignment = ov_eval::InitializationAlignment::POSYAW;
};

std::string default_output_path(const std::string &result_path) {
  const boost::filesystem::path input(result_path);
  return (input.parent_path() / (input.stem().string() + "_init_eval.md")).string();
}

bool paths_match(const std::string &first, const std::string &second) {
  const boost::filesystem::path first_absolute = boost::filesystem::absolute(first).lexically_normal();
  const boost::filesystem::path second_absolute = boost::filesystem::absolute(second).lexically_normal();
  if (first_absolute == second_absolute) {
    return true;
  }
  boost::system::error_code error;
  const bool equivalent = boost::filesystem::exists(first_absolute, error) && !error &&
                          boost::filesystem::exists(second_absolute, error) && !error &&
                          boost::filesystem::equivalent(first_absolute, second_absolute, error);
  return !error && equivalent;
}

void print_usage(const char *executable) {
  std::cerr << "Usage: " << executable
            << " <result.tum> <groundtruth.tum|data.csv> [posyaw|se3|sim3] [--output <report.md>]\n"
            << "       alignment defaults to posyaw\n"
            << "       output defaults to <result_stem>_init_eval.md beside the trajectory\n";
}

bool parse_cli(int argc, char **argv, CliOptions &options, std::string &error) {
  if (argc < 3) {
    error = "missing result or ground-truth path";
    return false;
  }
  options.result_path = argv[1];
  options.ground_truth_path = argv[2];
  bool alignment_seen = false;
  bool output_seen = false;
  for (int index = 3; index < argc; index++) {
    const std::string argument = argv[index];
    if (argument == "--output" || argument == "-o") {
      if (output_seen || index + 1 >= argc) {
        error = output_seen ? "output path was specified more than once" : "--output requires a path";
        return false;
      }
      options.output_path = argv[++index];
      output_seen = true;
      continue;
    }
    const std::string output_prefix = "--output=";
    if (argument.compare(0, output_prefix.size(), output_prefix) == 0) {
      if (output_seen || argument.size() == output_prefix.size()) {
        error = output_seen ? "output path was specified more than once" : "--output requires a path";
        return false;
      }
      options.output_path = argument.substr(output_prefix.size());
      output_seen = true;
      continue;
    }
    if (!alignment_seen && ov_eval::parse_initialization_alignment(argument, options.alignment)) {
      alignment_seen = true;
      continue;
    }
    error = "unrecognized argument '" + argument + "'";
    return false;
  }
  if (!output_seen) {
    options.output_path = default_output_path(options.result_path);
  }
  if (paths_match(options.result_path, options.output_path)) {
    error = "report path must differ from the input trajectory";
    return false;
  }
  if (paths_match(options.ground_truth_path, options.output_path)) {
    error = "report path must differ from the ground-truth input";
    return false;
  }
  return true;
}

bool complete(const ov_eval::InitializationSegmentEvaluation &evaluation) {
  return evaluation.linear.rotation_degrees.valid && evaluation.linear.gravity_degrees.valid &&
         evaluation.linear.velocity_meters_per_second.valid && evaluation.nonlinear.rotation_degrees.valid &&
         evaluation.nonlinear.gravity_degrees.valid && evaluation.nonlinear.velocity_meters_per_second.valid &&
         evaluation.time_seconds.valid;
}

} // namespace

int main(int argc, char **argv) {
  CliOptions options;
  std::string error;
  if (!parse_cli(argc, argv, options, error)) {
    std::cerr << "Invalid arguments: " << error << "\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  ov_eval::InitializationResultFile result;
  if (!ov_eval::parse_initialization_result(options.result_path, result, error)) {
    std::cerr << "Failed to parse initialization result: " << error << "\n";
    return EXIT_FAILURE;
  }
  ov_eval::GroundTruthTrajectory ground_truth;
  if (!ov_eval::load_initialization_ground_truth(options.ground_truth_path, ground_truth, error)) {
    std::cerr << "Failed to load ground truth: " << error << "\n";
    return EXIT_FAILURE;
  }

  std::vector<ov_eval::InitializationSegmentEvaluation> evaluations;
  evaluations.reserve(result.segments.size());
  bool have_complete_sequence = false;
  for (const ov_eval::InitializationSegment &segment : result.segments) {
    evaluations.push_back(ov_eval::evaluate_initialization_segment(segment, ground_truth, options.alignment));
    have_complete_sequence = have_complete_sequence || complete(evaluations.back());
  }

  const std::string report = ov_eval::format_initialization_report(evaluations);
  std::ofstream output(options.output_path, std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    std::cerr << "Failed to open initialization report: " << options.output_path << "\n";
    return EXIT_FAILURE;
  }
  output << report;
  output.flush();
  if (!output.good()) {
    std::cerr << "Failed to write initialization report: " << options.output_path << "\n";
    return EXIT_FAILURE;
  }
  output.close();
  if (output.fail()) {
    std::cerr << "Failed to close initialization report: " << options.output_path << "\n";
    return EXIT_FAILURE;
  }

  std::cout << report;
  std::cerr << "Wrote initialization report: " << boost::filesystem::absolute(options.output_path).lexically_normal().string() << "\n";
  for (const ov_eval::InitializationSegmentEvaluation &evaluation : evaluations) {
    for (const std::string &diagnostic : evaluation.diagnostics) {
      std::cerr << "segment " << evaluation.id << ": " << diagnostic << "\n";
    }
  }
  if (!have_complete_sequence) {
    std::cerr << "No sequence contained all initialization states, timing, poses, and ground-truth associations.\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
