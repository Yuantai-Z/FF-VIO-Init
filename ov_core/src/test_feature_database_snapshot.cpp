/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"

#include <atomic>
#include <iostream>
#include <thread>

int main() {
  ov_core::FeatureDatabase database;
  database.update_feature(7, 1.0, 0, 10.0f, 20.0f, 0.1f, 0.2f);
  const auto snapshot = database.clone_measurements();

  database.update_feature(7, 2.0, 0, 11.0f, 21.0f, 0.2f, 0.3f);
  const auto snapshot_data = snapshot->get_internal_data();
  if (snapshot_data.at(7)->timestamps.at(0).size() != 1 || snapshot_data.at(7)->timestamps.at(0).front() != 1.0) {
    std::cerr << "feature snapshot changed after the live database was updated\n";
    return 1;
  }

  std::atomic<bool> writer_done{false};
  std::thread writer([&database, &writer_done] {
    for (int index = 0; index < 1000; ++index) {
      const double timestamp = 3.0 + 0.01 * index;
      database.update_feature(7, timestamp, 0, static_cast<float>(index), static_cast<float>(index + 1),
                              static_cast<float>(index) * 0.01f, static_cast<float>(index + 1) * 0.01f);
    }
    writer_done.store(true);
  });

  while (!writer_done.load()) {
    const auto concurrent_snapshot = database.clone_measurements()->get_internal_data();
    const auto &feature = concurrent_snapshot.at(7);
    const size_t timestamp_count = feature->timestamps.at(0).size();
    if (feature->uvs.at(0).size() != timestamp_count || feature->uvs_norm.at(0).size() != timestamp_count) {
      std::cerr << "feature snapshot contained a partially updated measurement\n";
      writer.join();
      return 1;
    }
  }
  writer.join();
  return 0;
}
