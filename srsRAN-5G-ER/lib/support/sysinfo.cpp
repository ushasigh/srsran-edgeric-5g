/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include "srsran/support/sysinfo.h"
#include "srsran/support/executors/unique_thread.h"
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

static const std::string isolated_cgroup_path     = "/sys/fs/cgroup/srs_isolated";
static const std::string housekeeping_cgroup_path = "/sys/fs/cgroup/srs_housekeeping";

/// Executes system command, deletes the given path if the command fails.
static bool exec_system_command(const std::string& command, const std::string& cleanup_path = "")
{
  if (::system(command.c_str()) < 0) {
    fmt::print("{} command failed. error=\"{}\"\n", command, strerror(errno));
    if (!cleanup_path.empty()) {
      ::rmdir(cleanup_path.c_str());
    }

    return false;
  }
  return true;
}

/// Writing the value 0 to a cgroup.procs file causes the writing process to be moved to the corresponding cgroup.
static void move_to_cgroup(const std::string& cgroup_path)
{
  std::ofstream output(cgroup_path + "/cgroup.procs");
  if (output.fail()) {
    fmt::print("Could not open {} for writing. error=\"{}\"\n", cgroup_path + "/cgroup.procs", strerror(errno));
  }
  output.write("0\n", 2);
}

/// Moves processes from source cgroup to destination cgroup.
static bool move_procs_between_cgroups(const std::string& dst_path, const std::string& src_path)
{
  using namespace std::chrono_literals;

  std::ifstream source_file(src_path);
  if (source_file.fail()) {
    fmt::print("Could not open {} directory. error=\"{}\"\n", src_path, strerror(errno));
    return false;
  }
  std::string pid;
  while (std::getline(source_file, pid)) {
    std::ofstream destination_file(dst_path);
    if (destination_file.fail()) {
      fmt::print("Could not open {} directory. error=\"{}\"\n", dst_path, strerror(errno));
      return false;
    }
    destination_file << std::stoi(pid) << "\n";
  }
  std::this_thread::sleep_for(50ms);

  return true;
}

bool srsran::configure_cgroups(const srsran::os_sched_affinity_bitmask& isol_cpus)
{
  std::string isolated_cpus;
  std::string os_cpus;

  // Create the string with the CPU indexes.
  for (unsigned pos = 0; pos != isol_cpus.size(); ++pos) {
    if (isol_cpus.test(pos)) {
      isolated_cpus += (isolated_cpus.empty()) ? std::to_string(pos) : "," + std::to_string(pos);
    } else {
      os_cpus += (os_cpus.empty()) ? std::to_string(pos) : "," + std::to_string(pos);
    }
  }

  std::string cgroup_path = "/sys/fs/cgroup";
  struct stat info;
  if (::stat(cgroup_path.c_str(), &info) < 0) {
    fmt::print("Could not find {}, make sure cgroups-v2 is mounted. error=\"{}\"\n", cgroup_path, strerror(errno));
    return false;
  }

  /// First move itself to root cgroup.
  move_to_cgroup(cgroup_path);

  /// Create cgroup for OS tasks, call it 'housekeeping' cgroup.
  if (!os_cpus.empty()) {
    std::string housekeeping = cgroup_path + "/srs_housekeeping";
    if ((::mkdir(housekeeping.c_str(), 0755)) < 0 && errno != EEXIST) {
      fmt::print("Could not create {} directory. error=\"{}\"\n", housekeeping, strerror(errno));
      return false;
    }
    std::string set_cpus_cmd = "echo " + os_cpus + " > " + housekeeping + "/cpuset.cpus";
    if (!exec_system_command(set_cpus_cmd, housekeeping)) {
      return false;
    }
    /// Migrate all processes to the default cgroup, that will be using housekeeping cpuset.
    std::string procs_filename = housekeeping + "/cgroup.procs";
    FILE*       file           = popen("ps -eLo lwp=", "r");
    if (!file) {
      fmt::print("Couldn't move system processes to a dedicated cgroup\n");
      return false;
    }
    const size_t len = 32;
    char         pid_buffer[len];
    while (::fgets(pid_buffer, len, file)) {
      unsigned          pid;
      std::stringstream ss(pid_buffer);
      std::ofstream     output(procs_filename);
      if (output.fail()) {
        fmt::print("Could not open {} directory. error=\"{}\"\n", procs_filename, strerror(errno));
        return false;
      }
      ss >> pid;
      output << pid << "\n";
    }
    ::pclose(file);
  }

  /// Create cgroup with isolated CPUs dedicated for the gNB app.
  std::string isol_cgroup_path = cgroup_path + "/srs_isolated";
  if ((::mkdir(isol_cgroup_path.c_str(), 0755)) < 0 && errno != EEXIST) {
    fmt::print("Could not create {} directory. error=\"{}\"\n", isol_cgroup_path, strerror(errno));
    return false;
  }

  std::string set_cpus_cmd = "echo " + isolated_cpus + " > " + isol_cgroup_path + "/cpuset.cpus";
  if (!exec_system_command(set_cpus_cmd, isol_cgroup_path)) {
    return false;
  }

  /// Finally move itself to isolcated cgroup.
  move_to_cgroup(isol_cgroup_path);

  return true;
}

void srsran::cleanup_cgroups()
{
  using namespace std::chrono_literals;

  bool        cgroup_changed   = false;
  std::string root_cgroup_path = "/sys/fs/cgroup/cgroup.procs";

  struct stat sysfs_info;
  if (::stat("/sys/fs/cgroup", &sysfs_info) < 0) {
    return;
  }

  struct stat info;
  if (::stat(housekeeping_cgroup_path.c_str(), &info) == 0) {
    move_procs_between_cgroups(root_cgroup_path, housekeeping_cgroup_path + "/cgroup.procs");
    // Remove previously created cgroups.
    if (::rmdir(housekeeping_cgroup_path.c_str()) < 0) {
      fmt::print("Could not delete {}. error=\"{}\"\n", housekeeping_cgroup_path, strerror(errno));
    }
    cgroup_changed = true;
  }
  if (::stat(isolated_cgroup_path.c_str(), &info) == 0) {
    // Move all processes to the parent cgroup.
    move_procs_between_cgroups(root_cgroup_path, isolated_cgroup_path + "/cgroup.procs");
    if (::rmdir(isolated_cgroup_path.c_str())) {
      fmt::print("Could not delete {}. error=\"{}\"\n", isolated_cgroup_path, strerror(errno));
    }
    cgroup_changed = true;
  }
  // Wait for changes in cpuset to take place.
  if (cgroup_changed) {
    std::this_thread::sleep_for(100ms);
  }
}

std::optional<std::string> srsran::check_cgroups()
{
  // Return false if the system doesn't have cgroups-v2 mounted.
  struct stat sysfs_info;
  if (::stat("/sys/fs/cgroup", &sysfs_info) < 0) {
    return {};
  }

  fmt::memory_buffer buffer;
  struct stat        info;
  if (::stat(housekeeping_cgroup_path.c_str(), &info) == 0) {
    fmt::format_to(buffer, "{}", housekeeping_cgroup_path);
  }
  if (::stat(isolated_cgroup_path.c_str(), &info) == 0) {
    if (buffer.size() != 0) {
      fmt::format_to(buffer, ", ");
    }
    fmt::format_to(buffer, "{}", isolated_cgroup_path);
  }

  return (buffer.size() != 0) ? to_string(buffer) : std::optional<std::string>{};
}

bool srsran::check_cpu_governor(srslog::basic_logger& logger)
{
  unsigned int n_cpus        = std::thread::hardware_concurrency();
  std::string  filename_base = "/sys/devices/system/cpu/cpu";
  for (unsigned int i = 0; i < n_cpus; ++i) {
    std::string   filename = filename_base + std::to_string(i) + "/cpufreq/scaling_governor";
    std::ifstream input(filename);
    if (input.fail()) {
      logger.warning("Could not check scaling governor. filename={} error=\"{}\"", filename, strerror(errno));
      return false;
    }
    std::string gov;
    std::getline(input, gov);
    if (input.fail()) {
      logger.warning("Could not check scaling governor. filename={} error=\"{}\"", filename, strerror(errno));
      return false;
    }
    if (gov == "performance") {
      logger.debug("CPU{} scaling governor is set to performance", i);
    } else {
      logger.warning(
          "CPU{} scaling governor is not set to performance, which may hinder performance. You can set it to "
          "performance using the "
          "\"srsran_performance\" script",
          i);
    }
  }
  return true;
}

bool srsran::check_drm_kms_polling(srslog::basic_logger& logger)
{
  std::string   filename = "/sys/module/drm_kms_helper/parameters/poll";
  std::ifstream input(filename);
  if (input.fail()) {
    logger.warning("Could not check DRM KMS polling. filename={} error=\"{}\"", filename, strerror(errno));
    return false;
  }
  std::string polling;
  std::getline(input, polling);
  if (input.fail()) {
    logger.warning("Could not check DRM KMS polling. filename={} error=\"{}\"", filename, strerror(errno));
    return false;
  }
  if (polling == "N") {
    logger.debug("DRM KMS polling is disabled");
  } else {
    logger.warning("DRM KMS polling is enabled, which may hinder performance. You can disable it using the "
                   "\"srsran_performance\" script");
  }
  return true;
}
