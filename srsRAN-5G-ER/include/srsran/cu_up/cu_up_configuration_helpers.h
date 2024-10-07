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

#pragma once

#include "srsran/cu_up/cu_up_types.h"
#include "srsran/ran/five_qi.h"
#include <map>

namespace srsran {
namespace config_helpers {

/// Generates default QoS configuration used by gNB CU-UP.
inline std::map<five_qi_t, srs_cu_up::cu_up_qos_config>
make_default_cu_up_qos_config_list(bool warn_on_drop, timer_duration metrics_period, bool test_mode)
{
  std::map<five_qi_t, srs_cu_up::cu_up_qos_config> qos_list = {};
  {
    // 5QI=1
    srs_cu_up::cu_up_qos_config cfg{};
    cfg.pdcp_custom_cfg = pdcp_custom_config{}; // defaults are configured as member-initialization within the struct
    cfg.pdcp_custom_cfg.metrics_period  = metrics_period;
    cfg.pdcp_custom_cfg.tx.warn_on_drop = warn_on_drop;
    cfg.pdcp_custom_cfg.tx.test_mode    = test_mode;

    qos_list[uint_to_five_qi(1)] = cfg;
  }
  {
    // 5QI=2
    srs_cu_up::cu_up_qos_config cfg{};
    cfg.pdcp_custom_cfg = pdcp_custom_config{}; // defaults are configured as member-initialization within the struct
    cfg.pdcp_custom_cfg.metrics_period  = metrics_period;
    cfg.pdcp_custom_cfg.tx.warn_on_drop = warn_on_drop;
    cfg.pdcp_custom_cfg.tx.test_mode    = test_mode;

    qos_list[uint_to_five_qi(2)] = cfg;
  }
  {
    // 5QI=5
    srs_cu_up::cu_up_qos_config cfg{};
    cfg.pdcp_custom_cfg = pdcp_custom_config{}; // defaults are configured as member-initialization within the struct
    cfg.pdcp_custom_cfg.metrics_period  = metrics_period;
    cfg.pdcp_custom_cfg.tx.warn_on_drop = warn_on_drop;
    cfg.pdcp_custom_cfg.tx.test_mode    = test_mode;

    qos_list[uint_to_five_qi(5)] = cfg;
  }
  {
    // 5QI=7
    srs_cu_up::cu_up_qos_config cfg{};
    cfg.pdcp_custom_cfg = pdcp_custom_config{}; // defaults are configured as member-initialization within the struct
    cfg.pdcp_custom_cfg.metrics_period  = metrics_period;
    cfg.pdcp_custom_cfg.tx.warn_on_drop = warn_on_drop;
    cfg.pdcp_custom_cfg.tx.test_mode    = test_mode;

    cfg.f1u_cfg.warn_on_drop = warn_on_drop;

    qos_list[uint_to_five_qi(7)] = cfg;
  }
  {
    // 5QI=9
    srs_cu_up::cu_up_qos_config cfg{};
    cfg.pdcp_custom_cfg = pdcp_custom_config{}; // defaults are configured as member-initialization within the struct
    cfg.pdcp_custom_cfg.metrics_period  = metrics_period;
    cfg.pdcp_custom_cfg.tx.warn_on_drop = warn_on_drop;
    cfg.pdcp_custom_cfg.tx.test_mode    = test_mode;

    cfg.f1u_cfg.warn_on_drop = warn_on_drop;

    qos_list[uint_to_five_qi(9)] = cfg;
  }
  return qos_list;
}

} // namespace config_helpers
} // namespace srsran
