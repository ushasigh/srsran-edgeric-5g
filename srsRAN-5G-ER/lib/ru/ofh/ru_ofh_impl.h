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

#include "ru_ofh_controller_impl.h"
#include "ru_ofh_downlink_plane_handler_proxy.h"
#include "ru_ofh_error_handler_impl.h"
#include "ru_ofh_timing_notifier_impl.h"
#include "ru_ofh_uplink_plane_handler_proxy.h"
#include "srsran/ofh/ethernet/ethernet_frame_pool.h"
#include "srsran/ofh/ethernet/ethernet_gateway.h"
#include "srsran/ofh/ofh_sector.h"
#include "srsran/ofh/ofh_uplane_rx_symbol_notifier.h"
#include "srsran/ofh/timing/ofh_timing_manager.h"
#include "srsran/ru/ru.h"
#include "srsran/ru/ru_ofh_configuration.h"
#include "srsran/ru/ru_timing_notifier.h"
#include "srsran/ru/ru_uplink_plane.h"
#include "srsran/srslog/logger.h"

namespace srsran {

/// Open Fronthaul implementation configuration.
struct ru_ofh_impl_config {
  unsigned nof_slot_offset_du_ru;
  unsigned nof_symbols_per_slot;
};

/// Open Fronthaul implementation dependencies.
struct ru_ofh_impl_dependencies {
  srslog::basic_logger*                     logger;
  std::unique_ptr<ofh::timing_manager>      timing_mngr;
  ru_timing_notifier*                       timing_notifier = nullptr;
  ru_error_notifier*                        error_notifier  = nullptr;
  std::vector<std::unique_ptr<ofh::sector>> sectors;
};

/// Open Fronthaul Radio Unit implementation.
class ru_ofh_impl : public radio_unit
{
public:
  ru_ofh_impl(const ru_ofh_impl_config& config, ru_ofh_impl_dependencies&& dependencies);

  // See interface for documentation.
  ru_controller& get_controller() override;

  // See interface for documentation.
  ru_downlink_plane_handler& get_downlink_plane_handler() override;

  // See interface for documentation.
  ru_uplink_plane_handler& get_uplink_plane_handler() override;

private:
  ru_ofh_timing_notifier_impl               timing_notifier;
  ru_ofh_error_handler_impl                 error_handler;
  std::vector<std::unique_ptr<ofh::sector>> sectors;
  std::unique_ptr<ofh::timing_manager>      ofh_timing_mngr;
  ru_ofh_controller_impl                    controller;
  ru_downlink_plane_handler_proxy           downlink_handler;
  ru_uplink_plane_handler_proxy             uplink_handler;
};

} // namespace srsran
