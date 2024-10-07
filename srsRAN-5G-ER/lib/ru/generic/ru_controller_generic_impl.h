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

#include "srsran/phy/adapters/phy_metrics_adapter.h"
#include "srsran/phy/lower/lower_phy_controller.h"
#include "srsran/phy/lower/lower_phy_metrics_notifier.h"
#include "srsran/ru/ru_controller.h"
#include <vector>

namespace srsran {

class lower_phy_controller;
class lower_phy_metrics_notifier;
class radio_session;

/// Radio Unit controller generic implementation.
class ru_controller_generic_impl : public ru_controller
{
public:
  ru_controller_generic_impl(std::vector<lower_phy_controller*> low_phy_crtl_,
                             std::vector<phy_metrics_adapter*>  low_phy_metrics_,
                             radio_session&                     radio_,
                             double                             srate_MHz_);

  // See interface for documentation.
  void start() override;

  // See interface for documentation.
  void stop() override;

  // See interface for documentation.
  bool set_tx_gain(unsigned port_id, double gain_dB) override;

  // See interface for documentation.
  bool set_rx_gain(unsigned port_id, double gain_dB) override;

  // See interface for documentation.
  void print_metrics() override;

private:
  std::vector<lower_phy_controller*> low_phy_crtl;
  std::vector<phy_metrics_adapter*>  low_phy_metrics;
  radio_session&                     radio;
  const double                       srate_MHz;
};

} // namespace srsran
