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

#include "du_ue_manager_repository.h"
#include "srsran/srslog/srslog.h"

namespace srsran {
namespace srs_du {

class du_ue_controller_impl final : public du_ue
{
public:
  du_ue_controller_impl(const du_ue_context&         context_,
                        du_ue_manager_repository&    ue_db_,
                        const du_manager_params&     cfg_,
                        ue_ran_resource_configurator ue_ran_res_);
  ~du_ue_controller_impl() override;

  /// Launches a task to stop all the UE SRB and DRB traffic.
  async_task<void> handle_traffic_stop_request() override;

  /// Launches a task to stop the UE DRB traffic due to inactivity.
  async_task<void> handle_activity_stop_request() override;

  /// Launches a task to stop the traffic for some UE DRBs.
  async_task<void> handle_drb_traffic_stop_request(span<const drb_id_t> drbs_to_stop) override;

  void schedule_async_task(async_task<void> task) override { ue_db.schedule_async_task(ue_index, std::move(task)); }

  void handle_rlf_detection(rlf_cause cause) override;

  void handle_crnti_ce_detection() override;

  /// Schedule a task to stop DRB traffic.
  void stop_drb_traffic() override;

  mac_ue_radio_link_notifier&          get_mac_rlf_notifier() override { return *mac_rlf_notifier; }
  rlc_tx_upper_layer_control_notifier& get_rlc_rlf_notifier() override { return *rlc_rlf_notifier; }

private:
  class rlf_state_machine;

  async_task<void> run_in_ue_executor(unique_task task);

  async_task<void> create_stop_drb_traffic_task();
  async_task<void> create_stop_traffic_task();

  du_ue_manager_repository& ue_db;
  const du_manager_params&  cfg;
  srslog::basic_logger&     logger = srslog::fetch_basic_logger("DU-MNG");

  std::unique_ptr<rlf_state_machine>                   rlf_handler;
  std::unique_ptr<mac_ue_radio_link_notifier>          mac_rlf_notifier;
  std::unique_ptr<rlc_tx_upper_layer_control_notifier> rlc_rlf_notifier;
};

} // namespace srs_du
} // namespace srsran
