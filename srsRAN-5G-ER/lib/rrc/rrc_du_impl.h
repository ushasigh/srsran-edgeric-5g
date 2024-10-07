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

#include "ue/rrc_ue_impl.h"
#include "srsran/ran/nr_cgi.h"
#include "srsran/rrc/rrc_config.h"
#include "srsran/rrc/rrc_du.h"
#include <unordered_map>

namespace srsran {

namespace srs_cu_cp {

/// Main RRC representation with the DU
class rrc_du_impl : public rrc_du_interface
{
public:
  rrc_du_impl(const rrc_cfg_t&                    cfg_,
              rrc_ue_nas_notifier&                nas_notif_,
              rrc_ue_control_notifier&            ngap_ctrl_notif_,
              rrc_du_measurement_config_notifier& meas_config_notifier_);
  ~rrc_du_impl() = default;

  // rrc_du_cell_manager
  bool handle_served_cell_list(const std::vector<cu_cp_du_served_cells_item>& served_cell_list) override;

  // rrc_du_ue_repository
  byte_buffer       get_rrc_reject() override;
  rrc_ue_interface* add_ue(const rrc_ue_creation_message& msg) override;
  void              release_ues() override;

  // rrc_ue_handler
  rrc_ue_interface* find_ue(ue_index_t ue_index) override
  {
    srsran_assert(ue_db.find(ue_index) != ue_db.end(), "UE not found");
    return ue_db.at(ue_index).get();
  }
  void remove_ue(ue_index_t ue_index) override;

  // rrc_du_statistics_handler
  size_t get_nof_ues() const override { return ue_db.size(); }

  rrc_du_cell_manager&       get_rrc_du_cell_manager() override { return *this; }
  rrc_du_ue_repository&      get_rrc_du_ue_repository() override { return *this; }
  rrc_ue_handler&            get_rrc_ue_handler() override { return *this; }
  rrc_du_statistics_handler& get_rrc_du_statistics_handler() override { return *this; }

private:
  // helpers
  const rrc_cfg_t cfg;

  rrc_ue_nas_notifier&                nas_notifier;         // PDU notifier to the NGAP
  rrc_ue_control_notifier&            ngap_ctrl_notifier;   // Control notifier to the NGAP
  rrc_du_measurement_config_notifier& meas_config_notifier; // notifier to the CU-CP
  srslog::basic_logger&               logger;

  // RRC-internal user database indexed by ue_index
  std::unordered_map<ue_index_t, std::unique_ptr<rrc_ue_impl>> ue_db;
  // Cell database to store cell information from the DU
  std::map<nr_cell_identity, rrc_cell_info> cell_info_db;
};

} // namespace srs_cu_cp

} // namespace srsran
