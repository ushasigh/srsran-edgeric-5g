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

#include "lib/scheduler/pucch_scheduling/pucch_allocator_impl.h"
#include "lib/scheduler/uci_scheduling/uci_allocator_impl.h"
#include "lib/scheduler/uci_scheduling/uci_scheduler_impl.h"
#include "tests/test_doubles/scheduler/pucch_res_test_builder_helper.h"
#include "tests/unittests/scheduler/test_utils/config_generators.h"
#include "tests/unittests/scheduler/test_utils/dummy_test_components.h"
#include "tests/unittests/scheduler/test_utils/scheduler_test_suite.h"
#include "srsran/du/du_cell_config_helpers.h"
#include "srsran/mac/config/mac_config_helpers.h"
#include <gtest/gtest.h>

namespace srsran {

// Helper function to create a sched_cell_configuration_request_message that allows a configuration with either 15kHz or
// 30kHz SCS. By default, it creates a bandwidth of 20MHz.
inline sched_cell_configuration_request_message
make_default_sched_cell_configuration_request_scs(subcarrier_spacing scs, bool tdd_mode = false)
{
  cell_config_builder_params params{
      .scs_common = scs, .channel_bw_mhz = bs_channel_bandwidth_fr1::MHz20, .nof_dl_ports = 1};
  if (scs == subcarrier_spacing::kHz15) {
    // Band n5 for FDD, band n41 for TDD.
    params.dl_arfcn = tdd_mode ? 499200 : 530000;
    params.band     = band_helper::get_band_from_dl_arfcn(params.dl_arfcn);
  } else if (scs == subcarrier_spacing::kHz30) {
    // Band n5 for FDD, band n77 or n78 for TDD.
    params.dl_arfcn = tdd_mode ? 630000 : 176000;
    params.band     = band_helper::get_band_from_dl_arfcn(params.dl_arfcn);
  }
  return sched_cell_configuration_request_message{test_helpers::make_default_sched_cell_configuration_request(params)};
}

////////////    Builder of PUCCH scheduler output     ////////////

// Builds the PUCCH scheduler output for test.
pucch_info build_pucch_info(const bwp_configuration* bwp_cfg,
                            unsigned                 pci,
                            pucch_format             format,
                            prb_interval             prbs,
                            prb_interval             second_hop_prbs,
                            ofdm_symbol_range        symbols,
                            uint8_t                  initial_cyclic_shift,
                            sr_nof_bits              sr_bits,
                            unsigned                 harq_ack_nof_bits,
                            uint8_t                  time_domain_occ);

bool pucch_info_match(const pucch_info& expected, const pucch_info& test);

// Wrapper for std::find_if() to find a PUCCH PDU in a vector of PUCCH PDUs.
template <typename F>
bool find_pucch_pdu(const static_vector<pucch_info, MAX_PUCCH_PDUS_PER_SLOT>& pucch_pdus, const F& func)
{
  return std::find_if(pucch_pdus.begin(), pucch_pdus.end(), func) != pucch_pdus.end();
}

// Makes a default DCI for PUCCH test purposes but some given parameters.
inline pdcch_dl_information make_default_dci(unsigned n_cces, const coreset_configuration* coreset_cfg_)
{
  pdcch_dl_information dci{};
  dci.ctx.cces.ncce   = n_cces;
  dci.ctx.coreset_cfg = coreset_cfg_;
  return dci;
}

inline sched_cell_configuration_request_message make_custom_sched_cell_configuration_request(unsigned pucch_res_common,
                                                                                             bool     is_tdd = false)
{
  sched_cell_configuration_request_message req = test_helpers::make_default_sched_cell_configuration_request(
      cell_config_builder_params{.scs_common     = is_tdd ? subcarrier_spacing::kHz30 : subcarrier_spacing::kHz15,
                                 .channel_bw_mhz = bs_channel_bandwidth_fr1::MHz10,
                                 .dl_arfcn       = is_tdd ? 520000U : 365000U});
  req.ul_cfg_common.init_ul_bwp.pucch_cfg_common->pucch_resource_common = pucch_res_common;
  return req;
}

/////////        TEST BENCH for PUCCH scheduler        /////////

struct test_bench_params {
  unsigned               pucch_res_common   = 11;
  unsigned               n_cces             = 0;
  sr_periodicity         period             = sr_periodicity::sl_40;
  unsigned               offset             = 0;
  csi_report_periodicity csi_period         = csi_report_periodicity::slots320;
  unsigned               csi_offset         = 9;
  bool                   is_tdd             = false;
  bool                   pucch_f2_more_prbs = false;
  bool                   cfg_for_mimo_4x4   = false;
  bool                   use_format_0       = false;
};

// Test bench with all that is needed for the PUCCH.
class test_bench
{
public:
  explicit test_bench(const test_bench_params& params                  = {},
                      unsigned                 max_pucchs_per_slot_    = 32U,
                      unsigned                 max_ul_grants_per_slot_ = 32U);

  // Return the main UE, which has RNTI 0x4601.
  const ue& get_main_ue() const;
  const ue& get_ue(du_ue_index_t ue_idx) const;

  // Add an extra UE, whose RNTI will have RNTI +1 with respect to the last_allocated_rnti.
  void add_ue();

  void slot_indication(slot_point slot_tx);

  void fill_all_grid(slot_point slot_tx);

  scheduler_expert_config                        expert_cfg;
  sched_cfg_dummy_notifier                       mac_notif;
  scheduler_harq_timeout_dummy_handler           harq_timeout_handler;
  cell_common_configuration_list                 cell_cfg_list{};
  const cell_configuration&                      cell_cfg;
  std::vector<std::unique_ptr<ue_configuration>> ue_ded_cfgs;

  cell_resource_allocator res_grid{cell_cfg};
  pdcch_dl_information    dci_info;
  const unsigned          k0;
  const unsigned          k1{4};
  const unsigned          max_pucchs_per_slot;
  const unsigned          max_ul_grants_per_slot;
  du_ue_index_t           main_ue_idx{du_ue_index_t::MIN_DU_UE_INDEX};
  ue_repository           ues;
  bool                    pucch_f2_more_prbs;
  const bool              use_format_0;

  // last_allocated_rnti keeps track of the last RNTI allocated.
  rnti_t                        last_allocated_rnti;
  du_ue_index_t                 last_allocated_ue_idx;
  pucch_allocator_impl          pucch_alloc;
  uci_allocator_impl            uci_alloc;
  uci_scheduler_impl            uci_sched;
  slot_point                    sl_tx;
  pucch_res_builder_test_helper pucch_builder;
  srslog::basic_logger&         mac_logger  = srslog::fetch_basic_logger("SCHED", true);
  srslog::basic_logger&         test_logger = srslog::fetch_basic_logger("TEST");
};

} // namespace srsran
