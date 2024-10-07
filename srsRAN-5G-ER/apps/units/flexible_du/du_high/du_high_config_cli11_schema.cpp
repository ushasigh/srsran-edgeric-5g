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

#include "du_high_config_cli11_schema.h"
#include "apps/services/logger/logger_appconfig_cli11_utils.h"
#include "apps/units/flexible_du/support/cli11_cpu_affinities_parser_helper.h"
#include "du_high_config.h"
#include "srsran/ran/du_types.h"
#include "srsran/ran/duplex_mode.h"
#include "srsran/support/cli11_utils.h"
#include "srsran/support/config_parsers.h"
#include "srsran/support/format_utils.h"

using namespace srsran;

template <typename Integer>
static expected<Integer, std::string> parse_int(const std::string& value)
{
  try {
    return std::stoi(value);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(e.what());
  } catch (const std::out_of_range& e) {
    return make_unexpected(e.what());
  }
}

/// Returns a default capture function for vectors of integers.
template <typename Integer>
static std::function<std::string()> get_vector_default_function(span<const Integer> value)
{
  static_assert(std::is_integral_v<Integer>, "Invalid Integer");

  return [value]() -> std::string {
    if (value.empty()) {
      return {};
    }

    fmt::memory_buffer buffer;
    fmt::format_to(buffer, "[");
    for (unsigned i = 0, e = value.size() - 1; i != e; ++i) {
      fmt::format_to(buffer, "{},", value[i]);
    }
    fmt::format_to(buffer, "{}]", value.back());

    return to_c_str(buffer);
  };
}

static void configure_cli11_log_args(CLI::App& app, du_high_unit_logger_config& log_params)
{
  app_services::add_log_option(app, log_params.mac_level, "--mac_level", "MAC log level");
  app_services::add_log_option(app, log_params.rlc_level, "--rlc_level", "RLC log level");
  app_services::add_log_option(app, log_params.f1ap_level, "--f1ap_level", "F1AP log level");
  app_services::add_log_option(app, log_params.f1u_level, "--f1u_level", "F1-U log level");
  app_services::add_log_option(app, log_params.du_level, "--du_level", "Log level for the DU");

  add_option(
      app, "--hex_max_size", log_params.hex_max_size, "Maximum number of bytes to print in hex (zero for no hex dumps)")
      ->capture_default_str()
      ->check(CLI::Range(0, 1024));
  add_option(app,
             "--broadcast_enabled",
             log_params.broadcast_enabled,
             "Enable logging in the physical and MAC layer of broadcast messages and all PRACH opportunities")
      ->always_capture_default();
  add_option(app, "--f1ap_json_enabled", log_params.f1ap_json_enabled, "Enable JSON logging of F1AP PDUs")
      ->always_capture_default();
}

static void configure_cli11_cell_affinity_args(CLI::App& app, du_high_unit_cpu_affinities_cell_config& config)
{
  add_option_function<std::string>(
      app,
      "--l2_cell_cpus",
      [&config](const std::string& value) { parse_affinity_mask(config.l2_cell_cpu_cfg.mask, value, "l2_cell_cpus"); },
      "CPU cores assigned to L2 cell tasks");

  add_option_function<std::string>(
      app,
      "--l2_cell_pinning",
      [&config](const std::string& value) {
        config.l2_cell_cpu_cfg.pinning_policy = to_affinity_mask_policy(value);
        if (config.l2_cell_cpu_cfg.pinning_policy == sched_affinity_mask_policy::last) {
          report_error("Incorrect value={} used in {} property", value, "l2_cell_pinning");
        }
      },
      "Policy used for assigning CPU cores to L2 cell tasks");
}

static void configure_cli11_expert_execution_args(CLI::App& app, du_high_unit_expert_execution_config& config)
{
  // Cell affinity section.
  add_option_cell(
      app,
      "--cell_affinities",
      [&config](const std::vector<std::string>& values) {
        config.cell_affinities.resize(values.size());
        for (unsigned i = 0, e = values.size(); i != e; ++i) {
          CLI::App subapp("DU high expert execution cell CPU affinities",
                          "DU high expert execution cell CPU affinities config, item #" + std::to_string(i));
          subapp.config_formatter(create_yaml_config_parser());
          subapp.allow_config_extras();
          configure_cli11_cell_affinity_args(subapp, config.cell_affinities[i]);
          std::istringstream ss(values[i]);
          subapp.parse_from_stream(ss);
        }
      },
      "Sets the cell CPU affinities configuration on a per cell basis");
}

static void configure_cli11_pdcch_common_args(CLI::App& app, pdcch_common_unit_config& common_params)
{
  add_option(app, "--coreset0_index", common_params.coreset0_index, "CORESET#0 index")
      ->capture_default_str()
      ->check(CLI::Range(0, 15));

  add_option(app,
             "--ss1_n_candidates",
             common_params.ss1_n_candidates,
             "Number of PDCCH candidates per aggregation level for SearchSpace#1. Default: {0, 0, 1, 0, 0}")
      ->default_function(get_vector_default_function(span<const uint8_t>(common_params.ss1_n_candidates)))
      ->capture_default_str()
      ->check(CLI::IsMember({0, 1, 2, 3, 4, 5, 6, 8}));

  add_option(app, "--ss0_index", common_params.ss0_index, "SearchSpace#0 index")
      ->capture_default_str()
      ->check(CLI::Range(0, 15));

  // NOTE: The CORESET duration of 3 symbols is only permitted if the dmrs-typeA-Position information element has
  // been set to 3. And, we use only pos2 or pos1.
  add_option(app,
             "--max_coreset0_duration",
             common_params.max_coreset0_duration,
             "Maximum CORESET#0 duration in OFDM symbols to consider when deriving CORESET#0 index")
      ->capture_default_str()
      ->check(CLI::Range(1, 2));
}

static void configure_cli11_pdcch_dedicated_args(CLI::App& app, pdcch_dedicated_unit_config& ded_params)
{
  add_option(app,
             "--coreset1_rb_start",
             ded_params.coreset1_rb_start,
             "Starting Common Resource Block (CRB) number for CORESET 1 relative to CRB 0. Default: CRB0")
      ->capture_default_str()
      ->check(CLI::Range(0, 275));

  add_option(app,
             "--coreset1_l_crb",
             ded_params.coreset1_l_crb,
             "Length of CORESET 1 in number of CRBs. Default: Across entire BW of cell")
      ->capture_default_str()
      ->check(CLI::Range(0, 275));

  // NOTE: The CORESET duration of 3 symbols is only permitted if the dmrs-typeA-Position information element has been
  // set to 3. And, we use only pos2 or pos1.
  add_option(app,
             "--coreset1_duration",
             ded_params.coreset1_duration,
             "Duration of CORESET 1 in number of OFDM symbols. Default: Max(2, Nof. CORESET#0 symbols)")
      ->capture_default_str()
      ->check(CLI::Range(1, 2));

  add_option(app,
             "--ss2_n_candidates",
             ded_params.ss2_n_candidates,
             "Number of PDCCH candidates per aggregation level for SearchSpace#2. Default: {0, 0, 0, 0, 0} i.e. "
             "auto-compute nof. candidates")
      ->default_function(get_vector_default_function(span<const uint8_t>(ded_params.ss2_n_candidates)))
      ->capture_default_str()
      ->check(CLI::IsMember({0, 1, 2, 3, 4, 5, 6, 8}));

  add_option(app,
             "--dci_format_0_1_and_1_1",
             ded_params.dci_format_0_1_and_1_1,
             "DCI format to use in UE dedicated SearchSpace#2")
      ->capture_default_str();
  add_option_function<std::string>(
      app,
      "--ss2_type",
      [&ded_params](const std::string& value) {
        ded_params.ss2_type = (value == "common") ? search_space_configuration::type_t::common
                                                  : search_space_configuration::type_t::ue_dedicated;
      },
      "SearchSpace type for UE dedicated SearchSpace#2")
      ->default_str("ue_dedicated")
      ->check(CLI::IsMember({"common", "ue_dedicated"}, CLI::ignore_case));
}

static void configure_cli11_pdcch_args(CLI::App& app, du_high_unit_pdcch_config& pdcch_params)
{
  // PDCCH Common configuration.
  CLI::App* pdcch_common_subcmd = add_subcommand(app, "common", "PDCCH Common configuration parameters");
  configure_cli11_pdcch_common_args(*pdcch_common_subcmd, pdcch_params.common);

  // PDCCH Dedicated configuration.
  CLI::App* pdcch_dedicated_subcmd = add_subcommand(app, "dedicated", "PDCCH Dedicated configuration parameters");
  configure_cli11_pdcch_dedicated_args(*pdcch_dedicated_subcmd, pdcch_params.dedicated);
}

static void configure_cli11_pdsch_args(CLI::App& app, du_high_unit_pdsch_config& pdsch_params)
{
  add_option(app, "--min_ue_mcs", pdsch_params.min_ue_mcs, "Minimum UE MCS")
      ->capture_default_str()
      ->check(CLI::Range(0, 28));
  add_option(app, "--max_ue_mcs", pdsch_params.max_ue_mcs, "Maximum UE MCS")
      ->capture_default_str()
      ->check(CLI::Range(0, 28));
  add_option(app, "--fixed_rar_mcs", pdsch_params.fixed_rar_mcs, "Fixed RAR MCS")
      ->capture_default_str()
      ->check(CLI::Range(0, 28));
  add_option(app, "--fixed_sib1_mcs", pdsch_params.fixed_sib1_mcs, "Fixed SIB1 MCS")
      ->capture_default_str()
      ->check(CLI::Range(0, 28));
  add_option(app, "--nof_harqs", pdsch_params.nof_harqs, "Number of DL HARQ processes")
      ->capture_default_str()
      ->check(CLI::IsMember({2, 4, 6, 8, 10, 12, 16}));
  add_option(app,
             "--max_nof_harq_retxs",
             pdsch_params.max_nof_harq_retxs,
             "Maximum number of times a DL HARQ can be retransmitted, before it gets discarded")
      ->capture_default_str()
      ->check(CLI::Range(0, 4));
  add_option(app,
             "--max_consecutive_kos",
             pdsch_params.max_consecutive_kos,
             "Maximum number of HARQ-ACK consecutive KOs before an Radio Link Failure is reported")
      ->capture_default_str();
  add_option(app, "--rv_sequence", pdsch_params.rv_sequence, "RV sequence for PUSCH. (e.g. [0 2 3 1]")
      ->capture_default_str()
      ->check(CLI::IsMember({0, 1, 2, 3}));
  add_option_function<std::string>(
      app,
      "--mcs_table",
      [&pdsch_params](const std::string& value) {
        if (value == "qam256") {
          pdsch_params.mcs_table = pdsch_mcs_table::qam256;
        }
      },
      "MCS table to use PDSCH")
      ->default_str("qam64")
      ->check(CLI::IsMember({"qam64", "qam256"}, CLI::ignore_case));
  add_option(app, "--min_rb_size", pdsch_params.min_rb_size, "Minimum RB size for UE PDSCH resource allocation")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)MAX_NOF_PRBS));
  add_option(app, "--max_rb_size", pdsch_params.max_rb_size, "Maximum RB size for UE PDSCH resource allocation")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)MAX_NOF_PRBS));
  add_option(app, "--start_rb", pdsch_params.start_rb, "Start RB for resource allocation of UE PDSCHs")
      ->capture_default_str()
      ->check(CLI::Range(0U, (unsigned)MAX_NOF_PRBS));
  add_option(app, "--end_rb", pdsch_params.end_rb, "End RB for resource allocation of UE PDSCHs")
      ->capture_default_str()
      ->check(CLI::Range(0U, (unsigned)MAX_NOF_PRBS));
  add_option(app,
             "--max_pdschs_per_slot",
             pdsch_params.max_pdschs_per_slot,
             "Maximum number of PDSCH grants per slot, including SIB, RAR, Paging and UE data grants.")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)MAX_PDSCH_PDUS_PER_SLOT));
  add_option(app,
             "--max_alloc_attempts",
             pdsch_params.max_pdcch_alloc_attempts_per_slot,
             "Maximum number of DL or UL PDCCH grant allocation attempts per slot before scheduler skips the slot")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)std::max(MAX_DL_PDCCH_PDUS_PER_SLOT, MAX_UL_PDCCH_PDUS_PER_SLOT)));
  add_option(app,
             "--olla_cqi_inc_step",
             pdsch_params.olla_cqi_inc,
             "Outer-loop link adaptation (OLLA) increment value. The value 0 means that OLLA is disabled")
      ->capture_default_str()
      ->check(CLI::Range(0.0, 1.0));
  add_option(app,
             "--olla_target_bler",
             pdsch_params.olla_target_bler,
             "Target DL BLER set in Outer-loop link adaptation (OLLA) algorithm")
      ->capture_default_str()
      ->check(CLI::Range(0.0, 0.5));
  add_option(app,
             "--olla_max_cqi_offset",
             pdsch_params.olla_max_cqi_offset,
             "Maximum offset that the Outer-loop link adaptation (OLLA) can apply to CQI")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  add_option_function<std::string>(
      app,
      "--dc_offset",
      [&pdsch_params](const std::string& value) {
        if (value == "undetermined") {
          pdsch_params.dc_offset = dc_offset_t::undetermined;
        } else if (value == "outside") {
          pdsch_params.dc_offset = dc_offset_t::outside;
        } else if (value == "center") {
          pdsch_params.dc_offset = dc_offset_t::center;
        } else {
          pdsch_params.dc_offset = static_cast<dc_offset_t>(parse_int<int>(value).value());
        }
      },
      "Direct Current (DC) Offset in number of subcarriers, using the common SCS as reference for carrier spacing, "
      "and the center of the gNB DL carrier as DC offset value 0. The user can additionally set \"outside\" to "
      "define that the DC offset falls outside the DL carrier or \"undetermined\" in the case the DC offset is "
      "unknown.")
      ->capture_default_str()
      ->check(CLI::Range(static_cast<int>(dc_offset_t::min), static_cast<int>(dc_offset_t::max)) |
              CLI::IsMember({"outside", "undetermined", "center"}));
  add_option<uint8_t>(app,
                      "--harq_la_cqi_drop_threshold",
                      pdsch_params.harq_la_cqi_drop_threshold,
                      "Link Adaptation (LA) threshold for drop in CQI of the first HARQ transmission above which HARQ "
                      "retransmissions are cancelled. Set this value to 0 to disable this feature")
      ->default_function([values = pdsch_params.harq_la_cqi_drop_threshold]() { return std::to_string(values); })
      ->capture_default_str()
      ->check(CLI::Range(0, 15));
  add_option(app,
             "--harq_la_ri_drop_threshold",
             pdsch_params.harq_la_ri_drop_threshold,
             "Link Adaptation (LA) threshold for drop in nof. layers of the first HARQ transmission above which "
             "HARQ retransmission is cancelled. Set this value to 0 to disable this feature")
      ->default_function([values = pdsch_params.harq_la_ri_drop_threshold]() { return std::to_string(values); })
      ->capture_default_str()
      ->check(CLI::Range(0, 4));
  add_option(app, "--dmrs_additional_position", pdsch_params.dmrs_add_pos, "PDSCH DMRS additional position")
      ->capture_default_str()
      ->check(CLI::Range(0, 3));
}

static void configure_cli11_du_args(CLI::App& app, bool& warn_on_drop)
{
  add_option(
      app, "--warn_on_drop", warn_on_drop, "Log a warning for dropped packets in F1-U, RLC and MAC due to full queues")
      ->capture_default_str();
}

static void configure_cli11_mac_bsr_args(CLI::App& app, mac_bsr_unit_config& bsr_params)
{
  add_option(app,
             "--periodic_bsr_timer",
             bsr_params.periodic_bsr_timer,
             "Periodic Buffer Status Report Timer value in nof. subframes. Value 0 equates to infinity")
      ->capture_default_str()
      ->check(CLI::IsMember({1, 5, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640, 1280, 2560, 0}));
  add_option(app,
             "--retx_bsr_timer",
             bsr_params.retx_bsr_timer,
             "Retransmission Buffer Status Report Timer value in nof. subframes")
      ->capture_default_str()
      ->check(CLI::IsMember({10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240}));
  add_option(
      app, "--lc_sr_delay_timer", bsr_params.lc_sr_delay_timer, "Logical Channel SR delay timer in nof. subframes")
      ->capture_default_str()
      ->check(CLI::IsMember({20, 40, 64, 128, 512, 1024, 2560}));
}

static void configure_cli11_mac_phr_args(CLI::App& app, mac_phr_unit_config& phr_params)
{
  add_option(app, "--phr_prohibit_timer", phr_params.phr_prohib_timer, "PHR prohibit timer in nof. subframes")
      ->capture_default_str()
      ->check(CLI::IsMember({0, 10, 20, 50, 100, 200, 500, 1000}));
}

static void configure_cli11_mac_sr_args(CLI::App& app, mac_sr_unit_config& sr_params)
{
  add_option(app, "--sr_trans_max", sr_params.sr_trans_max, "Maximum number of SR transmissions")
      ->capture_default_str()
      ->check(CLI::IsMember({4, 8, 16, 32, 64}));
  add_option(app, "--sr_prohibit_timer", sr_params.sr_prohibit_timer, "Timer for SR transmission on PUCCH in ms")
      ->check(CLI::IsMember({1, 2, 4, 8, 16, 32, 64, 128}));
}

static void configure_cli11_mac_cell_group_args(CLI::App& app, du_high_unit_mac_cell_group_config& mcg_params)
{
  // BSR configuration.
  CLI::App* bsr_subcmd = app.add_subcommand("bsr_cfg", "Buffer status report configuration parameters");
  configure_cli11_mac_bsr_args(*bsr_subcmd, mcg_params.bsr_cfg);
  CLI::App* phr_subcmd = app.add_subcommand("phr_cfg", "Power Headroom report configuration parameters");
  configure_cli11_mac_phr_args(*phr_subcmd, mcg_params.phr_cfg);
  CLI::App* sr_subcmd = app.add_subcommand("sr_cfg", "Scheduling Request configuration parameters");
  configure_cli11_mac_sr_args(*sr_subcmd, mcg_params.sr_cfg);
}

static void configure_cli11_ssb_args(CLI::App& app, du_high_unit_ssb_config& ssb_params)
{
  add_option(app, "--ssb_period", ssb_params.ssb_period_msec, "Period of SSB scheduling in milliseconds")
      ->capture_default_str()
      ->check(CLI::IsMember({5, 10, 20}));
  add_option(app, "--ssb_block_power_dbm", ssb_params.ssb_block_power, "SS_PBCH_power_block in dBm")
      ->capture_default_str()
      ->check(CLI::Range(-60, 50));
  add_option_function<std::string>(
      app,
      "--pss_to_sss_epre_db",
      [&ssb_params](const std::string& value) {
        unsigned temp_value = CLI::detail::lexical_cast(value, temp_value);
        if (temp_value == 0) {
          ssb_params.pss_to_sss_epre = srsran::ssb_pss_to_sss_epre::dB_0;
        } else if (temp_value == 3) {
          ssb_params.pss_to_sss_epre = srsran::ssb_pss_to_sss_epre::dB_3;
        }
      },
      "SSB PSS to SSS EPRE ratio in dB {0, 3}")
      ->check(CLI::IsMember({0, 3}));
}

static void configure_cli11_tdd_ul_dl_pattern_args(CLI::App& app, tdd_ul_dl_pattern_unit_config& pattern_params)
{
  add_option(app,
             "--dl_ul_tx_period",
             pattern_params.dl_ul_period_slots,
             "TDD pattern periodicity in slots. The combination of this value and the chosen numerology must lead"
             " to a TDD periodicity of 0.5, 0.625, 1, 1.25, 2, 2.5, 3, 4, 5 or 10 milliseconds.")
      ->capture_default_str()
      ->check(CLI::Range(2, 80));
  add_option(app, "--nof_dl_slots", pattern_params.nof_dl_slots, "TDD pattern nof. consecutive full DL slots")
      ->capture_default_str()
      ->check(CLI::Range(0, 80));
  add_option(app,
             "--nof_dl_symbols",
             pattern_params.nof_dl_symbols,
             "TDD pattern nof. DL symbols at the beginning of the slot following full DL slots")
      ->capture_default_str()
      ->check(CLI::Range(0, 13));
  add_option(app, "--nof_ul_slots", pattern_params.nof_ul_slots, "TDD pattern nof. consecutive full UL slots")
      ->capture_default_str()
      ->check(CLI::Range(0, 80));
  add_option(app,
             "--nof_ul_symbols",
             pattern_params.nof_ul_symbols,
             "TDD pattern nof. UL symbols at the end of the slot preceding the first full UL slot")
      ->capture_default_str()
      ->check(CLI::Range(0, 13));
}

static void configure_cli11_tdd_ul_dl_args(CLI::App& app, du_high_unit_tdd_ul_dl_config& tdd_ul_dl_params)
{
  configure_cli11_tdd_ul_dl_pattern_args(app, tdd_ul_dl_params.pattern1);
  // TDD pattern 2.
  // NOTE: As the pattern2 can be configured in the common cell and not in the cell, use a different variable to parse
  // the cell pattern 2 property. If the pattern 2 is present for the cell, update the cell pattern 2 value, otherwise
  // do nothing (this will cause that the cell pattern 2 value equals than the common cell TDD pattern 2). CLI11 needs
  // that the life of the variable last longer than the call of callback function. Therefore, the pattern2_cfg variable
  // needs to be static.
  static tdd_ul_dl_pattern_unit_config pattern2_cfg;
  CLI::App*                            pattern2_sub_cmd =
      add_subcommand(app, "pattern2", "TDD UL DL pattern2 configuration parameters")->configurable();
  configure_cli11_tdd_ul_dl_pattern_args(*pattern2_sub_cmd, pattern2_cfg);
  auto tdd_pattern2_verify_callback = [&]() {
    CLI::App* sub_cmd = app.get_subcommand("pattern2");
    if (sub_cmd->count() != 0) {
      tdd_ul_dl_params.pattern2.emplace(pattern2_cfg);
    }
    if (!tdd_ul_dl_params.pattern2.has_value()) {
      pattern2_sub_cmd->disabled();
    }
  };
  pattern2_sub_cmd->parse_complete_callback(tdd_pattern2_verify_callback);
}

static void configure_cli11_paging_args(CLI::App& app, du_high_unit_paging_config& pg_params)
{
  add_option(app, "--pg_search_space_id", pg_params.paging_search_space_id, "SearchSpace to use for Paging")
      ->capture_default_str()
      ->check(CLI::IsMember({0, 1}));
  add_option(
      app, "--default_pg_cycle_in_rf", pg_params.default_paging_cycle, "Default Paging cycle in nof. Radio Frames")
      ->capture_default_str()
      ->check(CLI::IsMember({32, 64, 128, 256}));
  add_option_function<std::string>(
      app,
      "--nof_pf_per_paging_cycle",
      [&pg_params](const std::string& value) {
        if (value == "oneT") {
          pg_params.nof_pf = pcch_config::nof_pf_per_drx_cycle::oneT;
        } else if (value == "halfT") {
          pg_params.nof_pf = pcch_config::nof_pf_per_drx_cycle::halfT;
        } else if (value == "quarterT") {
          pg_params.nof_pf = pcch_config::nof_pf_per_drx_cycle::quarterT;
        } else if (value == "oneEighthT") {
          pg_params.nof_pf = pcch_config::nof_pf_per_drx_cycle::oneEighthT;
        } else if (value == "oneSixteethT") {
          pg_params.nof_pf = pcch_config::nof_pf_per_drx_cycle::oneSixteethT;
        }
      },
      "Number of paging frames per DRX cycle {oneT, halfT, quarterT, oneEighthT, oneSixteethT}. Default: oneT")
      ->check(CLI::IsMember({"oneT", "halfT", "quarterT", "oneEighthT", "oneSixteethT"}, CLI::ignore_case));
  add_option(app, "--pf_offset", pg_params.pf_offset, "Paging frame offset")->capture_default_str();
  add_option(app, "--nof_po_per_pf", pg_params.nof_po_per_pf, "Number of paging occasions per paging frame")
      ->capture_default_str()
      ->check(CLI::IsMember({1, 2, 4}));
}

static void configure_cli11_csi_args(CLI::App& app, du_high_unit_csi_config& csi_params)
{
  add_option(app, "--csi_rs_enabled", csi_params.csi_rs_enabled, "Enable CSI-RS resources and CSI reporting")
      ->capture_default_str();
  add_option(app, "--csi_rs_period", csi_params.csi_rs_period_msec, "CSI-RS period in milliseconds")
      ->capture_default_str()
      ->check(CLI::IsMember({10, 20, 40, 80}));
  add_option(app,
             "--meas_csi_rs_slot_offset",
             csi_params.meas_csi_slot_offset,
             "Slot offset of first CSI-RS resource used for measurement")
      ->capture_default_str();
  add_option(app,
             "--tracking_csi_rs_slot_offset",
             csi_params.tracking_csi_slot_offset,
             "Slot offset of first CSI-RS resource used for tracking")
      ->capture_default_str();
  add_option(app, "--zp_csi_rs_slot_offset", csi_params.zp_csi_slot_offset, "Slot offset of the ZP CSI-RS resources")
      ->capture_default_str();
  add_option(app,
             "--pwr_ctrl_offset",
             csi_params.pwr_ctrl_offset,
             "powerControlOffset, Power offset of PDSCH RE to NZP CSI-RS RE in dB")
      ->capture_default_str()
      ->check(CLI::Range(-8, 15));
}

static void configure_cli11_pf_scheduler_expert_args(CLI::App& app, time_pf_scheduler_expert_config& expert_params)
{
  add_option(app,
             "--pf_sched_fairness_coeff",
             expert_params.pf_sched_fairness_coeff,
             "Fairness Coefficient to use in Proportional Fair policy scheduler")
      ->capture_default_str();
}

static void configure_cli11_policy_scheduler_expert_args(CLI::App&                             app,
                                                         du_high_unit_scheduler_expert_config& expert_params)
{
  static time_pf_scheduler_expert_config pf_sched_expert_cfg;
  CLI::App*                              pf_sched_cfg_subcmd =
      add_subcommand(app, "pf_sched", "Proportional Fair policy scheduler expert configuration")->configurable();
  configure_cli11_pf_scheduler_expert_args(*pf_sched_cfg_subcmd, pf_sched_expert_cfg);
  auto pf_sched_verify_callback = [&]() {
    CLI::App* pf_sched_sub_cmd = app.get_subcommand("pf_sched");
    if (pf_sched_sub_cmd->count() != 0) {
      expert_params.policy_sched_expert_cfg = pf_sched_expert_cfg;
    }
  };
  pf_sched_cfg_subcmd->parse_complete_callback(pf_sched_verify_callback);
}

static void configure_cli11_scheduler_expert_args(CLI::App& app, du_high_unit_scheduler_expert_config& expert_params)
{
  CLI::App* policy_sched_cfg_subcmd =
      add_subcommand(app, "policy_sched_cfg", "Policy scheduler expert configuration")->configurable();
  configure_cli11_policy_scheduler_expert_args(*policy_sched_cfg_subcmd, expert_params);
}

static void configure_cli11_ul_common_args(CLI::App& app, du_high_unit_ul_common_config& ul_common_params)
{
  add_option(app, "--p_max", ul_common_params.p_max, "Maximum transmit power allowed in this serving cell")
      ->capture_default_str()
      ->check(CLI::Range(-30, 23));
  add_option(app,
             "--max_pucchs_per_slot",
             ul_common_params.max_pucchs_per_slot,
             "Maximum number of PUCCH grants that can be allocated per slot")
      ->capture_default_str()
      ->check(CLI::Range(1U, static_cast<unsigned>(MAX_PUCCH_PDUS_PER_SLOT)));
  add_option(app,
             "--max_ul_grants_per_slot",
             ul_common_params.max_ul_grants_per_slot,
             "Maximum number of UL grants that can be allocated per slot")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)(MAX_PUSCH_PDUS_PER_SLOT + MAX_PUCCH_PDUS_PER_SLOT)));
}

static void configure_cli11_pusch_args(CLI::App& app, du_high_unit_pusch_config& pusch_params)
{
  add_option(app, "--min_ue_mcs", pusch_params.min_ue_mcs, "Minimum UE MCS")
      ->capture_default_str()
      ->check(CLI::Range(0, 28));
  add_option(app, "--max_ue_mcs", pusch_params.max_ue_mcs, "Maximum UE MCS")
      ->capture_default_str()
      ->check(CLI::Range(0, 28));
  add_option(app,
             "--max_consecutive_kos",
             pusch_params.max_consecutive_kos,
             "Maximum number of CRC consecutive KOs before an Radio Link Failure is reported")
      ->capture_default_str();
  add_option(app, "--rv_sequence", pusch_params.rv_sequence, "RV sequence for PUSCH. (e.g. [0 2 3 1]")
      ->capture_default_str()
      ->check(CLI::IsMember({0, 1, 2, 3}));
  add_option_function<std::string>(
      app,
      "--mcs_table",
      [&pusch_params](const std::string& value) {
        if (value == "qam256") {
          pusch_params.mcs_table = pusch_mcs_table::qam256;
        }
      },
      "MCS table to use PUSCH")
      ->default_str("qam64")
      ->check(CLI::IsMember({"qam64", "qam256"}, CLI::ignore_case));
  add_option(app,
             "--msg3_delta_preamble",
             pusch_params.msg3_delta_preamble,
             "msg3-DeltaPreamble, Power offset between msg3 and RACH preamble transmission")
      ->capture_default_str()
      ->check(CLI::Range(-1, 6));
  add_option(app,
             "--p0_nominal_with_grant",
             pusch_params.p0_nominal_with_grant,
             "P0 value for PUSCH with grant (except msg3). Value in dBm. Valid values must be multiple of 2 and "
             "within the [-202, 24] interval.  Default: -76")
      ->capture_default_str()
      ->check([](const std::string& value) -> std::string {
        std::stringstream ss(value);
        int               pw;
        ss >> pw;
        const std::string& error_message = "Must be a multiple of 2 and within the [-202, 24] interval";
        if (pw < -202 or pw > 24 or pw % 2 != 0) {
          return error_message;
        }

        return "";
      });
  add_option(app,
             "--msg3_delta_power",
             pusch_params.msg3_delta_power,
             "Target power level at the network receiver side, in dBm. Valid values must be multiple of 2 and "
             "within the [-6, 8] interval. Default: 8")
      ->capture_default_str()
      ->check([](const std::string& value) -> std::string {
        std::stringstream ss(value);
        int               pw;
        ss >> pw;
        const std::string& error_message = "Must be a multiple of 2 and within the [-6, 8] interval";
        if (pw < -6 or pw > 8 or pw % 2 != 0) {
          return error_message;
        }

        return "";
      });
  add_option(app, "--max_puschs_per_slot", pusch_params.max_puschs_per_slot, "Maximum number of PUSCH grants per slot")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)MAX_PUSCH_PDUS_PER_SLOT));
  add_option(app, "--b_offset_ack_idx_1", pusch_params.b_offset_ack_idx_1, "betaOffsetACK-Index1 part of UCI-OnPUSCH")
      ->capture_default_str()
      ->check(CLI::Range(0, 31));
  add_option(app, "--b_offset_ack_idx_2", pusch_params.b_offset_ack_idx_2, "betaOffsetACK-Index2 part of UCI-OnPUSCH")
      ->capture_default_str()
      ->check(CLI::Range(0, 31));
  add_option(app, "--b_offset_ack_idx_3", pusch_params.b_offset_ack_idx_3, "betaOffsetACK-Index3 part of UCI-OnPUSCH")
      ->capture_default_str()
      ->check(CLI::Range(0, 31));
  add_option(app,
             "--beta_offset_csi_p1_idx_1",
             pusch_params.b_offset_csi_p1_idx_1,
             "b_offset_csi_p1_idx_1 part of UCI-OnPUSCH")
      ->capture_default_str()
      ->check(CLI::Range(0, 31));
  add_option(app,
             "--beta_offset_csi_p1_idx_2",
             pusch_params.b_offset_csi_p1_idx_2,
             "b_offset_csi_p1_idx_2 part of UCI-OnPUSCH")
      ->capture_default_str()
      ->check(CLI::Range(0, 31));
  add_option(app,
             "--beta_offset_csi_p2_idx_1",
             pusch_params.b_offset_csi_p2_idx_1,
             "b_offset_csi_p2_idx_1 part of UCI-OnPUSCH")
      ->capture_default_str()
      ->check(CLI::Range(0, 31));
  add_option(app,
             "--beta_offset_csi_p2_idx_2",
             pusch_params.b_offset_csi_p2_idx_2,
             "b_offset_csi_p2_idx_2 part of UCI-OnPUSCH")
      ->capture_default_str()
      ->check(CLI::Range(0, 31));
  add_option(app, "--min_k2", pusch_params.min_k2, "Minimum value of K2 (difference in slots between PDCCH and PUSCH).")
      ->capture_default_str()
      ->check(CLI::Range(1, 4));
  add_option_function<std::string>(
      app,
      "--dc_offset",
      [&pusch_params](const std::string& value) {
        if (value == "undetermined") {
          pusch_params.dc_offset = dc_offset_t::undetermined;
        } else if (value == "outside") {
          pusch_params.dc_offset = dc_offset_t::outside;
        } else if (value == "center") {
          pusch_params.dc_offset = dc_offset_t::center;
        } else {
          pusch_params.dc_offset = static_cast<dc_offset_t>(parse_int<int>(value).value());
        }
      },
      "Direct Current (DC) Offset in number of subcarriers, using the common SCS as reference for carrier spacing, "
      "and the center of the gNB UL carrier as DC offset value 0. The user can additionally set \"outside\" to "
      "define that the DC offset falls outside the UL carrier or \"undetermined\" in the case the DC offset is "
      "unknown.")
      ->capture_default_str()
      ->check(CLI::Range(static_cast<int>(dc_offset_t::min), static_cast<int>(dc_offset_t::max)) |
              CLI::IsMember({"outside", "undetermined", "center"}));
  add_option(app,
             "--olla_snr_inc_step",
             pusch_params.olla_snr_inc,
             "Outer-loop link adaptation (OLLA) increment value. The value 0 means that OLLA is disabled")
      ->capture_default_str()
      ->check(CLI::Range(0.0, 1.0));
  add_option(app,
             "--olla_target_bler",
             pusch_params.olla_target_bler,
             "Target UL BLER set in Outer-loop link adaptation (OLLA) algorithm")
      ->capture_default_str()
      ->check(CLI::Range(0.0, 0.5));
  add_option(app,
             "--olla_max_snr_offset",
             pusch_params.olla_max_snr_offset,
             "Maximum offset that the Outer-loop link adaptation (OLLA) can apply to the estimated UL SINR")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  add_option(app, "--dmrs_additional_position", pusch_params.dmrs_add_pos, "PUSCH DMRS additional position")
      ->capture_default_str()
      ->check(CLI::Range(0, 3));
  add_option(app, "--min_rb_size", pusch_params.min_rb_size, "Minimum RB size for UE PUSCH resource allocation")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)MAX_NOF_PRBS));
  add_option(app, "--max_rb_size", pusch_params.max_rb_size, "Maximum RB size for UE PUSCH resource allocation")
      ->capture_default_str()
      ->check(CLI::Range(1U, (unsigned)MAX_NOF_PRBS));
  app.add_option("--start_rb", pusch_params.start_rb, "Start RB for resource allocation of UE PUSCHs")
      ->capture_default_str()
      ->check(CLI::Range(0U, (unsigned)MAX_NOF_PRBS));
  app.add_option("--end_rb", pusch_params.end_rb, "End RB for resource allocation of UE PUSCHs")
      ->capture_default_str()
      ->check(CLI::Range(0U, (unsigned)MAX_NOF_PRBS));
}

static void configure_cli11_pucch_args(CLI::App& app, du_high_unit_pucch_config& pucch_params)
{
  add_option(app,
             "--p0_nominal",
             pucch_params.p0_nominal,
             "Power control parameter P0 for PUCCH transmissions. Value in dBm. Valid values must be multiple of 2 and "
             "within the [-202, 24] interval. Default: -90")
      ->capture_default_str()
      ->check([](const std::string& value) -> std::string {
        std::stringstream ss(value);
        int               pw;
        ss >> pw;
        const std::string& error_message = "Must be a multiple of 2 and within the [-202, 24] interval";
        if (pw < -202 or pw > 24 or pw % 2 != 0) {
          return error_message;
        }

        return "";
      });
  add_option(app,
             "--pucch_resource_common",
             pucch_params.pucch_resource_common,
             "Index of PUCCH resource set for the common configuration")
      ->capture_default_str()
      ->check(CLI::Range(0, 15));
  add_option(app, "--sr_period_ms", pucch_params.sr_period_msec, "SR period in msec")
      ->capture_default_str()
      ->check(CLI::IsMember({1.0F, 2.0F, 2.5F, 4.0F, 5.0F, 8.0F, 10.0F, 16.0F, 20.0F, 40.0F, 80.0F, 160.0F, 320.0F}));
  add_option(app, "--use_format_0", pucch_params.use_format_0, "Use Format 0 for PUCCH resources from resource set 0")
      ->capture_default_str();
  add_option(app,
             "--f1_nof_ue_res_harq",
             pucch_params.nof_ue_pucch_f0_or_f1_res_harq,
             "Number of PUCCH F0/F1 resources available per UE for HARQ")
      ->capture_default_str()
      ->check(CLI::Range(1, 8));
  add_option(app,
             "--f0_or_f1_nof_cell_res_sr",
             pucch_params.nof_cell_sr_resources,
             "Number of PUCCH F1 resources available per cell for SR")
      ->capture_default_str()
      ->check(CLI::Range(1, 50));
  add_option(app,
             "--f0_intraslot_freq_hop",
             pucch_params.f0_intraslot_freq_hopping,
             "Enable intra-slot frequency hopping for PUCCH F0")
      ->capture_default_str();
  add_option(app, "--f1_enable_occ", pucch_params.f1_enable_occ, "Enable OCC for PUCCH F1")->capture_default_str();
  add_option(app,
             "--f1_nof_cyclic_shifts",
             pucch_params.nof_cyclic_shift,
             "Number of possible cyclic shifts available for PUCCH F1 resources")
      ->capture_default_str()
      ->check(CLI::IsMember({1, 2, 3, 4, 6, 12}));
  add_option(app,
             "--f1_intraslot_freq_hop",
             pucch_params.f1_intraslot_freq_hopping,
             "Enable intra-slot frequency hopping for PUCCH F1")
      ->capture_default_str();
  add_option(app,
             "--nof_cell_harq_pucch_res_sets",
             pucch_params.nof_cell_harq_pucch_sets,
             "Number of separate PUCCH resource sets for HARQ-ACK that are available in the cell. NOTE: the "
             "higher the number of sets, the lower the chances UEs have to share the same PUCCH resources.")
      ->capture_default_str()
      ->check(CLI::Range(1, 10));
  add_option(app,
             "--f2_nof_ue_res_harq",
             pucch_params.nof_ue_pucch_f2_res_harq,
             "Number of PUCCH F2 resources available per UE for HARQ")
      ->capture_default_str()
      ->check(CLI::Range(1, 8));
  add_option(app,
             "--f2_nof_cell_res_csi",
             pucch_params.nof_cell_csi_resources,
             "Number of PUCCH F2 resources available per cell for CSI")
      ->capture_default_str()
      ->check(CLI::Range(0, 50));
  add_option(app, "--f2_max_nof_rbs", pucch_params.f2_max_nof_rbs, "Max number of RBs for PUCCH F2 resources")
      ->capture_default_str()
      ->check(CLI::Range(1, 16));
  add_option(app, "--f2_max_payload", pucch_params.max_payload_bits, "Max number payload bits for PUCCH F2 resources")
      ->check(CLI::Range(1, 11));
  add_option_function<std::string>(
      app,
      "--f2_max_code_rate",
      [&pucch_params](const std::string& value) {
        if (value == "dot08") {
          pucch_params.max_code_rate = max_pucch_code_rate::dot_08;
        } else if (value == "dot15") {
          pucch_params.max_code_rate = max_pucch_code_rate::dot_15;
        } else if (value == "dot25") {
          pucch_params.max_code_rate = max_pucch_code_rate::dot_25;
        } else if (value == "dot35") {
          pucch_params.max_code_rate = max_pucch_code_rate::dot_35;
        } else if (value == "dot45") {
          pucch_params.max_code_rate = max_pucch_code_rate::dot_45;
        } else if (value == "dot60") {
          pucch_params.max_code_rate = max_pucch_code_rate::dot_60;
        } else if (value == "dot80") {
          pucch_params.max_code_rate = max_pucch_code_rate::dot_80;
        }
      },
      "PUCCH F2 max code rate {dot08, dot15, dot25, dot35, dot45, dot60, dot80}. Default: dot35")
      ->check(CLI::IsMember({"dot08", "dot15", "dot25", "dot35", "dot45", "dot60", "dot80"}, CLI::ignore_case));
  add_option(app,
             "--f2_intraslot_freq_hop",
             pucch_params.f2_intraslot_freq_hopping,
             "Enable intra-slot frequency hopping for PUCCH F2")
      ->capture_default_str();
  add_option(app,
             "--min_k1",
             pucch_params.min_k1,
             "Minimum value of K1 (difference in slots between PDSCH and HARQ-ACK). Lower k1 values will reduce "
             "latency, but place a stricter requirement on the UE decode latency.")
      ->capture_default_str()
      ->check(CLI::Range(1, 4));

  add_option(app,
             "--max_consecutive_kos",
             pucch_params.max_consecutive_kos,
             "Maximum number of consecutive undecoded PUCCH F2 for CSI before an Radio Link Failure is reported")
      ->capture_default_str();
}

static void configure_cli11_si_sched_info(CLI::App& app, du_high_unit_sib_config::si_sched_info_config& si_sched_info)
{
  add_option(app, "--si_period", si_sched_info.si_period_rf, "SI message scheduling period in radio frames")
      ->capture_default_str()
      ->check(CLI::IsMember({8, 16, 32, 64, 128, 256, 512}));
  add_option(app,
             "--sib_mapping",
             si_sched_info.sib_mapping_info,
             "Mapping of SIB types to SI-messages. SIB numbers should not be repeated")
      ->default_function(get_vector_default_function(span<const uint8_t>(si_sched_info.sib_mapping_info)))
      ->capture_default_str()
      ->check(CLI::IsMember({2, 19}));
  add_option(
      app, "--si_window_position", si_sched_info.si_window_position, "SI window position of the associated SI-message")
      ->capture_default_str()
      ->check(CLI::Range(1, 256));
}

static void configure_cli11_prach_args(CLI::App& app, du_high_unit_prach_config& prach_params)
{
  add_option(app,
             "--prach_config_index",
             prach_params.prach_config_index,
             "PRACH configuration index. If not set, the value is derived, so that the PRACH fits in an UL slot")
      ->capture_default_str()
      ->check(CLI::Range(0, 255));
  add_option(app, "--prach_root_sequence_index", prach_params.prach_root_sequence_index, "PRACH root sequence index")
      ->capture_default_str()
      ->check(CLI::Range(0, 837));
  add_option(app, "--zero_correlation_zone", prach_params.zero_correlation_zone, "Zero correlation zone index")
      ->capture_default_str()
      ->check(CLI::Range(0, 15));
  add_option(app, "--fixed_msg3_mcs", prach_params.fixed_msg3_mcs, "Fixed message 3 MCS")
      ->capture_default_str()
      ->check(CLI::Range(0, 28));
  add_option(
      app, "--max_msg3_harq_retx", prach_params.max_msg3_harq_retx, "Maximum number of message 3 HARQ retransmissions")
      ->capture_default_str()
      ->check(CLI::Range(0, 4));
  add_option(
      app, "--total_nof_ra_preambles", prach_params.total_nof_ra_preambles, "Number of different PRACH preambles")
      ->check(CLI::Range(1, 64));
  add_option(
      app, "--prach_frequency_start", prach_params.prach_frequency_start, "PRACH message frequency offset in PRBs")
      ->capture_default_str()
      ->check(CLI::Range(0, 274));
  add_option(app,
             "--preamble_rx_target_pw",
             prach_params.preamble_rx_target_pw,
             "Target power level at the network receiver side, in dBm")
      ->capture_default_str()
      ->check([](const std::string& value) -> std::string {
        std::stringstream ss(value);
        int               pw;
        ss >> pw;
        const std::string& error_message = "Must be a multiple of 2 and within the [-202, -60] interval";
        // Bandwidth cannot be less than 5MHz.
        if (pw < -202 or pw > -60 or pw % 2 != 0) {
          return error_message;
        }

        return "";
      });
  add_option(app,
             "--preamble_trans_max",
             prach_params.preamble_trans_max,
             "Max number of RA preamble transmissions performed before declaring a failure")
      ->default_function([value = prach_params.preamble_trans_max]() { return std::to_string(value); })
      ->capture_default_str()
      ->check(CLI::IsMember({3, 4, 5, 6, 7, 8, 10, 20, 50, 100, 200}));
  add_option(app, "--power_ramping_step_db", prach_params.power_ramping_step_db, "Power ramping steps for PRACH")
      ->default_function([value = prach_params.power_ramping_step_db]() { return std::to_string(value); })
      ->capture_default_str()
      ->check(CLI::IsMember({0, 2, 4, 6}));
  add_option(app, "--ports", prach_params.ports, "List of antenna ports")
      ->default_function(get_vector_default_function(span<const uint8_t>(prach_params.ports)))
      ->capture_default_str();
  add_option(app, "--nof_ssb_per_ro", prach_params.nof_ssb_per_ro, "Number of SSBs per RACH occasion")
      ->check(CLI::IsMember({1}));
  add_option(app,
             "--nof_cb_preambles_per_ssb",
             prach_params.nof_cb_preambles_per_ssb,
             "Number of Contention Based preambles per SSB")
      ->default_function([&value = prach_params.nof_cb_preambles_per_ssb]() { return std::to_string(value); })
      ->check(CLI::Range(1, 64));
}

static void configure_cli11_sib_args(CLI::App& app, du_high_unit_sib_config& sib_params)
{
  add_option(app,
             "--si_window_length",
             sib_params.si_window_len_slots,
             "The length of the SI scheduling window, in slots. It must be always shorter or equal to the period of "
             "the SI message.")
      ->capture_default_str()
      ->check(CLI::IsMember({5, 10, 20, 40, 80, 160, 320, 640, 1280}));

  // SI message scheduling parameters.
  add_option_cell(
      app,
      "--si_sched_info",
      [&sib_params](const std::vector<std::string>& values) {
        sib_params.si_sched_info.resize(values.size());

        for (unsigned i = 0, e = values.size(); i != e; ++i) {
          CLI::App subapp("SI-message scheduling information",
                          "SI-message scheduling information config, item #" + std::to_string(i));
          subapp.config_formatter(create_yaml_config_parser());
          subapp.allow_config_extras(CLI::config_extras_mode::capture);
          configure_cli11_si_sched_info(subapp, sib_params.si_sched_info[i]);
          std::istringstream ss(values[i]);
          subapp.parse_from_stream(ss);
        }
      },
      "Configures the scheduling for each of the SI-messages broadcast by the gNB");

  add_option(app,
             "--t300",
             sib_params.ue_timers_and_constants.t300,
             "RRC Connection Establishment timer in ms. The timer starts upon transmission of RRCSetupRequest.")
      ->capture_default_str()
      ->check(CLI::IsMember({100, 200, 300, 400, 600, 1000, 1500, 2000}));
  add_option(app,
             "--t301",
             sib_params.ue_timers_and_constants.t301,
             "RRC Connection Re-establishment timer in ms. The timer starts upon transmission of "
             "RRCReestablishmentRequest.")
      ->capture_default_str()
      ->check(CLI::IsMember({100, 200, 300, 400, 600, 1000, 1500, 2000}));
  add_option(app,
             "--t310",
             sib_params.ue_timers_and_constants.t310,
             "Out-of-sync timer in ms. The timer starts upon detecting physical layer problems for the SpCell i.e. "
             "upon receiving N310 consecutive out-of-sync indications from lower layers.")
      ->capture_default_str()
      ->check(CLI::IsMember({0, 50, 100, 200, 500, 1000, 2000}));
  add_option(app,
             "--n310",
             sib_params.ue_timers_and_constants.n310,
             "Out-of-sync counter. The counter is increased upon reception of \"out-of-sync\" from lower layer "
             "while the timer T310 is stopped. Starts the timer T310, when configured value is reached.")
      ->capture_default_str()
      ->check(CLI::IsMember({1, 2, 3, 4, 6, 8, 10, 20}));
  add_option(app,
             "--t311",
             sib_params.ue_timers_and_constants.t311,
             "RRC Connection Re-establishment procedure timer in ms. The timer starts upon initiating the RRC "
             "connection re-establishment procedure.")
      ->capture_default_str()
      ->check(CLI::IsMember({1000, 3000, 5000, 10000, 15000, 20000, 30000}));
  add_option(app,
             "--n311",
             sib_params.ue_timers_and_constants.n311,
             "In-sync counter. The counter is increased upon reception of the \"in-sync\" from lower layer while "
             "the timer T310 is running. Stops the timer T310, when configured value is reached.")
      ->capture_default_str()
      ->check(CLI::IsMember({1, 2, 3, 4, 5, 6, 8, 10}));
  add_option(app,
             "--t319",
             sib_params.ue_timers_and_constants.t319,
             "RRC Connection Resume timer in ms. The timer starts upon transmission of RRCResumeRequest "
             "or RRCResumeRequest1.")
      ->capture_default_str()
      ->check(CLI::IsMember({100, 200, 300, 400, 600, 1000, 1500, 2000}));
}

static void configure_cli11_slicing_scheduling_args(CLI::App&                             app,
                                                    du_high_unit_cell_slice_sched_config& slice_sched_params)
{
  add_option(app,
             "--min_prb_policy_ratio",
             slice_sched_params.min_prb_policy_ratio,
             "Minimum percentage of PRBs to be allocated to the slice")
      ->capture_default_str()
      ->check(CLI::Range(0U, 100U));
  add_option(app,
             "--max_prb_policy_ratio",
             slice_sched_params.max_prb_policy_ratio,
             "Maximum percentage of PRBs to be allocated to the slice")
      ->capture_default_str()
      ->check(CLI::Range(1U, 100U));
}

static void configure_cli11_slicing_args(CLI::App& app, du_high_unit_cell_slice_config& slice_params)
{
  add_option(app, "--sst", slice_params.s_nssai.sst, "Slice Service Type")
      ->capture_default_str()
      ->check(CLI::Range(0, 255));
  add_option(app, "--sd", slice_params.s_nssai.sd, "Service Differentiator")
      ->capture_default_str()
      ->check(CLI::Range(0, 0xffffff));

  // Scheduling configuration.
  CLI::App* sched_cfg_subcmd = add_subcommand(app, "sched_cfg", "Slice scheduling configuration")->configurable();
  configure_cli11_slicing_scheduling_args(*sched_cfg_subcmd, slice_params.sched_cfg);
}

static void configure_cli11_common_cell_args(CLI::App& app, du_high_unit_base_cell_config& cell_params)
{
  add_option(app, "--pci", cell_params.pci, "PCI")->capture_default_str()->check(CLI::Range(0, 1007));
  add_option(app,
             "--sector_id",
             cell_params.sector_id,
             "Sector ID (4-14 bits). This value is concatenated with the gNB Id to form the NR Cell Identity "
             "(NCI). If not specified, a unique value for the DU is automatically derived")
      ->capture_default_str()
      ->check(CLI::Range(0U, (1U << 14) - 1U));
  add_option(app, "--dl_arfcn", cell_params.dl_arfcn, "Downlink ARFCN")->capture_default_str();
  add_auto_enum_option(app, "--band", cell_params.band, "NR band");
  add_option_function<std::string>(
      app,
      "--common_scs",
      [&scs = cell_params.common_scs](const std::string& value) -> std::string {
        scs = to_subcarrier_spacing(value);
        if (scs == subcarrier_spacing::invalid) {
          return fmt::format("Invalid common subcarrier spacing '{}'", value);
        }
        return {};
      },
      "Cell common subcarrier spacing")
      ->capture_default_str();
  add_option(app, "--channel_bandwidth_MHz", cell_params.channel_bw_mhz, "Channel bandwidth in MHz")
      ->capture_default_str()
      ->check([](const std::string& value) -> std::string {
        std::stringstream ss(value);
        unsigned          bw;
        ss >> bw;
        const std::string& error_message = "Error in the channel bandwidth property. Valid values "
                                           "[5,10,15,20,25,30,40,50,60,70,80,90,100]";
        // Bandwidth cannot be less than 5MHz.
        if (bw < 5U) {
          return error_message;
        }

        // Check from [5-25] in steps of 5.
        if (bw < 26U) {
          return ((bw % 5) == 0) ? "" : error_message;
        }

        // Check from [30-100] in steps of 10.
        if (bw < 101U) {
          return ((bw % 10) == 0) ? "" : error_message;
        }

        return error_message;
      });
  add_option(app, "--nof_antennas_ul", cell_params.nof_antennas_ul, "Number of antennas in uplink")
      ->capture_default_str();
  add_option(app, "--nof_antennas_dl", cell_params.nof_antennas_dl, "Number of antennas in downlink")
      ->capture_default_str();
  add_option(app, "--plmn", cell_params.plmn, "PLMN")->capture_default_str();
  add_option(app, "--tac", cell_params.tac, "TAC")->capture_default_str()->check([](const std::string& value) {
    std::stringstream ss(value);
    unsigned          tac;
    ss >> tac;

    // Values 0 and 0xfffffe are reserved.
    if (tac == 0U || tac == 0xfffffeU) {
      return "TAC values 0 or 0xfffffe are reserved";
    }

    return (tac <= 0xffffffU) ? "" : "TAC value out of range";
  });
  add_option(app,
             "--q_rx_lev_min",
             cell_params.q_rx_lev_min,
             "q-RxLevMin, required minimum received RSRP level for cell selection/re-selection, in dBm")
      ->capture_default_str()
      ->check(CLI::Range(-70, -22));
  add_option(app,
             "--q_qual_min",
             cell_params.q_qual_min,
             "q-QualMin, required minimum received RSRQ level for cell selection/re-selection, in dB")
      ->capture_default_str()
      ->check(CLI::Range(-43, -12));
  add_option(app,
             "--pcg_p_nr_fr1",
             cell_params.pcg_cfg.p_nr_fr1,
             "p-nr-fr1, maximum total TX power to be used by the UE in this NR cell group across in FR1")
      ->capture_default_str()
      ->check(CLI::Range(-30, 23));

  // MAC Cell group parameters.
  CLI::App* mcg_subcmd = add_subcommand(app, "mac_cell_group", "MAC Cell Group parameters")->configurable();
  configure_cli11_mac_cell_group_args(*mcg_subcmd, cell_params.mcg_cfg);

  // SSB configuration.
  CLI::App* ssb_subcmd = add_subcommand(app, "ssb", "SSB parameters");
  configure_cli11_ssb_args(*ssb_subcmd, cell_params.ssb_cfg);

  // SIB configuration.
  CLI::App* sib_subcmd = add_subcommand(app, "sib", "SIB configuration parameters");
  configure_cli11_sib_args(*sib_subcmd, cell_params.sib_cfg);

  // UL common configuration.
  CLI::App* ul_common_subcmd = add_subcommand(app, "ul_common", "UL common parameters");
  configure_cli11_ul_common_args(*ul_common_subcmd, cell_params.ul_common_cfg);

  // PDCCH configuration.
  CLI::App* pdcch_subcmd = add_subcommand(app, "pdcch", "PDCCH parameters");
  configure_cli11_pdcch_args(*pdcch_subcmd, cell_params.pdcch_cfg);

  // PDSCH configuration.
  CLI::App* pdsch_subcmd = add_subcommand(app, "pdsch", "PDSCH parameters");
  configure_cli11_pdsch_args(*pdsch_subcmd, cell_params.pdsch_cfg);

  // PUSCH configuration.
  CLI::App* pusch_subcmd = add_subcommand(app, "pusch", "PUSCH parameters");
  configure_cli11_pusch_args(*pusch_subcmd, cell_params.pusch_cfg);

  // PUSCH configuration.
  CLI::App* pucch_subcmd = add_subcommand(app, "pucch", "PUCCH parameters");
  configure_cli11_pucch_args(*pucch_subcmd, cell_params.pucch_cfg);

  // PRACH configuration.
  CLI::App* prach_subcmd = add_subcommand(app, "prach", "PRACH parameters");
  configure_cli11_prach_args(*prach_subcmd, cell_params.prach_cfg);

  // TDD UL DL configuration.
  // NOTE: As the cell TDD pattern can be configured in the common cell and not in the cell, use a different variable to
  // parse the cell TDD property. If the TDD pattern is present for the cell, update the cell TDD pattern value,
  // otherwise do nothing (this will cause that the cell TDD pattern value equals than the common cell TDD pattern).
  // CLI11 needs that the life of the variable last longer than the call of callback function. Therefore, the
  // cell_tdd_pattern variable needs to be static.
  static du_high_unit_tdd_ul_dl_config cell_tdd_pattern;
  CLI::App*                            tdd_ul_dl_subcmd =
      add_subcommand(app, "tdd_ul_dl_cfg", "TDD UL DL configuration parameters")->configurable();
  configure_cli11_tdd_ul_dl_args(*tdd_ul_dl_subcmd, cell_tdd_pattern);
  auto tdd_ul_dl_verify_callback = [&]() {
    CLI::App* tdd_sub_cmd = app.get_subcommand("tdd_ul_dl_cfg");
    if (tdd_sub_cmd->count() != 0) {
      cell_params.tdd_ul_dl_cfg.emplace(cell_tdd_pattern);
    }
    if (!cell_params.tdd_ul_dl_cfg.has_value()) {
      tdd_sub_cmd->disabled();
    }
  };
  tdd_ul_dl_subcmd->parse_complete_callback(tdd_ul_dl_verify_callback);

  // Paging configuration.
  CLI::App* paging_subcmd = add_subcommand(app, "paging", "Paging parameters");
  configure_cli11_paging_args(*paging_subcmd, cell_params.paging_cfg);

  // CSI configuration.
  CLI::App* csi_subcmd = add_subcommand(app, "csi", "CSI-Meas parameters");
  configure_cli11_csi_args(*csi_subcmd, cell_params.csi_cfg);

  // Scheduler expert configuration.
  CLI::App* sched_expert_subcmd = add_subcommand(app, "sched_expert_cfg", "Scheduler expert parameters");
  configure_cli11_scheduler_expert_args(*sched_expert_subcmd, cell_params.sched_expert_cfg);

  // Slicing configuration.
  auto slicing_lambda = [&cell_params](const std::vector<std::string>& values) {
    // Prepare the slices and its configuration.
    cell_params.slice_cfg.resize(values.size());

    // Format every slicing setting.
    for (unsigned i = 0, e = values.size(); i != e; ++i) {
      CLI::App subapp("Slicing parameters", "Slicing config, item #" + std::to_string(i));
      subapp.config_formatter(create_yaml_config_parser());
      subapp.allow_config_extras(CLI::config_extras_mode::capture);
      configure_cli11_slicing_args(subapp, cell_params.slice_cfg[i]);
      std::istringstream ss(values[i]);
      subapp.parse_from_stream(ss);
    }
  };
  add_option_cell(app, "--slicing", slicing_lambda, "Network slicing configuration");
}

static void configure_cli11_cells_args(CLI::App& app, du_high_unit_cell_config& cell_params)
{
  configure_cli11_common_cell_args(app, cell_params.cell);
}

static void configure_cli11_test_ue_mode_args(CLI::App& app, du_high_unit_test_mode_ue_config& test_params)
{
  add_option(app, "--rnti", test_params.rnti, "C-RNTI (0x0 if not configured)")
      ->capture_default_str()
      ->check(CLI::Range(to_value((rnti_t::INVALID_RNTI)), to_value(rnti_t::MAX_CRNTI)));
  add_option(app, "--nof_ues", test_params.nof_ues, "Number of test UE(s) to create.")
      ->capture_default_str()
      ->check(CLI::Range((uint16_t)1, (uint16_t)MAX_NOF_DU_UES));
  add_option(app,
             "--auto_ack_indication_delay",
             test_params.auto_ack_indication_delay,
             "Delay before the UL and DL HARQs are automatically ACKed. This feature should only be used if the UL "
             "PHY is not operational")
      ->capture_default_str();
  add_option(app, "--pdsch_active", test_params.pdsch_active, "PDSCH enabled")->capture_default_str();
  add_option(app, "--pusch_active", test_params.pusch_active, "PUSCH enabled")->capture_default_str();
  add_option(app, "--cqi", test_params.cqi, "Channel Quality Information (CQI) to be forwarded to test UE.")
      ->capture_default_str()
      ->check(CLI::Range(1, 15));
  add_option(app, "--ri", test_params.ri, "Rank Indicator (RI) to be forwarded to test UE.")
      ->capture_default_str()
      ->check(CLI::Range(1, 4));
  add_option(app, "--pmi", test_params.pmi, "Precoder Matrix Indicator (PMI) to be forwarded to test UE.")
      ->capture_default_str()
      ->check(CLI::Range(0, 3));
  add_option(
      app,
      "--i_1_1",
      test_params.i_1_1,
      "Precoder Matrix codebook index \"i_1_1\" to be forwarded to test UE, in the case of more than 2 antennas.")
      ->capture_default_str()
      ->check(CLI::Range(0, 7));
  add_option(
      app,
      "--i_1_3",
      test_params.i_1_3,
      "Precoder Matrix codebook index \"i_1_3\" to be forwarded to test UE, in the case of more than 2 antennas.")
      ->capture_default_str()
      ->check(CLI::Range(0, 1));
  add_option(app,
             "--i_2",
             test_params.i_2,
             "Precoder Matrix codebook index \"i_2\" to be forwarded to test UE, in the case of more than 2 antennas.")
      ->capture_default_str()
      ->check(CLI::Range(0, 3));
}

static void configure_cli11_test_mode_args(CLI::App& app, du_high_unit_test_mode_config& test_params)
{
  CLI::App* test_ue = add_subcommand(app, "test_ue", "automatically created UE for testing purposes");
  configure_cli11_test_ue_mode_args(*test_ue, test_params.test_ue);
}

static void configure_cli11_pcap_args(CLI::App& app, du_high_unit_pcap_config& pcap_params)
{
  add_option(app, "--e2ap_filename", pcap_params.e2ap.filename, "E2AP PCAP file output path")->capture_default_str();
  add_option(app, "--e2ap_enable", pcap_params.e2ap.enabled, "Enable E2AP packet capture")->always_capture_default();
  add_option(app, "--f1ap_filename", pcap_params.f1ap.filename, "F1AP PCAP file output path")->capture_default_str();
  add_option(app, "--f1ap_enable", pcap_params.f1ap.enabled, "Enable F1AP packet capture")->always_capture_default();
  add_option(app, "--f1u_filename", pcap_params.f1u.filename, "F1-U PCAP file output path")->capture_default_str();
  add_option(app, "--f1u_enable", pcap_params.f1u.enabled, "Enable F1-U packet capture")->always_capture_default();
  add_option(app, "--rlc_filename", pcap_params.rlc.filename, "RLC PCAP file output path")->capture_default_str();
  add_option(app, "--rlc_rb_type", pcap_params.rlc.rb_type, "RLC PCAP RB type (all, srb, drb)")->capture_default_str();
  add_option(app, "--rlc_enable", pcap_params.rlc.enabled, "Enable RLC packet capture")->always_capture_default();
  add_option(app, "--mac_filename", pcap_params.mac.filename, "MAC PCAP file output path")->capture_default_str();
  add_option(app, "--mac_type", pcap_params.mac.type, "MAC PCAP pcap type (dlt or udp)")->capture_default_str();
  add_option(app, "--mac_enable", pcap_params.mac.enabled, "Enable MAC packet capture")->always_capture_default();
}

static void configure_cli11_rlc_am_args(CLI::App& app, du_high_unit_rlc_am_config& rlc_am_params)
{
  CLI::App* rlc_tx_am_subcmd = add_subcommand(app, "tx", "AM TX parameters");
  add_option(*rlc_tx_am_subcmd, "--sn", rlc_am_params.tx.sn_field_length, "RLC AM TX SN size")->capture_default_str();
  add_option(*rlc_tx_am_subcmd, "--t-poll-retransmit", rlc_am_params.tx.t_poll_retx, "RLC AM TX t-PollRetransmit (ms)")
      ->capture_default_str();
  add_option(*rlc_tx_am_subcmd, "--max-retx-threshold", rlc_am_params.tx.max_retx_thresh, "RLC AM max retx threshold")
      ->capture_default_str();
  add_option(*rlc_tx_am_subcmd, "--poll-pdu", rlc_am_params.tx.poll_pdu, "RLC AM TX PollPdu")->capture_default_str();
  add_option(*rlc_tx_am_subcmd, "--poll-byte", rlc_am_params.tx.poll_byte, "RLC AM TX PollByte")->capture_default_str();
  add_option(*rlc_tx_am_subcmd,
             "--max_window",
             rlc_am_params.tx.max_window,
             "Non-standard parameter that limits the tx window size. Can be used for limiting memory usage with "
             "large windows. 0 means no limits other than the SN size (i.e. 2^[sn_size-1]).");
  add_option(*rlc_tx_am_subcmd, "--queue-size", rlc_am_params.tx.queue_size, "RLC AM TX SDU queue size")
      ->capture_default_str();
  CLI::App* rlc_rx_am_subcmd = add_subcommand(app, "rx", "AM RX parameters");
  add_option(*rlc_rx_am_subcmd, "--sn", rlc_am_params.rx.sn_field_length, "RLC AM RX SN")->capture_default_str();
  add_option(*rlc_rx_am_subcmd, "--t-reassembly", rlc_am_params.rx.t_reassembly, "RLC AM RX t-Reassembly")
      ->capture_default_str();
  add_option(*rlc_rx_am_subcmd, "--t-status-prohibit", rlc_am_params.rx.t_status_prohibit, "RLC AM RX t-StatusProhibit")
      ->capture_default_str();
  add_option(*rlc_rx_am_subcmd, "--max_sn_per_status", rlc_am_params.rx.max_sn_per_status, "RLC AM RX status SN limit")
      ->capture_default_str();
}

static void configure_cli11_mac_args(CLI::App& app, du_high_unit_mac_lc_config& mac_params)
{
  add_option(app,
             "--lc_priority",
             mac_params.priority,
             "Logical Channel priority. An increasing priority value indicates a lower priority level")
      ->capture_default_str()
      ->check(CLI::Range(4, 16));
  add_option(app, "--lc_group_id", mac_params.lc_group_id, "Logical Channel Group id")
      ->capture_default_str()
      ->check(CLI::Range(1, 7));
  add_option(
      app, "--bucket_size_duration_ms", mac_params.bucket_size_duration_ms, "Bucket size duration in milliseconds")
      ->capture_default_str()
      ->check(CLI::IsMember({5, 10, 20, 50, 100, 150, 300, 500, 1000}));
  add_option(app,
             "--prioritized_bit_rate_kBps",
             mac_params.prioritized_bit_rate_kBps,
             "Prioritized bit rate in kBps. Value 65537 means infinity")
      ->capture_default_str()
      ->check(CLI::IsMember({0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 2768, 65536, 65537}));
}

static void configure_cli11_srb_args(CLI::App& app, du_high_unit_srb_config& srb_params)
{
  add_option(app, "--srb_id", srb_params.srb_id, "SRB Id")->capture_default_str()->check(CLI::IsMember({1, 2}));
  CLI::App* rlc_subcmd = add_subcommand(app, "rlc", "RLC parameters");
  configure_cli11_rlc_am_args(*rlc_subcmd, srb_params.rlc);

  CLI::App* mac_subcmd = add_subcommand(app, "mac", "MAC parameters");
  configure_cli11_mac_args(*mac_subcmd, srb_params.mac);

  // Require RLC configuration.
  app.needs(rlc_subcmd);
  app.needs(mac_subcmd);
}

static void configure_cli11_metrics_args(CLI::App& app, du_high_unit_metrics_config& metrics_params)
{
  add_option(app, "--rlc_report_period", metrics_params.rlc.report_period, "RLC metrics report period")
      ->capture_default_str();
  add_option(app, "--rlc_json_enable", metrics_params.rlc.json_enabled, "Enable RLC JSON metrics reporting")
      ->always_capture_default();

  add_option(app, "--enable_json_metrics", metrics_params.enable_json_metrics, "Enable JSON metrics reporting")
      ->always_capture_default();

  add_option(app,
             "--stdout_metrics_period",
             metrics_params.stdout_metrics_period,
             "DU statistics report period in milliseconds. This metrics sets the console output period.")
      ->capture_default_str();
}

static void configure_cli11_epoch_time(CLI::App& app, epoch_time_t& epoch_time)
{
  add_option(app, "--sfn", epoch_time.sfn, "SFN Part")->capture_default_str()->check(CLI::Range(0, 1023));
  add_option(app, "--subframe_number", epoch_time.subframe_number, "Sub-frame number Part")
      ->capture_default_str()
      ->check(CLI::Range(0, 9));
}

static void configure_cli11_ephemeris_info_ecef(CLI::App& app, ecef_coordinates_t& ephemeris_info)
{
  add_option(app, "--pos_x", ephemeris_info.position_x, "X Position of the satellite")
      ->capture_default_str()
      ->check(CLI::Range(-67108864, 67108863));
  add_option(app, "--pos_y", ephemeris_info.position_y, "Y Position of the satellite")
      ->capture_default_str()
      ->check(CLI::Range(-67108864, 67108863));
  add_option(app, "--pos_z", ephemeris_info.position_z, "Z Position of the satellite")
      ->capture_default_str()
      ->check(CLI::Range(-67108864, 67108863));
  add_option(app, "--vel_x", ephemeris_info.velocity_vx, "X Velocity of the satellite")
      ->capture_default_str()
      ->check(CLI::Range(-32768, 32767));
  add_option(app, "--vel_y", ephemeris_info.velocity_vy, "Y Velocity of the satellite")
      ->capture_default_str()
      ->check(CLI::Range(-32768, 32767));
  add_option(app, "--vel_z", ephemeris_info.velocity_vz, "Z Velocity of the satellite")
      ->capture_default_str()
      ->check(CLI::Range(-32768, 32767));
}

static void configure_cli11_ephemeris_info_orbital(CLI::App& app, orbital_coordinates_t& ephemeris_info)
{
  add_option(app, "--semi_major_axis", ephemeris_info.semi_major_axis, "Semi-major axis of the satellite")
      ->capture_default_str()
      ->check(CLI::Range(0, 1000000000));
  add_option(app, "--eccentricity", ephemeris_info.eccentricity, "Eccentricity of the satellite")
      ->capture_default_str();
  add_option(app, "--periapsis", ephemeris_info.periapsis, "Periapsis of the satellite")->capture_default_str();
  add_option(app, "--longitude", ephemeris_info.longitude, "Longitude of the satellites angle of ascending node")
      ->capture_default_str();
  add_option(app, "--inclination", ephemeris_info.inclination, "Inclination of the satellite")->capture_default_str();
  add_option(app, "--mean_anomaly", ephemeris_info.mean_anomaly, "Mean anomaly of the satellite")
      ->capture_default_str();
}

static void configure_cli11_ntn_args(CLI::App&                  app,
                                     std::optional<ntn_config>& ntn,
                                     epoch_time_t&              epoch_time,
                                     orbital_coordinates_t&     orbital_coordinates,
                                     ecef_coordinates_t&        ecef_coordinates)
{
  ntn_config& config = ntn.emplace();

  add_option(app, "--cell_specific_koffset", config.cell_specific_koffset, "Cell-specific k-offset to be used for NTN.")
      ->capture_default_str()
      ->check(CLI::Range(0, 1023));

  ta_common_t& ta = config.ta_info.emplace();
  add_option(app, "--ta_common", ta.ta_common, "TA common offset");

  // epoch time.
  CLI::App* epoch_time_subcmd = add_subcommand(app, "epoch_time", "Epoch time for the NTN assistance information");
  configure_cli11_epoch_time(*epoch_time_subcmd, epoch_time);

  // ephemeris configuration.
  CLI::App* ephem_subcmd_ecef =
      add_subcommand(app, "ephemeris_info_ecef", "ephermeris information of the satellite in ecef coordinates");
  configure_cli11_ephemeris_info_ecef(*ephem_subcmd_ecef, ecef_coordinates);

  CLI::App* ephem_subcmd_orbital =
      add_subcommand(app, "ephemeris_orbital", "ephermeris information of the satellite in orbital coordinates");
  configure_cli11_ephemeris_info_orbital(*ephem_subcmd_orbital, orbital_coordinates);
}

static epoch_time_t          epoch_time;
static ecef_coordinates_t    ecef_coordinates;
static orbital_coordinates_t orbital_coordinates;

static void configure_cli11_rlc_um_args(CLI::App& app, du_high_unit_rlc_um_config& rlc_um_params)
{
  CLI::App* rlc_tx_um_subcmd = app.add_subcommand("tx", "UM TX parameters");
  rlc_tx_um_subcmd->add_option("--sn", rlc_um_params.tx.sn_field_length, "RLC UM TX SN")->capture_default_str();
  rlc_tx_um_subcmd->add_option("--queue-size", rlc_um_params.tx.queue_size, "RLC UM TX SDU queue size")
      ->capture_default_str();
  CLI::App* rlc_rx_um_subcmd = app.add_subcommand("rx", "UM TX parameters");
  rlc_rx_um_subcmd->add_option("--sn", rlc_um_params.rx.sn_field_length, "RLC UM RX SN")->capture_default_str();
  rlc_rx_um_subcmd->add_option("--t-reassembly", rlc_um_params.rx.t_reassembly, "RLC UM t-Reassembly")
      ->capture_default_str();
}

static void configure_cli11_rlc_args(CLI::App& app, du_high_unit_rlc_config& rlc_params)
{
  add_option(app, "--mode", rlc_params.mode, "RLC mode")->capture_default_str();

  // UM section.
  CLI::App* rlc_um_subcmd = app.add_subcommand("um-bidir", "UM parameters");
  configure_cli11_rlc_um_args(*rlc_um_subcmd, rlc_params.um);

  // AM section.
  CLI::App* rlc_am_subcmd = app.add_subcommand("am", "AM parameters");
  configure_cli11_rlc_am_args(*rlc_am_subcmd, rlc_params.am);
}

static void configure_cli11_f1u_du_args(CLI::App& app, du_high_unit_f1u_du_config& f1u_du_params)
{
  add_option(app, "--backoff_timer", f1u_du_params.t_notify, "F1-U backoff timer (ms)")->capture_default_str();
}

static void configure_cli11_qos_args(CLI::App& app, du_high_unit_qos_config& qos_params)
{
  add_option(app, "--five_qi", qos_params.five_qi, "5QI")->capture_default_str()->check(CLI::Range(0, 255));

  // RLC section.
  CLI::App* rlc_subcmd = app.add_subcommand("rlc", "RLC parameters");
  configure_cli11_rlc_args(*rlc_subcmd, qos_params.rlc);

  // F1 section.
  CLI::App* f1u_du_subcmd = add_subcommand(app, "f1u_du", "F1-U parameters at DU side");
  configure_cli11_f1u_du_args(*f1u_du_subcmd, qos_params.f1u_du);

  // MAC section.
  CLI::App* mac_subcmd = add_subcommand(app, "mac", "MAC parameters");
  configure_cli11_mac_args(*mac_subcmd, qos_params.mac);

  // Mark the application that these subcommands need to be present.
  app.needs(rlc_subcmd);
  app.needs(f1u_du_subcmd);
  app.needs(mac_subcmd);
}

static void configure_cli11_e2_args(CLI::App& app, du_high_unit_e2_config& e2_params)
{
  add_option(app, "--enable_du_e2", e2_params.enable_du_e2, "Enable DU E2 agent")->capture_default_str();
  add_option(app, "--addr", e2_params.ip_addr, "RIC IP address")->capture_default_str();
  add_option(app, "--port", e2_params.port, "RIC port")->check(CLI::Range(20000, 40000))->capture_default_str();
  add_option(app, "--bind_addr", e2_params.bind_addr, "Local IP address to bind for RIC connection")
      ->capture_default_str()
      ->check(CLI::ValidIPV4);
  add_option(app, "--sctp_rto_initial", e2_params.sctp_rto_initial, "SCTP initial RTO value")->capture_default_str();
  add_option(app, "--sctp_rto_min", e2_params.sctp_rto_min, "SCTP RTO min")->capture_default_str();
  add_option(app, "--sctp_rto_max", e2_params.sctp_rto_max, "SCTP RTO max")->capture_default_str();
  add_option(app, "--sctp_init_max_attempts", e2_params.sctp_init_max_attempts, "SCTP init max attempts")
      ->capture_default_str();
  add_option(app, "--sctp_max_init_timeo", e2_params.sctp_max_init_timeo, "SCTP max init timeout")
      ->capture_default_str();
  add_option(app, "--e2sm_kpm_enabled", e2_params.e2sm_kpm_enabled, "Enable KPM service module")->capture_default_str();
  add_option(app, "--e2sm_rc_enabled", e2_params.e2sm_rc_enabled, "Enable RC service module")->capture_default_str();
}

void srsran::configure_cli11_with_du_high_config_schema(CLI::App& app, du_high_parsed_config& parsed_cfg)
{
  add_option(app, "--gnb_id", parsed_cfg.config.gnb_id.id, "gNodeB identifier")->capture_default_str();
  // Adding a default function to display correctly the uint8_t type.
  add_option(app, "--gnb_id_bit_length", parsed_cfg.config.gnb_id.bit_length, "gNodeB identifier length in bits")
      ->default_function([&value = parsed_cfg.config.gnb_id.bit_length]() { return std::to_string(value); })
      ->capture_default_str()
      ->check(CLI::Range(22, 32));
  add_option(app, "--gnb_du_id", parsed_cfg.config.gnb_du_id, "gNB-DU Id")
      ->capture_default_str()
      ->check(CLI::Range(static_cast<uint64_t>(0U), static_cast<uint64_t>(pow(2, 36) - 1)));

  // Loggers section.
  CLI::App* log_subcmd = add_subcommand(app, "log", "Logging configuration")->configurable();
  configure_cli11_log_args(*log_subcmd, parsed_cfg.config.loggers);

  // Metrics section.
  CLI::App* metrics_subcmd = add_subcommand(app, "metrics", "Metrics configuration")->configurable();
  configure_cli11_metrics_args(*metrics_subcmd, parsed_cfg.config.metrics);

  // PCAP section.
  CLI::App* pcap_subcmd = add_subcommand(app, "pcap", "PCAP configuration")->configurable();
  configure_cli11_pcap_args(*pcap_subcmd, parsed_cfg.config.pcaps);

  // Common cell section.
  CLI::App* common_cell_subcmd = add_subcommand(app, "cell_cfg", "Default cell configuration")->configurable();
  configure_cli11_common_cell_args(*common_cell_subcmd, parsed_cfg.common_cell_cfg);
  // Configure the cells to use the common cell parameters once it has been parsed and before parsing the cells.
  common_cell_subcmd->parse_complete_callback([&parsed_cfg, &app]() {
    for (auto& cell : parsed_cfg.config.cells_cfg) {
      cell.cell = parsed_cfg.common_cell_cfg;
    };
    // Run the callback again for the cells if the option callback is already run once.
    if (app.get_option("--cells")->get_callback_run()) {
      app.get_option("--cells")->run_callback();
    }
  });

  // DU section.
  CLI::App* du_subcmd = app.add_subcommand("du", "DU parameters")->configurable();
  configure_cli11_du_args(*du_subcmd, parsed_cfg.config.warn_on_drop);

  // Expert execution section.
  CLI::App* expert_subcmd = add_subcommand(app, "expert_execution", "Expert execution configuration")->configurable();
  configure_cli11_expert_execution_args(*expert_subcmd, parsed_cfg.config.expert_execution_cfg);

  // NTN section.
  CLI::App* ntn_subcmd = add_subcommand(app, "ntn", "NTN parameters")->configurable();
  configure_cli11_ntn_args(*ntn_subcmd, parsed_cfg.config.ntn_cfg, epoch_time, orbital_coordinates, ecef_coordinates);

  // Cell section.
  add_option_cell(
      app,
      "--cells",
      [&parsed_cfg](const std::vector<std::string>& values) {
        // Resize the number of cells that controls the CPU affinites.
        parsed_cfg.config.expert_execution_cfg.cell_affinities.resize(values.size());
        // Prepare the cells from the common cell.
        parsed_cfg.config.cells_cfg.resize(values.size());
        for (auto& cell : parsed_cfg.config.cells_cfg) {
          cell.cell = parsed_cfg.common_cell_cfg;
        }

        // Format every cell.
        for (unsigned i = 0, e = values.size(); i != e; ++i) {
          CLI::App subapp("Cells section", "Cells section config, item #" + std::to_string(i));
          subapp.config_formatter(create_yaml_config_parser());
          subapp.allow_config_extras(CLI::config_extras_mode::capture);
          configure_cli11_cells_args(subapp, parsed_cfg.config.cells_cfg[i]);
          std::istringstream ss(values[i]);
          subapp.parse_from_stream(ss);
        }
      },
      "Sets the cell configuration on a per cell basis, overwriting the default configuration defined by cell_cfg");

  // QoS section.
  auto qos_lambda = [&parsed_cfg](const std::vector<std::string>& values) {
    // Prepare the radio bearers
    parsed_cfg.config.qos_cfg.resize(values.size());

    // Format every QoS setting.
    for (unsigned i = 0, e = values.size(); i != e; ++i) {
      CLI::App subapp("QoS parameters", "QoS config, item #" + std::to_string(i));
      subapp.config_formatter(create_yaml_config_parser());
      subapp.allow_config_extras(CLI::config_extras_mode::capture);
      configure_cli11_qos_args(subapp, parsed_cfg.config.qos_cfg[i]);
      std::istringstream ss(values[i]);
      subapp.parse_from_stream(ss);
    }
  };

  add_option_cell(app, "--qos", qos_lambda, "Configures RLC and PDCP radio bearers on a per 5QI basis.");

  // SRB section.
  auto srb_lambda = [&parsed_cfg](const std::vector<std::string>& values) {
    // Format every SRB setting.
    for (unsigned i = 0, e = values.size(); i != e; ++i) {
      CLI::App subapp("SRB parameters", "SRB config, item #" + std::to_string(i));
      subapp.config_formatter(create_yaml_config_parser());
      subapp.allow_config_extras(CLI::config_extras_mode::capture);
      du_high_unit_srb_config srb_cfg;
      configure_cli11_srb_args(subapp, srb_cfg);
      std::istringstream ss(values[i]);
      subapp.parse_from_stream(ss);
      parsed_cfg.config.srb_cfg[static_cast<srb_id_t>(srb_cfg.srb_id)] = srb_cfg;
    }
  };
  app.add_option_function<std::vector<std::string>>("--srbs", srb_lambda, "Configures signaling radio bearers.");

  // Test mode section.
  CLI::App* test_mode_subcmd = add_subcommand(app, "test_mode", "Test mode configuration")->configurable();
  configure_cli11_test_mode_args(*test_mode_subcmd, parsed_cfg.config.test_mode_cfg);

  // E2 section.
  CLI::App* e2_subcmd = add_subcommand(app, "e2", "E2 parameters")->configurable();
  configure_cli11_e2_args(*e2_subcmd, parsed_cfg.config.e2_cfg);
}

static void manage_ntn_optional(CLI::App& app, du_high_unit_config& gnb_cfg)
{
  auto     ntn_app             = app.get_subcommand_ptr("ntn");
  unsigned nof_epoch_entries   = ntn_app->get_subcommand("epoch_time")->count_all();
  unsigned nof_ecef_entries    = ntn_app->get_subcommand("ephemeris_info_ecef")->count_all();
  unsigned nof_orbital_entries = ntn_app->get_subcommand("ephemeris_orbital")->count_all();

  if (nof_epoch_entries) {
    gnb_cfg.ntn_cfg.value().epoch_time = epoch_time;
  }

  if (nof_ecef_entries) {
    gnb_cfg.ntn_cfg.value().ephemeris_info = ecef_coordinates;
  } else if (nof_orbital_entries) {
    gnb_cfg.ntn_cfg.value().ephemeris_info = orbital_coordinates;
  }
  if (app.get_subcommand("ntn")->count_all() == 0) {
    gnb_cfg.ntn_cfg.reset();
    // As NTN configuration is optional, disable the command when it is not present in the configuration.
    ntn_app->disabled();
  }
}

// Derive the parameters set to "auto"-derived for a cell.
static void derive_cell_auto_params(du_high_unit_base_cell_config& cell_cfg)
{
  // If NR band is not set, derive a valid one from the DL-ARFCN.
  if (not cell_cfg.band.has_value()) {
    cell_cfg.band = band_helper::get_band_from_dl_arfcn(cell_cfg.dl_arfcn);
  }

  // If in TDD mode, and pattern was not set, generate a pattern DDDDDDXUUU.
  const duplex_mode dplx_mode = band_helper::get_duplex_mode(cell_cfg.band.value());
  if (dplx_mode == duplex_mode::TDD and not cell_cfg.tdd_ul_dl_cfg.has_value()) {
    cell_cfg.tdd_ul_dl_cfg.emplace();
    cell_cfg.tdd_ul_dl_cfg->pattern1.dl_ul_period_slots = 10;
    cell_cfg.tdd_ul_dl_cfg->pattern1.nof_dl_slots       = 6;
    cell_cfg.tdd_ul_dl_cfg->pattern1.nof_dl_symbols     = 8;
    cell_cfg.tdd_ul_dl_cfg->pattern1.nof_ul_slots       = 3;
    cell_cfg.tdd_ul_dl_cfg->pattern1.nof_ul_symbols     = 0;
  }

  // If PRACH configuration Index not set, a default one is assigned.
  if (not cell_cfg.prach_cfg.prach_config_index.has_value()) {
    if (band_helper::get_duplex_mode(cell_cfg.band.value()) == duplex_mode::FDD) {
      cell_cfg.prach_cfg.prach_config_index = 16;
    } else {
      // Valid for TDD period of 5 ms. And, PRACH index 159 is well tested.
      cell_cfg.prach_cfg.prach_config_index = 159;
    }
  }
}

static void derive_auto_params(du_high_unit_config& config)
{
  unsigned next_sector_id = 0;
  for (auto& cell : config.cells_cfg) {
    if (not cell.cell.sector_id.has_value()) {
      // Auto-derive sector ID if not defined.
      cell.cell.sector_id = next_sector_id;
      next_sector_id++;
    } else {
      // If sector ID defined for this cell, recompute the next sector ID.
      next_sector_id = std::max(next_sector_id, cell.cell.sector_id.value() + 1);
    }

    derive_cell_auto_params(cell.cell);
  }
}

void srsran::autoderive_du_high_parameters_after_parsing(CLI::App& app, du_high_unit_config& unit_cfg)
{
  manage_ntn_optional(app, unit_cfg);
  derive_auto_params(unit_cfg);
}
