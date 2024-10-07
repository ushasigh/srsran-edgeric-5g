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

#include "lib/pdcp/pdcp_entity_tx.h"
#include "srsran/support/benchmark_utils.h"
#include "srsran/support/executors/manual_task_worker.h"
#include <getopt.h>

using namespace srsran;

const std::array<uint8_t, 16> k_128_int =
    {0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x30, 0x31};
const std::array<uint8_t, 16> k_128_enc =
    {0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x30, 0x31};

/// Mocking class of the surrounding layers invoked by the PDCP.
class pdcp_tx_gen_frame : public pdcp_tx_lower_notifier, public pdcp_tx_upper_control_notifier
{
public:
  /// PDCP TX upper layer control notifier
  void on_max_count_reached() final {}
  void on_protocol_failure() final {}

  /// PDCP TX lower layer data notifier
  void on_new_pdu(byte_buffer pdu, bool is_retx) final {}
  void on_discard_pdu(uint32_t pdcp_sn) final {}
};

struct bench_params {
  unsigned nof_repetitions   = 1000;
  bool     print_timing_info = false;
};

struct app_params {
  int                  algo         = -1;
  srslog::basic_levels log_level    = srslog::basic_levels::error;
  std::string          log_filename = "stdout";
};

static void usage(const char* prog, const bench_params& params, const app_params& app)
{
  fmt::print("Usage: {} [-R repetitions] [-t timing information]\n", prog);
  fmt::print("\t-a Security algorithm to use [Default {}, valid {{-1,0,1,2,3}}]\n", app.algo);
  fmt::print("\t-t Print timing information [Default {}]\n", params.print_timing_info);
  fmt::print("\t-R Repetitions [Default {}]\n", params.nof_repetitions);
  fmt::print("\t-l Log level to use [Default {}, valid {{error, warning, info, debug}}]\n", app.log_level);
  fmt::print("\t-f Log filename to use [Default {}]\n", app.log_filename);
  fmt::print("\t-h Show this message\n");
}

static void parse_args(int argc, char** argv, bench_params& params, app_params& app)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "a:R:l:f:th")) != -1) {
    switch (opt) {
      case 'R':
        params.nof_repetitions = std::strtol(optarg, nullptr, 10);
        break;
      case 'a':
        app.algo = std::strtol(optarg, nullptr, 10);
        break;
      case 't':
        params.print_timing_info = true;
        break;
      case 'l': {
        auto value    = srslog::str_to_basic_level(std::string(optarg));
        app.log_level = value.has_value() ? value.value() : srslog::basic_levels::none;
        break;
      }
      case 'f':
        app.log_filename = std::string(optarg);
        break;
      case 'h':
      default:
        usage(argv[0], params, app);
        exit(0);
    }
  }
}

void benchmark_pdcp_tx(bench_params                  params,
                       security::integrity_enabled   int_enabled,
                       security::ciphering_enabled   ciph_enabled,
                       security::integrity_algorithm int_algo,
                       security::ciphering_algorithm ciph_algo)
{
  fmt::memory_buffer buffer;
  fmt::format_to(buffer,
                 "Benchmark PDCP TX. int={} ciph={} int_algo={} ciph_algo={}",
                 int_enabled,
                 ciph_enabled,
                 int_algo,
                 ciph_algo);
  std::unique_ptr<benchmarker> bm = std::make_unique<benchmarker>(to_c_str(buffer), params.nof_repetitions);

  timer_manager      timers;
  manual_task_worker worker{64};

  // Set TX config
  pdcp_tx_config config         = {};
  config.rb_type                = pdcp_rb_type::drb;
  config.rlc_mode               = pdcp_rlc_mode::um;
  config.sn_size                = pdcp_sn_size::size18bits;
  config.direction              = pdcp_security_direction::downlink;
  config.discard_timer          = pdcp_discard_timer::infinity;
  config.status_report_required = false;
  config.custom.rlc_sdu_queue   = params.nof_repetitions;

  security::sec_128_as_config sec_cfg = {};

  // Set security domain
  sec_cfg.domain = security::sec_domain::up; // DRB

  // Set security keys
  sec_cfg.k_128_int = k_128_int;
  sec_cfg.k_128_enc = k_128_enc;

  // Set encription/integrity algorithms
  sec_cfg.integ_algo  = int_algo;
  sec_cfg.cipher_algo = ciph_algo;

  // Create test frame
  pdcp_tx_gen_frame frame = {};

  // Create PDCP entities
  std::unique_ptr<pdcp_entity_tx> pdcp_tx = std::make_unique<pdcp_entity_tx>(
      0, drb_id_t::drb1, config, frame, frame, timer_factory{timers, worker}, worker, worker);
  pdcp_tx->configure_security(sec_cfg, int_enabled, ciph_enabled);

  // Prepare SDU list for benchmark
  std::vector<byte_buffer> sdu_list  = {};
  int                      num_sdus  = params.nof_repetitions;
  int                      num_bytes = 1500;
  for (int i = 0; i < num_sdus; i++) {
    byte_buffer sdu_buf = {};
    for (int j = 0; j < num_bytes; ++j) {
      report_fatal_error_if_not(sdu_buf.append(rand()), "Failed to allocate SDU buffer");
    }
    sdu_list.push_back(std::move(sdu_buf));
  }

  // Run benchmark.
  int  pdcp_sn = 0;
  auto measure = [&pdcp_tx, &sdu_list, pdcp_sn]() mutable {
    pdcp_tx->handle_sdu(sdu_list[pdcp_sn].copy());
    pdcp_sn++;
  };
  bm->new_measure("PDCP TX", 1500 * 8, measure);

  // Output results.
  if (params.print_timing_info) {
    bm->print_percentiles_time();
  }

  // Output results.
  bm->print_percentiles_throughput(" bps");
}

int run_benchmark(bench_params params, int algo)
{
  if (algo != 0 && algo != 1 && algo != 2 && algo != 3) {
    fmt::print("Unsupported algortithm. Use NIA/NEA 0, 1, 2 or 3.\n");
    return -1;
  }
  fmt::print("------ Benchmarcking: NIA{} NEA{} ------\n", algo, algo);

  auto int_algo  = static_cast<security::integrity_algorithm>(algo);
  auto ciph_algo = static_cast<security::ciphering_algorithm>(algo);

  if (algo == 0) {
    benchmark_pdcp_tx(params,
                      security::integrity_enabled::off,
                      security::ciphering_enabled::off,
                      security::integrity_algorithm::nia2, // NIA0 is forbiden, use NIA2 and disable integrity
                      ciph_algo);
  } else {
    benchmark_pdcp_tx(params, security::integrity_enabled::on, security::ciphering_enabled::on, int_algo, ciph_algo);
    benchmark_pdcp_tx(params, security::integrity_enabled::on, security::ciphering_enabled::off, int_algo, ciph_algo);
    benchmark_pdcp_tx(params, security::integrity_enabled::off, security::ciphering_enabled::on, int_algo, ciph_algo);
  }
  fmt::print("------ End of Benchmarck ------\n");
  return 0;
}

int main(int argc, char** argv)
{
  bench_params params{};
  app_params   app_params{};
  parse_args(argc, argv, params, app_params);

  srslog::init();

  srslog::sink* log_sink = (app_params.log_filename == "stdout") ? srslog::create_stdout_sink()
                                                                 : srslog::create_file_sink(app_params.log_filename);
  if (log_sink == nullptr) {
    return -1;
  }
  srslog::set_default_sink(*log_sink);
  srslog::fetch_basic_logger("PDCP").set_level(app_params.log_level);

  if (app_params.algo != -1) {
    run_benchmark(params, app_params.algo);
  } else {
    for (unsigned i = 0; i < 4; i++) {
      run_benchmark(params, i);
    }
  }
  srslog::flush();
}
