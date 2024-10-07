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

#include "srsran/phy/upper/signal_processors/signal_processor_factories.h"
#include "dmrs_pbch_processor_impl.h"
#include "dmrs_pdcch_processor_impl.h"
#include "dmrs_pdsch_processor_impl.h"
#include "dmrs_pusch_estimator_impl.h"
#include "nzp_csi_rs_generator_impl.h"
#include "nzp_csi_rs_generator_pool.h"
#include "port_channel_estimator_average_impl.h"
#include "pss_processor_impl.h"
#include "pucch/dmrs_pucch_processor_format1_impl.h"
#include "pucch/dmrs_pucch_processor_format2_impl.h"
#include "sss_processor_impl.h"
#include "srsran/phy/support/support_factories.h"
#include "srsran/phy/support/support_formatters.h"
#include "srsran/phy/support/time_alignment_estimator/time_alignment_estimator_factories.h"
#include "srsran/phy/upper/signal_processors/signal_processor_formatters.h"

using namespace srsran;

namespace {

class dmrs_pbch_processor_sw_factory : public dmrs_pbch_processor_factory
{
private:
  std::shared_ptr<pseudo_random_generator_factory> prg_factory;

public:
  explicit dmrs_pbch_processor_sw_factory(std::shared_ptr<pseudo_random_generator_factory> prg_factory_) :
    prg_factory(std::move(prg_factory_))
  {
    srsran_assert(prg_factory, "Invalid PRG factory.");
  }

  std::unique_ptr<dmrs_pbch_processor> create() override
  {
    return std::make_unique<dmrs_pbch_processor_impl>(prg_factory->create());
  }
};

class dmrs_pdcch_processor_sw_factory : public dmrs_pdcch_processor_factory
{
private:
  std::shared_ptr<pseudo_random_generator_factory> prg_factory;

public:
  explicit dmrs_pdcch_processor_sw_factory(std::shared_ptr<pseudo_random_generator_factory> prg_factory_) :
    prg_factory(std::move(prg_factory_))
  {
    srsran_assert(prg_factory, "Invalid PRG factory.");
  }

  std::unique_ptr<dmrs_pdcch_processor> create() override
  {
    return std::make_unique<dmrs_pdcch_processor_impl>(prg_factory->create());
  }
};

class dmrs_pdsch_processor_sw_factory : public dmrs_pdsch_processor_factory
{
private:
  std::shared_ptr<pseudo_random_generator_factory> prg_factory;

public:
  explicit dmrs_pdsch_processor_sw_factory(std::shared_ptr<pseudo_random_generator_factory> prg_factory_) :
    prg_factory(std::move(prg_factory_))
  {
    srsran_assert(prg_factory, "Invalid PRG factory.");
  }

  std::unique_ptr<dmrs_pdsch_processor> create() override
  {
    return std::make_unique<dmrs_pdsch_processor_impl>(prg_factory->create());
  }
};

class dmrs_pucch_estimator_sw_factory : public dmrs_pucch_estimator_factory
{
public:
  dmrs_pucch_estimator_sw_factory(std::shared_ptr<pseudo_random_generator_factory>&      prg_factory_,
                                  std::shared_ptr<low_papr_sequence_collection_factory>& lpc_factory_,
                                  std::shared_ptr<port_channel_estimator_factory>&       ch_estimator_factory_) :
    prg_factory(std::move(prg_factory_)),
    lpc_factory(std::move(lpc_factory_)),
    ch_estimator_factory(std::move(ch_estimator_factory_))
  {
    srsran_assert(prg_factory, "Invalid sequence generator factory.");
    srsran_assert(lpc_factory, "Invalid sequence collection factory.");
    srsran_assert(ch_estimator_factory, "Invalid channel estimator factory.");
  }

  std::unique_ptr<dmrs_pucch_processor> create_format1() override
  {
    // Prepare DM-RS for PUCCH Format 1 low PAPR sequence parameters.
    unsigned               m     = 1;
    unsigned               delta = 0;
    std::array<float, NRE> alphas;
    std::generate(alphas.begin(), alphas.end(), [&, n = 0]() mutable {
      return TWOPI * static_cast<float>(n++) / static_cast<float>(NRE);
    });

    return std::make_unique<dmrs_pucch_processor_format1_impl>(
        prg_factory->create(),
        lpc_factory->create(m, delta, alphas),
        ch_estimator_factory->create(port_channel_estimator_fd_smoothing_strategy::mean, /*compensate_cfo =*/false));
  }

  std::unique_ptr<dmrs_pucch_processor> create_format2() override
  {
    return std::make_unique<dmrs_pucch_processor_format2_impl>(
        prg_factory->create(), ch_estimator_factory->create(port_channel_estimator_fd_smoothing_strategy::filter));
  }

private:
  std::shared_ptr<pseudo_random_generator_factory>      prg_factory;
  std::shared_ptr<low_papr_sequence_collection_factory> lpc_factory;
  std::shared_ptr<port_channel_estimator_factory>       ch_estimator_factory;
};

class dmrs_pusch_estimator_factory_sw : public dmrs_pusch_estimator_factory
{
public:
  dmrs_pusch_estimator_factory_sw(std::shared_ptr<pseudo_random_generator_factory> prg_factory_,
                                  std::shared_ptr<port_channel_estimator_factory>  ch_estimator_factory_) :
    prg_factory(std::move(prg_factory_)), ch_estimator_factory(std::move(ch_estimator_factory_))
  {
    srsran_assert(prg_factory, "Invalid PRG factory.");
    srsran_assert(ch_estimator_factory, "Invalid channel estimator factory.");
  }

  std::unique_ptr<dmrs_pusch_estimator> create() override
  {
    return std::make_unique<dmrs_pusch_estimator_impl>(
        prg_factory->create(), ch_estimator_factory->create(port_channel_estimator_fd_smoothing_strategy::filter));
  }

private:
  std::shared_ptr<pseudo_random_generator_factory> prg_factory;
  std::shared_ptr<port_channel_estimator_factory>  ch_estimator_factory;
};

class nzp_csi_rs_generator_factory_sw : public nzp_csi_rs_generator_factory
{
public:
  nzp_csi_rs_generator_factory_sw(std::shared_ptr<pseudo_random_generator_factory> prg_factory_) :
    prg_factory(std::move(prg_factory_))
  {
    srsran_assert(prg_factory, "Invalid PRG factory.");
  }
  std::unique_ptr<nzp_csi_rs_generator> create() override
  {
    return std::make_unique<nzp_csi_rs_generator_impl>(prg_factory->create());
  }
  std::unique_ptr<nzp_csi_rs_configuration_validator> create_validator() override
  {
    return std::make_unique<nzp_csi_rs_configuration_validator_impl>();
  }

private:
  std::shared_ptr<pseudo_random_generator_factory> prg_factory;
};

class nzp_csi_rs_generator_pool_factory : public nzp_csi_rs_generator_factory
{
public:
  nzp_csi_rs_generator_pool_factory(std::shared_ptr<nzp_csi_rs_generator_factory> factory_,
                                    unsigned                                      nof_concurrent_threads_) :
    factory(std::move(factory_)), nof_concurrent_threads(nof_concurrent_threads_)
  {
    srsran_assert(factory, "Invalid NZP-CSI-RS generator factory.");
    srsran_assert(nof_concurrent_threads > 1, "Number of concurrent threads must be greater than one.");
  }

  std::unique_ptr<nzp_csi_rs_generator> create() override
  {
    if (!generators) {
      std::vector<std::unique_ptr<nzp_csi_rs_generator>> instances(nof_concurrent_threads);

      for (auto& processor : instances) {
        processor = factory->create();
      }

      generators = std::make_shared<nzp_csi_rs_generator_pool::generator_pool_type>(std::move(instances));
    }

    return std::make_unique<nzp_csi_rs_generator_pool>(generators);
  }

  std::unique_ptr<nzp_csi_rs_generator> create(srslog::basic_logger& logger) override
  {
    if (!generators) {
      std::vector<std::unique_ptr<nzp_csi_rs_generator>> instances(nof_concurrent_threads);

      for (auto& processor : instances) {
        processor = factory->create(logger);
      }

      generators = std::make_shared<nzp_csi_rs_generator_pool::generator_pool_type>(std::move(instances));
    }

    return std::make_unique<nzp_csi_rs_generator_pool>(generators);
  }

  std::unique_ptr<nzp_csi_rs_configuration_validator> create_validator() override
  {
    return factory->create_validator();
  }

private:
  std::shared_ptr<nzp_csi_rs_generator_factory>                   factory;
  unsigned                                                        nof_concurrent_threads;
  std::shared_ptr<nzp_csi_rs_generator_pool::generator_pool_type> generators;
};

class port_channel_estimator_factory_sw : public port_channel_estimator_factory
{
public:
  explicit port_channel_estimator_factory_sw(std::shared_ptr<time_alignment_estimator_factory> ta_estimator_factory_) :
    ta_estimator_factory(std::move(ta_estimator_factory_))
  {
    srsran_assert(ta_estimator_factory, "Invalid TA estimator factory.");
  }

  std::unique_ptr<port_channel_estimator> create(port_channel_estimator_fd_smoothing_strategy fd_smoothing_strategy,
                                                 bool                                         compensate_cfo) override
  {
    std::unique_ptr<interpolator> interp = create_interpolator();

    return std::make_unique<port_channel_estimator_average_impl>(
        std::move(interp), ta_estimator_factory->create(), fd_smoothing_strategy, compensate_cfo);
  }

private:
  std::shared_ptr<time_alignment_estimator_factory> ta_estimator_factory;
};

class pss_processor_factory_sw : public pss_processor_factory
{
public:
  std::unique_ptr<pss_processor> create() override { return std::make_unique<pss_processor_impl>(); }
};

class sss_processor_factory_sw : public sss_processor_factory
{
public:
  std::unique_ptr<sss_processor> create() override { return std::make_unique<sss_processor_impl>(); }
};

} // namespace

std::shared_ptr<dmrs_pbch_processor_factory>
srsran::create_dmrs_pbch_processor_factory_sw(std::shared_ptr<pseudo_random_generator_factory> prg_factory)
{
  return std::make_shared<dmrs_pbch_processor_sw_factory>(std::move(prg_factory));
}

std::shared_ptr<dmrs_pdcch_processor_factory>
srsran::create_dmrs_pdcch_processor_factory_sw(std::shared_ptr<pseudo_random_generator_factory> prg_factory)
{
  return std::make_shared<dmrs_pdcch_processor_sw_factory>(std::move(prg_factory));
}

std::shared_ptr<dmrs_pdsch_processor_factory>
srsran::create_dmrs_pdsch_processor_factory_sw(std::shared_ptr<pseudo_random_generator_factory> prg_factory)
{
  return std::make_shared<dmrs_pdsch_processor_sw_factory>(std::move(prg_factory));
}

std::shared_ptr<dmrs_pucch_estimator_factory>
srsran::create_dmrs_pucch_estimator_factory_sw(std::shared_ptr<pseudo_random_generator_factory>      prg_factory,
                                               std::shared_ptr<low_papr_sequence_collection_factory> lpc_factory,
                                               std::shared_ptr<port_channel_estimator_factory> ch_estimator_factory)
{
  return std::make_shared<dmrs_pucch_estimator_sw_factory>(prg_factory, lpc_factory, ch_estimator_factory);
}

std::shared_ptr<dmrs_pusch_estimator_factory>
srsran::create_dmrs_pusch_estimator_factory_sw(std::shared_ptr<pseudo_random_generator_factory> prg_factory,
                                               std::shared_ptr<port_channel_estimator_factory>  ch_estimator_factory)
{
  return std::make_shared<dmrs_pusch_estimator_factory_sw>(std::move(prg_factory), std::move(ch_estimator_factory));
}

std::shared_ptr<nzp_csi_rs_generator_factory>
srsran::create_nzp_csi_rs_generator_factory_sw(std::shared_ptr<pseudo_random_generator_factory> prg_factory)
{
  return std::make_shared<nzp_csi_rs_generator_factory_sw>(std::move(prg_factory));
}

std::shared_ptr<nzp_csi_rs_generator_factory>
srsran::create_nzp_csi_rs_generator_pool_factory(std::shared_ptr<nzp_csi_rs_generator_factory> generator_factory,
                                                 unsigned                                      nof_concurrent_threads)
{
  return std::make_shared<nzp_csi_rs_generator_pool_factory>(std::move(generator_factory), nof_concurrent_threads);
}

std::shared_ptr<port_channel_estimator_factory>
srsran::create_port_channel_estimator_factory_sw(std::shared_ptr<time_alignment_estimator_factory> ta_estimator_factory)
{
  return std::make_shared<port_channel_estimator_factory_sw>(std::move(ta_estimator_factory));
}

std::shared_ptr<pss_processor_factory> srsran::create_pss_processor_factory_sw()
{
  return std::make_shared<pss_processor_factory_sw>();
}

std::shared_ptr<sss_processor_factory> srsran::create_sss_processor_factory_sw()
{
  return std::make_shared<sss_processor_factory_sw>();
}

template <typename Func>
static std::chrono::nanoseconds time_execution(Func&& func)
{
  auto start = std::chrono::steady_clock::now();
  func();
  auto end = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

namespace {

class logging_nzp_csi_rs_generator_decorator : public nzp_csi_rs_generator
{
public:
  logging_nzp_csi_rs_generator_decorator(srslog::basic_logger&                 logger_,
                                         std::unique_ptr<nzp_csi_rs_generator> generator_) :
    logger(logger_), generator(std::move(generator_))
  {
    srsran_assert(generator, "Invalid NZP CSI-RS generator.");
  }

  void map(resource_grid_mapper& mapper, const config_t& config) override
  {
    const auto&& func = [&]() { generator->map(mapper, config); };

    std::chrono::nanoseconds time_ns = time_execution(func);

    if (logger.debug.enabled()) {
      // Detailed log information, including a list of all config fields.
      logger.debug(config.slot.sfn(), config.slot.slot_index(), "NZP-CSI: {:s} {}\n  {:n}", config, time_ns, config);
      return;
    }
    // Single line log entry.
    logger.info(config.slot.sfn(), config.slot.slot_index(), "NZP-CSI: {:s} {}", config, time_ns);
  }

private:
  srslog::basic_logger&                 logger;
  std::unique_ptr<nzp_csi_rs_generator> generator;
};

} // namespace

std::unique_ptr<nzp_csi_rs_generator> nzp_csi_rs_generator_factory::create(srslog::basic_logger& logger)
{
  return std::make_unique<logging_nzp_csi_rs_generator_decorator>(logger, create());
}
