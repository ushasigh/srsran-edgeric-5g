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

#include "../../../../lib/ofh/receiver/ofh_uplane_rx_symbol_data_flow_notifier.h"
#include "../../phy/support/resource_grid_test_doubles.h"
#include "../ofh_uplane_rx_symbol_notifier_test_doubles.h"
#include <gtest/gtest.h>

using namespace srsran;
using namespace ofh;
using namespace ofh::testing;

static constexpr ofdm_symbol_range symbol_range = {0, 14};

TEST(ofh_data_flow_uplane_rx_symbol_notifier, empty_context_does_not_notify)
{
  auto                                notifier = std::make_shared<uplane_rx_symbol_notifier_spy>();
  auto                                repo     = std::make_shared<uplink_context_repository>(1);
  uplane_rx_symbol_data_flow_notifier sender(srslog::fetch_basic_logger("TEST"), repo, notifier);
  slot_point                          slot(0, 0, 1);
  unsigned                            symbol = 0;

  sender.notify_received_symbol(slot, symbol);

  ASSERT_TRUE(repo->get(slot, symbol).empty());
  ASSERT_FALSE(notifier->has_new_uplink_symbol_function_been_called());
  ASSERT_FALSE(notifier->has_new_prach_function_been_called());
}

TEST(ofh_data_flow_uplane_rx_symbol_notifier, unwritten_grid_does_not_notify)
{
  auto                                notifier = std::make_shared<uplane_rx_symbol_notifier_spy>();
  auto                                repo     = std::make_shared<uplink_context_repository>(1);
  uplane_rx_symbol_data_flow_notifier sender(srslog::fetch_basic_logger("TEST"), repo, notifier);
  slot_point                          slot(0, 0, 1);
  unsigned                            symbol = 0;
  unsigned                            sector = 0;
  resource_grid_spy                   grid(2, 14, 273);

  repo->add({slot, sector}, grid, symbol_range);
  sender.notify_received_symbol(slot, symbol);

  ASSERT_FALSE(repo->get(slot, symbol).empty());
  ASSERT_FALSE(notifier->has_new_uplink_symbol_function_been_called());
  ASSERT_FALSE(notifier->has_new_prach_function_been_called());
}

TEST(ofh_data_flow_uplane_rx_symbol_notifier, completed_resource_grid_triggers_notification)
{
  auto                                notifier = std::make_shared<uplane_rx_symbol_notifier_spy>();
  auto                                repo     = std::make_shared<uplink_context_repository>(1);
  uplane_rx_symbol_data_flow_notifier sender(srslog::fetch_basic_logger("TEST"), repo, notifier);
  slot_point                          slot(0, 0, 1);
  unsigned                            symbol = 0;
  unsigned                            sector = 0;
  resource_grid_spy                   grid(2, 14, 273);

  static_vector<cbf16_t, MAX_NOF_PRBS * NOF_SUBCARRIERS_PER_RB> samples(grid.get_writer().get_nof_subc());
  repo->add({slot, sector}, grid, symbol_range);
  ASSERT_FALSE(repo->get(slot, symbol).empty());

  // Fill the grid.
  for (unsigned port = 0, port_end = grid.get_writer().get_nof_ports(); port != port_end; ++port) {
    repo->write_grid(slot, port, symbol, 0, samples);
  }

  sender.notify_received_symbol(slot, symbol);

  // Assert that the context is cleared when it is notified.
  ASSERT_TRUE(repo->get(slot, symbol).empty());

  // Assert the notification.
  ASSERT_TRUE(notifier->has_new_uplink_symbol_function_been_called());
  ASSERT_FALSE(notifier->has_new_prach_function_been_called());
}

TEST(ofh_data_flow_uplane_rx_symbol_notifier, uncompleted_port_does_not_notify)
{
  auto                                notifier = std::make_shared<uplane_rx_symbol_notifier_spy>();
  auto                                repo     = std::make_shared<uplink_context_repository>(1);
  uplane_rx_symbol_data_flow_notifier sender(srslog::fetch_basic_logger("TEST"), repo, notifier);
  slot_point                          slot(0, 0, 1);
  unsigned                            symbol = 0;
  unsigned                            sector = 0;
  resource_grid_spy                   grid(2, 14, 273);

  static_vector<cbf16_t, MAX_NOF_PRBS * NOF_SUBCARRIERS_PER_RB> samples(grid.get_writer().get_nof_subc());
  repo->add({slot, sector}, grid, symbol_range);

  // Fill the grid.
  repo->write_grid(slot, 0, symbol, 0, samples);

  sender.notify_received_symbol(slot, symbol);

  ASSERT_FALSE(repo->get(slot, symbol).empty());
  ASSERT_FALSE(notifier->has_new_uplink_symbol_function_been_called());
  ASSERT_FALSE(notifier->has_new_prach_function_been_called());
}

TEST(ofh_data_flow_uplane_rx_symbol_notifier, uncompleted_prbs_does_not_notify)
{
  auto                                notifier = std::make_shared<uplane_rx_symbol_notifier_spy>();
  auto                                repo     = std::make_shared<uplink_context_repository>(1);
  uplane_rx_symbol_data_flow_notifier sender(srslog::fetch_basic_logger("TEST"), repo, notifier);
  slot_point                          slot(0, 0, 1);
  unsigned                            symbol = 0;
  unsigned                            sector = 0;
  resource_grid_spy                   grid(1, 14, 273);

  static_vector<cbf16_t, MAX_NOF_PRBS * NOF_SUBCARRIERS_PER_RB> samples(grid.get_writer().get_nof_subc() - 1);
  repo->add({slot, sector}, grid, symbol_range);

  // Fill the grid.
  repo->write_grid(slot, 0, symbol, 0, samples);

  sender.notify_received_symbol(slot, symbol);

  ASSERT_FALSE(repo->get(slot, symbol).empty());
  ASSERT_FALSE(notifier->has_new_uplink_symbol_function_been_called());
  ASSERT_FALSE(notifier->has_new_prach_function_been_called());
}
