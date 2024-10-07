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

#include "ethernet_rx_buffer_pool.h"
#include "srsran/ofh/ethernet/ethernet_receiver.h"
#include "srsran/srslog/logger.h"

namespace srsran {

class task_executor;

namespace ether {

/// Implementation for the Ethernet receiver.
class receiver_impl : public receiver
{
  static constexpr unsigned BUFFER_SIZE = 9600;

  enum class receiver_status { idle, running, stop_requested, stopped };

public:
  receiver_impl(const std::string&    interface,
                bool                  is_promiscuous_mode_enabled,
                task_executor&        executor_,
                srslog::basic_logger& logger_);
  ~receiver_impl() override;

  // See interface for documentation.
  void start(frame_notifier& notifier_) override;

  // See interface for documentation.
  void stop() override;

private:
  /// Main receiving loop.
  void receive_loop();

  /// Receives new Ethernet frames from the socket.
  ///
  /// \note This function will block until new frames become available.
  void receive();

private:
  srslog::basic_logger&                  logger;
  task_executor&                         executor;
  std::reference_wrapper<frame_notifier> notifier;
  int                                    socket_fd = -1;
  std::atomic<receiver_status>           rx_status{receiver_status::idle};
  ethernet_rx_buffer_pool                buffer_pool;
};

} // namespace ether
} // namespace srsran
