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

#include "../logging/scheduler_event_logger.h"
#include "../policy/scheduler_policy.h"
#include "../pucch_scheduling/pucch_guardbands_scheduler.h"
#include "../slicing/slice_scheduler.h"
#include "../support/slot_sync_point.h"
#include "../uci_scheduling/uci_scheduler_impl.h"
#include "ue_cell_grid_allocator.h"
#include "ue_event_manager.h"
#include "ue_fallback_scheduler.h"
#include "ue_repository.h"
#include "ue_scheduler.h"
#include "srsran/scheduler/config/scheduler_expert_config.h"

namespace srsran {

/// \brief Interface of data scheduler that is used to allocate UE DL and UL grants in a given slot.
/// The data_scheduler object will be common to all cells and slots.
class ue_scheduler_impl final : public ue_scheduler
{
public:
  explicit ue_scheduler_impl(const scheduler_ue_expert_config& expert_cfg_,
                             sched_configuration_notifier&     mac_notif,
                             scheduler_metrics_handler&        metric_handler);

  void add_cell(const ue_scheduler_cell_params& params) override;

  /// Schedule UE DL grants for a given {slot, cell}.
  void run_slot(slot_point slot_tx, du_cell_index_t cell_index) override;

  void handle_error_indication(slot_point                            sl_tx,
                               du_cell_index_t                       cell_index,
                               scheduler_slot_handler::error_outcome event) override
  {
    event_mng.handle_error_indication(sl_tx, cell_index, event);
  }

  sched_ue_configuration_handler& get_ue_configurator() override { return event_mng; }

  scheduler_feedback_handler& get_feedback_handler() override { return event_mng; }

  scheduler_dl_buffer_state_indication_handler& get_dl_buffer_state_indication_handler() override { return event_mng; }

private:
  void run_sched_strategy(slot_point sl_tx, du_cell_index_t cell_index);

  /// Counts the number of PUCCH grants that are allocated for a given user at a specific slot.
  void update_harq_pucch_counter(cell_resource_allocator& cell_alloc);

  struct cell {
    cell_resource_allocator* cell_res_alloc;

    /// PUCCH scheduler.
    uci_scheduler_impl uci_sched;

    /// Fallback scheduler.
    ue_fallback_scheduler fallback_sched;

    /// Slice scheduler.
    slice_scheduler slice_sched;

    cell(const scheduler_ue_expert_config& expert_cfg, const ue_scheduler_cell_params& params, ue_repository& ues) :
      cell_res_alloc(params.cell_res_alloc),
      uci_sched(params.cell_res_alloc->cfg, *params.uci_alloc, ues),
      fallback_sched(expert_cfg, params.cell_res_alloc->cfg, *params.pdcch_sched, *params.pucch_alloc, ues),
      slice_sched(params.cell_res_alloc->cfg, ues)
    {
    }
  };

  // Helper to catch simultaneous PUCCH and PUSCH grants allocated for the same UE.
  // TODO: remove this if no longer needed.
  void puxch_grant_sanitizer(cell_resource_allocator& cell_alloc);

  const scheduler_ue_expert_config& expert_cfg;

  std::array<std::unique_ptr<cell>, MAX_NOF_DU_CELLS> cells;

  /// Scheduling Strategy.
  ue_resource_grid_view ue_res_grid_view;

  /// Repository of created UEs.
  ue_repository ue_db;

  /// Allocator of grants in the resource grid.
  ue_cell_grid_allocator ue_alloc;

  /// Processor of UE input events.
  ue_event_manager event_mng;

  /// Mutex used to lock carriers for joint carrier scheduling.
  slot_sync_point sync_point;

  srslog::basic_logger& logger;
};

} // namespace srsran
