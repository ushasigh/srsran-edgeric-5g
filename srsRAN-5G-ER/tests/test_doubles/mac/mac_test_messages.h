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

#include "srsran/mac/mac_cell_control_information_handler.h"
#include "srsran/mac/mac_pdu_handler.h"
#include "srsran/scheduler/harq_id.h"
#include "srsran/scheduler/scheduler_slot_handler.h"

namespace srsran {

struct pucch_info;

namespace test_helpers {

/// Creates a CCCH message with the given RNTI.
mac_rx_data_indication
create_ccch_message(slot_point sl_rx, rnti_t rnti, du_cell_index_t du_cell_index = to_du_cell_index(0));

mac_rx_data_indication create_pdu_with_sdu(slot_point sl_rx, rnti_t rnti, lcid_t lcid);

/// \brief Generate MAC UCI PDU from PUCCH information, with all HARQ-ACKs set to ACK, SRs set as not detected and CSI
/// set as 1s.
mac_uci_pdu create_uci_pdu(const pucch_info& pucch);

/// Generate MAC UCI PDU out of a PUSCH UCI PDU indication, with all HARQ-ACKs set to ACK and CSI set as 1s.
mac_uci_pdu create_uci_pdu(rnti_t rnti, const uci_info& pusch_uci);

mac_uci_indication_message create_uci_indication(slot_point sl_rx, span<const pucch_info> pucchs);

std::optional<mac_uci_indication_message> create_uci_indication(slot_point sl_rx, span<const ul_sched_info> puschs);

mac_crc_indication_message create_crc_indication(slot_point sl_rx, rnti_t rnti, harq_id_t h_id);

mac_crc_indication_message create_crc_indication(slot_point sl_rx, span<const ul_sched_info> puschs);

} // namespace test_helpers
} // namespace srsran
