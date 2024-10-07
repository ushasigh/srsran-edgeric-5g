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

/// \file
/// \brief Tests that check the setup/teardown, addition/removal of UEs in the DU-high class.

#include "tests/integrationtests/du_high/test_utils/du_high_env_simulator.h"
#include "tests/test_doubles/f1ap/f1ap_test_message_validators.h"
#include "tests/test_doubles/f1ap/f1ap_test_messages.h"
#include "tests/test_doubles/pdcp/pdcp_pdu_generator.h"
#include "tests/test_doubles/scheduler/scheduler_result_test.h"
#include "tests/unittests/f1ap/du/f1ap_du_test_helpers.h"
#include "srsran/asn1/f1ap/common.h"
#include "srsran/asn1/f1ap/f1ap_pdu_contents_ue.h"
#include "srsran/asn1/rrc_nr/cell_group_config.h"
#include "srsran/support/test_utils.h"

using namespace srsran;
using namespace srs_du;
using namespace asn1::f1ap;

class du_high_tester : public du_high_env_simulator, public testing::Test
{};

/// Test F1 setup over "local" connection to DU.
TEST_F(du_high_tester, when_du_high_started_then_connection_to_cu_takes_place)
{
  // Starting the DU-high initiates the F1 Setup procedure.
  ASSERT_EQ(this->cu_notifier.last_f1ap_msgs.size(), 1);
  ASSERT_EQ(this->cu_notifier.last_f1ap_msgs.back().pdu.type().value, f1ap_pdu_c::types_opts::init_msg);
  ASSERT_EQ(this->cu_notifier.last_f1ap_msgs.back().pdu.init_msg().proc_code, ASN1_F1AP_ID_F1_SETUP);
}

TEST_F(du_high_tester, when_ccch_msg_is_received_then_ue_context_is_created)
{
  // Reset the last sent F1AP PDU (e.g. F1 setup).
  cu_notifier.last_f1ap_msgs.clear();

  // Add UE
  du_hi->get_pdu_handler().handle_rx_data_indication(test_helpers::create_ccch_message(next_slot, to_rnti(0x4601)));

  this->run_until([this]() { return not cu_notifier.last_f1ap_msgs.empty(); });
  ASSERT_EQ(cu_notifier.last_f1ap_msgs.size(), 1);
  ASSERT_TRUE(test_helpers::is_init_ul_rrc_msg_transfer_valid(cu_notifier.last_f1ap_msgs.back(), to_rnti(0x4601)));
}

TEST_F(du_high_tester, when_two_concurrent_ccch_msg_are_received_then_two_ue_contexts_are_created)
{
  cu_notifier.last_f1ap_msgs.clear();

  // Add two UEs.
  du_hi->get_pdu_handler().handle_rx_data_indication(test_helpers::create_ccch_message(next_slot, to_rnti(0x4601)));
  du_hi->get_pdu_handler().handle_rx_data_indication(test_helpers::create_ccch_message(next_slot, to_rnti(0x4602)));

  this->run_until([this]() { return cu_notifier.last_f1ap_msgs.size() >= 2; });
  ASSERT_EQ(cu_notifier.last_f1ap_msgs.size(), 2);
  ASSERT_TRUE(test_helpers::is_init_ul_rrc_msg_transfer_valid(cu_notifier.last_f1ap_msgs[0], to_rnti(0x4601)));
  ASSERT_TRUE(test_helpers::is_init_ul_rrc_msg_transfer_valid(cu_notifier.last_f1ap_msgs[1], to_rnti(0x4602)));
}

TEST_F(du_high_tester, when_ue_context_setup_completes_then_drb_is_active)
{
  // Create UE.
  rnti_t rnti = to_rnti(0x4601);
  ASSERT_TRUE(add_ue(rnti));
  ASSERT_TRUE(run_rrc_setup(rnti));
  ASSERT_TRUE(run_ue_context_setup(rnti));

  // Ensure DU<->CU-UP tunnel was created.
  ASSERT_EQ(cu_up_sim.last_drb_id.value(), drb_id_t::drb1);

  // Forward several DRB PDUs.
  const unsigned nof_pdcp_pdus = 100, pdcp_pdu_size = 128;
  for (unsigned i = 0; i < nof_pdcp_pdus; ++i) {
    nru_dl_message f1u_pdu{
        .t_pdu = test_helpers::create_pdcp_pdu(pdcp_sn_size::size12bits, /* is_srb = */ false, i, pdcp_pdu_size, i)};
    cu_up_sim.created_du_notifs[0]->on_new_pdu(f1u_pdu);
  }

  // Ensure DRB is active by verifying that the DRB PDUs are scheduled.
  unsigned bytes_sched = 0;
  phy.cells[0].last_dl_data.reset();
  while (bytes_sched < nof_pdcp_pdus * pdcp_pdu_size and this->run_until([this]() {
    return phy.cells[0].last_dl_data.has_value() and not phy.cells[0].last_dl_data.value().ue_pdus.empty();
  })) {
    for (unsigned i = 0; i != phy.cells[0].last_dl_data.value().ue_pdus.size(); ++i) {
      if (phy.cells[0].last_dl_res.value().dl_res->ue_grants[i].pdsch_cfg.codewords[0].new_data) {
        bytes_sched += phy.cells[0].last_dl_data.value().ue_pdus[i].pdu.size();
      }
    }
    phy.cells[0].last_dl_data.reset();
  }
  ASSERT_GE(bytes_sched, nof_pdcp_pdus * pdcp_pdu_size)
      << "Not enough PDSCH grants were scheduled to meet the enqueued PDCP PDUs";
}

TEST_F(du_high_tester, when_ue_context_release_received_then_ue_gets_deleted)
{
  // Create UE.
  ASSERT_TRUE(add_ue(to_rnti(0x4601)));

  // Receive UE Context Release Command.
  cu_notifier.last_f1ap_msgs.clear();
  f1ap_message                    msg = generate_ue_context_release_command();
  const ue_context_release_cmd_s& cmd = msg.pdu.init_msg().value.ue_context_release_cmd();
  this->du_hi->get_f1ap_message_handler().handle_message(msg);

  const unsigned MAX_COUNT = 1000;
  for (unsigned i = 0; i != MAX_COUNT; ++i) {
    this->run_slot();

    // Stop loop, When UE Context Release complete has been sent to CU.
    if (not cu_notifier.last_f1ap_msgs.empty()) {
      break;
    }
  }
  ASSERT_EQ(cu_notifier.last_f1ap_msgs.size(), 1);
  ASSERT_TRUE(is_ue_context_release_complete_valid(cu_notifier.last_f1ap_msgs.back(),
                                                   (gnb_du_ue_f1ap_id_t)cmd->gnb_du_ue_f1ap_id,
                                                   (gnb_cu_ue_f1ap_id_t)cmd->gnb_cu_ue_f1ap_id));
}

TEST_F(du_high_tester, when_ue_context_setup_release_starts_then_drb_activity_stops)
{
  // Create UE.
  rnti_t rnti = to_rnti(0x4601);
  ASSERT_TRUE(add_ue(rnti));
  ASSERT_TRUE(run_rrc_setup(rnti));
  ASSERT_TRUE(run_ue_context_setup(rnti));

  // CU-UP forwards many DRB PDUs.
  const unsigned nof_pdcp_pdus = 100, pdcp_pdu_size = 128;

  for (unsigned i = 0; i < nof_pdcp_pdus; ++i) {
    nru_dl_message f1u_pdu{
        .t_pdu = test_helpers::create_pdcp_pdu(pdcp_sn_size::size12bits, /* is_srb = */ false, i, pdcp_pdu_size, i)};
    cu_up_sim.created_du_notifs[0]->on_new_pdu(f1u_pdu);
  }

  // DU receives F1AP UE Context Release Command.
  cu_notifier.last_f1ap_msgs.clear();
  f1ap_message msg = generate_ue_context_release_command();
  this->du_hi->get_f1ap_message_handler().handle_message(msg);
  this->test_logger.info("STATUS: UEContextReleaseCommand received by DU. Waiting for rrcRelease being transmitted...");

  // Ensure that once SRB1 (RRC Release) is scheduled.
  run_slot();
  EXPECT_TRUE(this->run_until([this, rnti]() {
    return find_ue_pdsch_with_lcid(rnti, LCID_SRB1, phy.cells[0].last_dl_res.value().dl_res->ue_grants) != nullptr;
  }));
  this->test_logger.info("STATUS: RRC Release started being scheduled...");

  // Ensure that DRBs stop being scheduled at this point, even if it takes a while for the UE release to complete.
  unsigned drb_data_count = 0;
  while (cu_notifier.last_f1ap_msgs.empty()) {
    run_slot();
    const dl_msg_alloc* pdsch = find_ue_pdsch(rnti, phy.cells[0].last_dl_res.value().dl_res->ue_grants);
    if (pdsch != nullptr) {
      // PDSCH scheduled. Ensure it was for SRB1.
      // Note: There might be at most one single DRB1 PDSCH that smuggles in after the RRC Release due to race
      // conditions.
      auto* drb_pdsch = find_ue_pdsch_with_lcid(rnti, LCID_MIN_DRB, phy.cells[0].last_dl_res.value().dl_res->ue_grants);
      if (drb_pdsch != nullptr) {
        drb_data_count++;
        ASSERT_LT(drb_data_count, 2) << "More than 1 PDSCH grant for DRB data was scheduled after RRC Release";
      }
    }
  }
}

TEST_F(du_high_tester, when_du_high_is_stopped_then_ues_are_removed)
{
  // Create UE.
  ASSERT_TRUE(add_ue(to_rnti(0x4601)));

  // Stop DU from another thread, because stop() call is blocking.
  std::atomic<bool> running{true};
  task_worker       worker("phy_worker", 2048);
  bool              ret = worker.push_task([this, &running]() {
    this->du_hi->stop();
    running = false;
  });
  ASSERT_TRUE(ret);

  // Keep running slot indications until DU-high is stopped.
  for (unsigned count = 0; count < 1000 and running; ++count) {
    this->run_slot();
  }
  ASSERT_FALSE(running) << "stop() call is hanging";
}

TEST_F(du_high_tester, when_ue_context_setup_received_for_inexistent_ue_then_ue_is_created)
{
  cu_notifier.last_f1ap_msgs.clear();

  gnb_cu_ue_f1ap_id_t cu_ue_id =
      int_to_gnb_cu_ue_f1ap_id(test_rgen::uniform_int<uint64_t>(0, (uint64_t)gnb_cu_ue_f1ap_id_t::max));
  f1ap_message cu_cp_msg = test_helpers::create_ue_context_setup_request(cu_ue_id, std::nullopt, 0, {drb_id_t::drb1});
  this->du_hi->get_f1ap_message_handler().handle_message(cu_cp_msg);

  ASSERT_TRUE(this->run_until([this]() { return not cu_notifier.last_f1ap_msgs.empty(); }));

  ASSERT_EQ(cu_notifier.last_f1ap_msgs.size(), 1);
  ASSERT_EQ(cu_notifier.last_f1ap_msgs.back().pdu.type().value, f1ap_pdu_c::types::options::successful_outcome);
  ASSERT_EQ(cu_notifier.last_f1ap_msgs.back().pdu.successful_outcome().proc_code, ASN1_F1AP_ID_UE_CONTEXT_SETUP);
  auto& resp = cu_notifier.last_f1ap_msgs.back().pdu.successful_outcome().value.ue_context_setup_resp();
  ASSERT_EQ(resp->gnb_cu_ue_f1ap_id, (unsigned)cu_ue_id);
  ASSERT_TRUE(resp->c_rnti_present);
  ASSERT_TRUE(is_crnti(to_rnti(resp->c_rnti)));
}

TEST_F(du_high_tester, when_ue_context_modification_with_rem_drbs_is_received_then_drbs_are_removed)
{
  // Create UE.
  rnti_t rnti = to_rnti(0x4601);
  ASSERT_TRUE(add_ue(rnti));
  ASSERT_TRUE(run_rrc_setup(rnti));
  ASSERT_TRUE(run_ue_context_setup(rnti));

  // CU-UP forwards many DRB PDUs.
  const unsigned nof_pdcp_pdus = 100, pdcp_pdu_size = 128;

  for (unsigned i = 0; i < nof_pdcp_pdus; ++i) {
    nru_dl_message f1u_pdu{
        .t_pdu = test_helpers::create_pdcp_pdu(pdcp_sn_size::size12bits, /* is_srb = */ false, i, pdcp_pdu_size, i)};
    cu_up_sim.created_du_notifs[0]->on_new_pdu(f1u_pdu);
  }

  // DU receives F1AP UE Context Modification Command.
  cu_notifier.last_f1ap_msgs.clear();
  f1ap_message msg = generate_ue_context_modification_request({}, {drb_id_t::drb1});
  this->du_hi->get_f1ap_message_handler().handle_message(msg);

  // Wait for DU to send F1AP UE Context Modification Response.
  this->run_until([this]() { return not cu_notifier.last_f1ap_msgs.empty(); });

  const f1ap_message& f1ap_pdu = cu_notifier.last_f1ap_msgs.back();
  ASSERT_EQ(f1ap_pdu.pdu.type().value, f1ap_pdu_c::types::options::successful_outcome);
  ASSERT_EQ(f1ap_pdu.pdu.successful_outcome().proc_code, ASN1_F1AP_ID_UE_CONTEXT_MOD);
  const ue_context_mod_resp_s& resp = f1ap_pdu.pdu.successful_outcome().value.ue_context_mod_resp();
  ASSERT_TRUE(resp->du_to_cu_rrc_info_present);
  ASSERT_FALSE(resp->du_to_cu_rrc_info.cell_group_cfg.empty());
  {
    asn1::cbit_ref                 bref{resp->du_to_cu_rrc_info.cell_group_cfg};
    asn1::rrc_nr::cell_group_cfg_s cell_grp_cfg;
    ASSERT_EQ(cell_grp_cfg.unpack(bref), asn1::SRSASN_SUCCESS);
    ASSERT_EQ(cell_grp_cfg.rlc_bearer_to_release_list.size(), 1);
    ASSERT_EQ(cell_grp_cfg.rlc_bearer_to_release_list[0], 4) << "DRB1 with LCID=4 should have been removed";
  }
}
