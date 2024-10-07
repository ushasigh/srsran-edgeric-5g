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

#include "procedures/rrc_reconfiguration_procedure.h"
#include "procedures/rrc_reestablishment_procedure.h"
#include "procedures/rrc_setup_procedure.h"
#include "procedures/rrc_ue_capability_transfer_procedure.h"
#include "rrc_asn1_helpers.h"
#include "rrc_ue_helpers.h"
#include "rrc_ue_impl.h"
#include "ue/rrc_measurement_types_asn1_converters.h"
#include "srsran/asn1/rrc_nr/dl_ccch_msg.h"
#include "srsran/asn1/rrc_nr/ul_ccch_msg.h"
#include "srsran/ran/lcid.h"

using namespace srsran;
using namespace srs_cu_cp;
using namespace asn1::rrc_nr;

void rrc_ue_impl::handle_ul_ccch_pdu(byte_buffer pdu)
{
  // Parse UL-CCCH
  ul_ccch_msg_s ul_ccch_msg;
  {
    asn1::cbit_ref bref(pdu);
    if (ul_ccch_msg.unpack(bref) != asn1::SRSASN_SUCCESS or
        ul_ccch_msg.msg.type().value != ul_ccch_msg_type_c::types_opts::c1) {
      logger.log_error(pdu.begin(), pdu.end(), "Failed to unpack CCCH UL PDU");
      on_ue_release_required(ngap_cause_radio_network_t::unspecified);
      return;
    }
  }

  // Log Rx message
  log_rrc_message(logger, Rx, pdu, ul_ccch_msg, "CCCH UL");

  // Handle message
  switch (ul_ccch_msg.msg.c1().type().value) {
    case ul_ccch_msg_type_c::c1_c_::types_opts::rrc_setup_request:
      handle_rrc_setup_request(ul_ccch_msg.msg.c1().rrc_setup_request());
      break;
    case ul_ccch_msg_type_c::c1_c_::types_opts::rrc_reest_request:
      handle_rrc_reest_request(ul_ccch_msg.msg.c1().rrc_reest_request());
      break;
    default:
      logger.log_error("Unsupported CCCH UL message type");
      on_ue_release_required(ngap_cause_radio_network_t::unspecified);
  }
}

void rrc_ue_impl::handle_rrc_setup_request(const asn1::rrc_nr::rrc_setup_request_s& request_msg)
{
  // Perform various checks to make sure we can serve the RRC Setup Request
  if (not cu_cp_notifier.on_ue_setup_request()) {
    logger.log_error("Sending Connection Reject. Cause: RRC connections not allowed");
    on_ue_release_required(ngap_cause_radio_network_t::unspecified);
    return;
  }

  if (du_to_cu_container.empty()) {
    // If the DU to CU container is missing, assume the DU can't serve the UE, so the CU-CP should reject the UE, see
    // TS 38.473 section 8.4.1.2.
    logger.log_debug("Sending rrcReject. Cause: DU is not able to serve the UE");
    on_ue_release_required(ngap_cause_radio_network_t::unspecified);
    return;
  }

  // Extract the setup ID and cause
  const rrc_setup_request_ies_s& request_ies = request_msg.rrc_setup_request;
  switch (request_ies.ue_id.type().value) {
    case init_ue_id_c::types_opts::ng_5_g_s_tmsi_part1: {
      context.setup_ue_id = request_ies.ue_id.ng_5_g_s_tmsi_part1();
      break;
    }
    case asn1::rrc_nr::init_ue_id_c::types_opts::random_value:
      context.setup_ue_id = request_ies.ue_id.random_value().to_number();
      // TODO: communicate with NGAP
      break;
    default:
      logger.log_error("Unsupported RRCSetupRequest");
      on_ue_release_required(ngap_cause_radio_network_t::unspecified);
      return;
  }
  context.connection_cause.value = request_ies.establishment_cause.value;

  // Launch RRC setup procedure
  cu_cp_ue_notifier.schedule_async_task(launch_async<rrc_setup_procedure>(
      context, du_to_cu_container, *this, get_rrc_ue_srb_handler(), nas_notifier, *event_mng, logger));
}

void rrc_ue_impl::handle_rrc_reest_request(const asn1::rrc_nr::rrc_reest_request_s& msg)
{
  // Launch RRC re-establishment procedure
  cu_cp_ue_notifier.schedule_async_task(launch_async<rrc_reestablishment_procedure>(msg,
                                                                                    context,
                                                                                    du_to_cu_container,
                                                                                    *this,
                                                                                    *this,
                                                                                    get_rrc_ue_srb_handler(),
                                                                                    cu_cp_notifier,
                                                                                    cu_cp_ue_notifier,
                                                                                    nas_notifier,
                                                                                    *event_mng,
                                                                                    logger));
}

void rrc_ue_impl::stop()
{
  event_mng->transactions.stop();
}

void rrc_ue_impl::handle_pdu(const srb_id_t srb_id, byte_buffer rrc_pdu)
{
  // Parse UL-DCCH
  ul_dcch_msg_s ul_dcch_msg;
  {
    asn1::cbit_ref bref(rrc_pdu);
    if (ul_dcch_msg.unpack(bref) != asn1::SRSASN_SUCCESS or
        ul_dcch_msg.msg.type().value != ul_dcch_msg_type_c::types_opts::c1) {
      logger.log_error(rrc_pdu.begin(), rrc_pdu.end(), "Failed to unpack DCCH UL PDU");
      return;
    }
  }

  // Log Rx message
  if (logger.get_basic_logger().debug.enabled()) {
    fmt::memory_buffer fmtbuf;
    fmt::format_to(fmtbuf, "{} DCCH UL", srb_id);
    log_rrc_message(logger, Rx, rrc_pdu, ul_dcch_msg, to_c_str(fmtbuf));
  }

  switch (ul_dcch_msg.msg.c1().type().value) {
    case ul_dcch_msg_type_c::c1_c_::types_opts::options::ul_info_transfer:
      handle_ul_info_transfer(ul_dcch_msg.msg.c1().ul_info_transfer().crit_exts.ul_info_transfer());
      break;
    case ul_dcch_msg_type_c::c1_c_::types_opts::rrc_setup_complete:
      handle_rrc_transaction_complete(ul_dcch_msg, ul_dcch_msg.msg.c1().rrc_setup_complete().rrc_transaction_id);
      break;
    case ul_dcch_msg_type_c::c1_c_::types_opts::security_mode_complete:
      handle_rrc_transaction_complete(ul_dcch_msg, ul_dcch_msg.msg.c1().security_mode_complete().rrc_transaction_id);
      break;
    case ul_dcch_msg_type_c::c1_c_::types_opts::security_mode_fail:
      handle_rrc_transaction_complete(ul_dcch_msg, ul_dcch_msg.msg.c1().security_mode_fail().rrc_transaction_id);
      break;
    case ul_dcch_msg_type_c::c1_c_::types_opts::ue_cap_info:
      handle_rrc_transaction_complete(ul_dcch_msg, ul_dcch_msg.msg.c1().ue_cap_info().rrc_transaction_id);
      break;
    case ul_dcch_msg_type_c::c1_c_::types_opts::rrc_recfg_complete:
      if (context.transfer_context.has_value() && context.transfer_context.value().is_inter_cu_handover) {
        logger.log_debug("Received a RRC Reconfiguration Complete during inter CU handover. Notifying NGAP");
        ngap_ctrl_notifier.on_inter_cu_ho_rrc_recfg_complete_received(
            context.ue_index, context.cell.cgi, context.cell.tac);
        context.transfer_context.value().is_inter_cu_handover = false;
      } else {
        handle_rrc_transaction_complete(ul_dcch_msg, ul_dcch_msg.msg.c1().rrc_recfg_complete().rrc_transaction_id);
      }
      break;
    case ul_dcch_msg_type_c::c1_c_::types_opts::rrc_reest_complete:
      handle_rrc_transaction_complete(ul_dcch_msg, ul_dcch_msg.msg.c1().rrc_reest_complete().rrc_transaction_id);
      break;
    case ul_dcch_msg_type_c::c1_c_::types_opts::meas_report:
      handle_measurement_report(ul_dcch_msg.msg.c1().meas_report());
      break;
    default:
      logger.log_error("Unsupported DCCH UL message type");
      break;
  }
  // TODO: Handle message
}

void rrc_ue_impl::handle_ul_dcch_pdu(const srb_id_t srb_id, byte_buffer pdcp_pdu)
{
  logger.log_debug(pdcp_pdu.begin(), pdcp_pdu.end(), "RX {} PDCP PDU", srb_id);

  if (context.srbs.find(srb_id) == context.srbs.end()) {
    logger.log_error(pdcp_pdu.begin(), pdcp_pdu.end(), "Dropping UL-DCCH PDU. Rx {} is not set up", srb_id);
    return;
  }

  // Unpack PDCP PDU
  pdcp_rx_result pdcp_unpacking_result = context.srbs.at(srb_id).unpack_pdcp_pdu(std::move(pdcp_pdu));
  if (!pdcp_unpacking_result.is_successful()) {
    logger.log_info("Requesting UE release. Cause: PDCP unpacking failed with {}",
                    pdcp_unpacking_result.get_failure_cause());
    on_ue_release_required(pdcp_unpacking_result.get_failure_cause());
    return;
  }

  std::vector<byte_buffer> rrc_pdus = pdcp_unpacking_result.pop_pdus();
  for (byte_buffer& pdu : rrc_pdus) {
    handle_pdu(srb_id, std::move(pdu));
  }
}

void rrc_ue_impl::handle_security_mode_complete(const asn1::rrc_nr::security_mode_complete_s& msg)
{
  srsran_sanity_check(context.srbs.find(srb_id_t::srb1) != context.srbs.end(),
                      "Attempted to configure security, but there is no interface to PDCP");

  context.srbs.at(srb_id_t::srb1)
      .enable_rx_security(
          security::integrity_enabled::on, security::ciphering_enabled::on, cu_cp_ue_notifier.get_rrc_128_as_config());
  context.srbs.at(srb_id_t::srb1)
      .enable_tx_security(
          security::integrity_enabled::on, security::ciphering_enabled::on, cu_cp_ue_notifier.get_rrc_128_as_config());
}

void rrc_ue_impl::handle_ul_info_transfer(const ul_info_transfer_ies_s& ul_info_transfer)
{
  cu_cp_ul_nas_transport ul_nas_msg         = {};
  ul_nas_msg.ue_index                       = context.ue_index;
  ul_nas_msg.nas_pdu                        = ul_info_transfer.ded_nas_msg.copy();
  ul_nas_msg.user_location_info.nr_cgi      = context.cell.cgi;
  ul_nas_msg.user_location_info.tai.plmn_id = context.cell.cgi.plmn_id;
  ul_nas_msg.user_location_info.tai.tac     = context.cell.tac;

  nas_notifier.on_ul_nas_transport_message(ul_nas_msg);
}

void rrc_ue_impl::handle_measurement_report(const asn1::rrc_nr::meas_report_s& msg)
{
  // convert asn1 to common type
  rrc_meas_results meas_results =
      asn1_to_measurement_results(msg.crit_exts.meas_report().meas_results, srslog::fetch_basic_logger("RRC"));
  // Send measurement results to cell measurement manager only measurements are not empty.
  if (meas_results.meas_result_neigh_cells.has_value() and
      not meas_results.meas_result_neigh_cells->meas_result_list_nr.empty()) {
    measurement_notifier.on_measurement_report(meas_results);
  }
}

void rrc_ue_impl::handle_dl_nas_transport_message(byte_buffer nas_pdu)
{
  dl_dcch_msg_s           dl_dcch_msg;
  dl_info_transfer_ies_s& dl_info_transfer =
      dl_dcch_msg.msg.set_c1().set_dl_info_transfer().crit_exts.set_dl_info_transfer();
  dl_info_transfer.ded_nas_msg = nas_pdu.copy();

  if (context.srbs.find(srb_id_t::srb2) != context.srbs.end()) {
    send_dl_dcch(srb_id_t::srb2, dl_dcch_msg);
  } else {
    send_dl_dcch(srb_id_t::srb1, dl_dcch_msg);
  }
}

void rrc_ue_impl::handle_rrc_transaction_complete(const ul_dcch_msg_s& msg, uint8_t transaction_id_)
{
  expected<uint8_t> transaction_id = transaction_id_;

  // Set transaction result and resume suspended procedure.
  if (not event_mng->transactions.set_response(transaction_id.value(), msg)) {
    logger.log_warning("Unexpected RRC transaction id={}", transaction_id.value());
  }
}

rrc_ue_security_mode_command_context rrc_ue_impl::get_security_mode_command_context()
{
  // activate SRB1 PDCP security
  on_new_as_security_context();

  rrc_ue_security_mode_command_context smc_ctxt;

  if (context.srbs.find(srb_id_t::srb1) == context.srbs.end()) {
    logger.log_error("Can't get security mode command. {} is not set up", srb_id_t::srb1);
    return smc_ctxt;
  }

  // Create transaction to get transaction ID
  rrc_transaction transaction = event_mng->transactions.create_transaction();
  smc_ctxt.transaction_id     = transaction.id();

  // Get selected security algorithms
  security::sec_selected_algos security_algos = cu_cp_ue_notifier.get_security_algos();

  // Pack SecurityModeCommand
  dl_dcch_msg_s dl_dcch_msg;
  dl_dcch_msg.msg.set_c1().set_security_mode_cmd().crit_exts.set_security_mode_cmd();
  fill_asn1_rrc_smc_msg(dl_dcch_msg.msg.c1().security_mode_cmd(),
                        security_algos.integ_algo,
                        security_algos.cipher_algo,
                        smc_ctxt.transaction_id);

  // Pack DL DCCH msg
  pdcp_tx_result pdcp_packing_result =
      context.srbs.at(srb_id_t::srb1).pack_rrc_pdu(pack_into_pdu(dl_dcch_msg, "SecurityModeCommand"));
  if (!pdcp_packing_result.is_successful()) {
    logger.log_info("Requesting UE release. Cause: PDCP packing failed with {}",
                    pdcp_packing_result.get_failure_cause());
    on_ue_release_required(pdcp_packing_result.get_failure_cause());
    return smc_ctxt;
  }

  smc_ctxt.rrc_ue_security_mode_command_pdu = pdcp_packing_result.pop_pdu();
  smc_ctxt.sp_cell_id                       = context.cell.cgi;

  // Log Tx message
  log_rrc_message(logger, Tx, smc_ctxt.rrc_ue_security_mode_command_pdu, dl_dcch_msg, "DCCH DL");

  return smc_ctxt;
}

async_task<bool> rrc_ue_impl::handle_security_mode_complete_expected(uint8_t transaction_id)
{
  // arbitrary timeout for RRC Reconfig procedure, UE will be removed if timer fires
  const std::chrono::milliseconds timeout_ms{1000};

  return launch_async(
      [this, timeout_ms, transaction_id, transaction = rrc_transaction{}](coro_context<async_task<bool>>& ctx) mutable {
        CORO_BEGIN(ctx);

        logger.log_debug("Awaiting RRC Security Mode Complete (timeout={}ms)", timeout_ms.count());
        // create new transaction for RRC Security Mode Command procedure
        transaction = event_mng->transactions.create_transaction(transaction_id, timeout_ms);

        CORO_AWAIT(transaction);

        if (!transaction.has_response()) {
          logger.log_debug("Did not receive RRC Security Mode Complete. Cause: timeout");
          CORO_EARLY_RETURN(false);
        }

        if (transaction.response().msg.c1().type() == ul_dcch_msg_type_c::c1_c_::types_opts::security_mode_fail) {
          logger.log_warning("Received RRC Security Mode Failure");
          CORO_EARLY_RETURN(false);
        }

        if (transaction.response().msg.c1().type() == ul_dcch_msg_type_c::c1_c_::types_opts::security_mode_complete) {
          logger.log_debug("Received RRC Security Mode Complete");
          handle_security_mode_complete(transaction.response().msg.c1().security_mode_complete());
        }

        CORO_RETURN(true);
      });
}

byte_buffer rrc_ue_impl::get_packed_ue_capability_rat_container_list() const
{
  byte_buffer pdu{};

  if (context.capabilities_list.has_value()) {
    asn1::bit_ref bref{pdu};

    if (pack_dyn_seq_of(bref, context.capabilities_list.value(), 0, 8) != asn1::SRSASN_SUCCESS) {
      logger.log_error("Error packing UE Capability RAT Container List");
      return byte_buffer{};
    }
  } else {
    logger.log_debug("No UE capabilites available");
  }

  return pdu.copy();
}

byte_buffer rrc_ue_impl::get_packed_ue_radio_access_cap_info() const
{
  asn1::rrc_nr::ue_radio_access_cap_info_s      ue_radio_access_cap_info;
  asn1::rrc_nr::ue_radio_access_cap_info_ies_s& ue_radio_access_cap_info_ies =
      ue_radio_access_cap_info.crit_exts.set_c1().set_ue_radio_access_cap_info();
  ue_radio_access_cap_info_ies.ue_radio_access_cap_info = get_packed_ue_capability_rat_container_list();

  return pack_into_pdu(ue_radio_access_cap_info, "UE Radio Access Cap Info");
}

async_task<bool> rrc_ue_impl::handle_rrc_reconfiguration_request(const rrc_reconfiguration_procedure_request& msg)
{
  return launch_async<rrc_reconfiguration_procedure>(context, msg, *this, *event_mng, get_rrc_ue_srb_handler(), logger);
}

rrc_ue_handover_reconfiguration_context
rrc_ue_impl::get_rrc_ue_handover_reconfiguration_context(const rrc_reconfiguration_procedure_request& request)
{
  rrc_ue_handover_reconfiguration_context ho_reconf_ctxt;

  if (context.srbs.find(srb_id_t::srb1) == context.srbs.end()) {
    logger.log_error("Can't get handover reconfiguraion context. {} is not set up", srb_id_t::srb1);
    return ho_reconf_ctxt;
  }

  // Create transaction to get transaction ID
  rrc_transaction transaction   = event_mng->transactions.create_transaction();
  ho_reconf_ctxt.transaction_id = transaction.id();

  // Pack RRC Reconfig
  dl_dcch_msg_s dl_dcch_msg;
  dl_dcch_msg.msg.set_c1().set_rrc_recfg().crit_exts.set_rrc_recfg();
  fill_asn1_rrc_reconfiguration_msg(dl_dcch_msg.msg.c1().rrc_recfg(), ho_reconf_ctxt.transaction_id, request);

  // Pack DL DCCH msg
  pdcp_tx_result pdcp_packing_result =
      context.srbs.at(srb_id_t::srb1).pack_rrc_pdu(pack_into_pdu(dl_dcch_msg, "RRCReconfiguration"));
  if (!pdcp_packing_result.is_successful()) {
    logger.log_info("Requesting UE release. Cause: PDCP packing failed with {}",
                    pdcp_packing_result.get_failure_cause());
    on_ue_release_required(pdcp_packing_result.get_failure_cause());
    return ho_reconf_ctxt;
  }

  ho_reconf_ctxt.rrc_ue_handover_reconfiguration_pdu = pdcp_packing_result.pop_pdu();

  // Log Tx message
  log_rrc_message(logger, Tx, ho_reconf_ctxt.rrc_ue_handover_reconfiguration_pdu, dl_dcch_msg, "DCCH DL");

  return ho_reconf_ctxt;
}

async_task<bool> rrc_ue_impl::handle_handover_reconfiguration_complete_expected(uint8_t transaction_id)
{
  // arbitrary timeout for RRC Reconfig procedure, UE will be removed if timer fires
  const std::chrono::milliseconds timeout_ms{1000};

  return launch_async(
      [this, timeout_ms, transaction_id, transaction = rrc_transaction{}](coro_context<async_task<bool>>& ctx) mutable {
        CORO_BEGIN(ctx);

        logger.log_debug("Awaiting RRC Reconfiguration Complete (timeout={}ms)", timeout_ms.count());
        // create new transaction for RRC Reconfiguration procedure
        transaction = event_mng->transactions.create_transaction(transaction_id, timeout_ms);

        CORO_AWAIT(transaction);

        bool procedure_result = false;
        if (transaction.has_response()) {
          logger.log_debug("Received RRC Reconfiguration Complete after HO");
          procedure_result = true;

          // The UE in the target cell is in connected state on RRC Reconfiguration Complete reception.
          context.state = rrc_state::connected;

        } else {
          logger.log_debug("Did not receive RRC Reconfiguration Complete after HO. Cause: timeout");
        }

        CORO_RETURN(procedure_result);
      });
}

async_task<bool> rrc_ue_impl::handle_rrc_ue_capability_transfer_request(const rrc_ue_capability_transfer_request& msg)
{
  //  Launch RRC UE capability transfer procedure
  return launch_async<rrc_ue_capability_transfer_procedure>(context, *this, *event_mng, logger);
}

rrc_ue_release_context rrc_ue_impl::get_rrc_ue_release_context(bool requires_rrc_message)
{
  // prepare location info to return
  rrc_ue_release_context release_context;
  release_context.user_location_info.nr_cgi      = context.cell.cgi;
  release_context.user_location_info.tai.plmn_id = context.cell.cgi.plmn_id;
  release_context.user_location_info.tai.tac     = context.cell.tac;

  if (requires_rrc_message) {
    if (context.srbs.empty()) {
      // SRB1 was not created, so we need to reject the UE
      // Create and RRCReject container, see section 5.3.15 in TS 38.331
      dl_ccch_msg_s dl_ccch_msg;
      // SRB1 was not created, so we create a RRC Container with RRCReject
      rrc_reject_ies_s& reject = dl_ccch_msg.msg.set_c1().set_rrc_reject().crit_exts.set_rrc_reject();

      // See TS 38.331, RejectWaitTime
      reject.wait_time_present = true;
      reject.wait_time         = rrc_reject_max_wait_time_s;

      // pack DL CCCH msg
      release_context.rrc_release_pdu = pack_into_pdu(dl_ccch_msg, "RRCReject");
      release_context.srb_id          = srb_id_t::srb0;

      // Log Tx message
      log_rrc_message(logger, Tx, release_context.rrc_release_pdu, dl_ccch_msg, "CCCH DL");
    } else {
      // prepare SRB1 RRC Release PDU to return
      if (context.srbs.find(srb_id_t::srb1) == context.srbs.end()) {
        logger.log_error("Can't create RRCRelease PDU. RX {} is not set up", srb_id_t::srb1);
        return release_context;
      }

      dl_dcch_msg_s dl_dcch_msg;
      dl_dcch_msg.msg.set_c1().set_rrc_release().crit_exts.set_rrc_release();

      // pack DL CCCH msg
      pdcp_tx_result pdcp_packing_result =
          context.srbs.at(srb_id_t::srb1).pack_rrc_pdu(pack_into_pdu(dl_dcch_msg, "RRCRelease"));
      if (!pdcp_packing_result.is_successful()) {
        logger.log_info("Requesting UE release. Cause: PDCP packing failed with {}",
                        pdcp_packing_result.get_failure_cause());
        on_ue_release_required(pdcp_packing_result.get_failure_cause());
        return release_context;
      }

      release_context.rrc_release_pdu = pdcp_packing_result.pop_pdu();
      release_context.srb_id          = srb_id_t::srb1;

      // Log Tx message
      log_rrc_message(logger, Tx, release_context.rrc_release_pdu, dl_dcch_msg, "DCCH DL");
    }

    // Log Tx message
    logger.log_debug(release_context.rrc_release_pdu.begin(),
                     release_context.rrc_release_pdu.end(),
                     "TX {} PDU",
                     release_context.srb_id);
  }

  return release_context;
}

std::optional<rrc_meas_cfg> rrc_ue_impl::generate_meas_config(std::optional<rrc_meas_cfg> current_meas_config)
{
  // (Re-)generate measurement config and return result.
  context.meas_cfg = measurement_notifier.on_measurement_config_request(context.cell.cgi.nci, current_meas_config);
  return context.meas_cfg;
}

rrc_ue_transfer_context rrc_ue_impl::get_transfer_context()
{
  rrc_ue_transfer_context transfer_context;
  transfer_context.sec_context               = cu_cp_ue_notifier.get_security_context();
  transfer_context.meas_cfg                  = context.meas_cfg;
  transfer_context.srbs                      = get_srbs();
  transfer_context.up_ctx                    = cu_cp_notifier.on_up_context_required();
  transfer_context.handover_preparation_info = get_packed_handover_preparation_message();
  transfer_context.ue_cap_rat_container_list = get_packed_ue_capability_rat_container_list();

  return transfer_context;
}

rrc_ue_reestablishment_context_response rrc_ue_impl::get_context()
{
  rrc_ue_reestablishment_context_response rrc_reest_context;
  rrc_reest_context.sec_context = cu_cp_ue_notifier.get_security_context();

  if (context.capabilities.has_value()) {
    rrc_reest_context.capabilities = context.capabilities.value();
  }
  rrc_reest_context.up_ctx = cu_cp_notifier.on_up_context_required();

  // TODO: Handle scenario with multiple reestablishments for the same UE
  rrc_reest_context.reestablishment_ongoing = context.reestablishment_ongoing;

  // If no reestablishment is ongoing, set it to true.
  if (not context.reestablishment_ongoing) {
    context.reestablishment_ongoing = true;
  }

  return rrc_reest_context;
}

byte_buffer rrc_ue_impl::get_rrc_handover_command(const rrc_reconfiguration_procedure_request& request,
                                                  unsigned                                     transaction_id)
{
  // pack RRC Reconfig
  rrc_recfg_s rrc_reconfig;
  fill_asn1_rrc_reconfiguration_msg(rrc_reconfig, transaction_id, request);
  byte_buffer reconfig_pdu = pack_into_pdu(rrc_reconfig, "RRCReconfiguration");

  ho_cmd_s ho_cmd;
  ho_cmd.crit_exts.set_c1().set_ho_cmd().ho_cmd_msg = reconfig_pdu.copy();

  // pack Handover Command
  byte_buffer ho_cmd_pdu = pack_into_pdu(ho_cmd, "RRCHandoverCommand");

  // Log message
  logger.log_debug(ho_cmd_pdu.begin(), ho_cmd_pdu.end(), "RRCHandoverCommand ({} B)", ho_cmd_pdu.length());
  if (logger.get_basic_logger().debug.enabled()) {
    asn1::json_writer js;
    ho_cmd.to_json(js);
    logger.log_debug("Containerized RRCHandoverCommand: {}", js.to_string());
  }

  return ho_cmd_pdu;
}

byte_buffer rrc_ue_impl::handle_rrc_handover_command(byte_buffer cmd)
{
  byte_buffer ho_reconf_pdu = byte_buffer{};

  // Unpack Handover Command
  asn1::rrc_nr::ho_cmd_s handover_command;
  asn1::cbit_ref         bref({cmd.begin(), cmd.end()});

  if (handover_command.unpack(bref) != asn1::SRSASN_SUCCESS) {
    logger.log_error("Couldn't unpack Handover Command RRC container");
    return ho_reconf_pdu;
  }

  // Unpack RRC Reconfiguration to new DL DCCH Message
  asn1::cbit_ref bref2({handover_command.crit_exts.c1().ho_cmd().ho_cmd_msg.begin(),
                        handover_command.crit_exts.c1().ho_cmd().ho_cmd_msg.end()});

  dl_dcch_msg_s dl_dcch_msg;
  auto&         rrc_recfg = dl_dcch_msg.msg.set_c1().set_rrc_recfg();

  if (rrc_recfg.unpack(bref2) != asn1::SRSASN_SUCCESS) {
    logger.log_error("Couldn't unpack RRC Reconfiguration container");
    return ho_reconf_pdu;
  }

  // pack DL CCCH msg
  pdcp_tx_result pdcp_packing_result =
      context.srbs.at(srb_id_t::srb1).pack_rrc_pdu(pack_into_pdu(dl_dcch_msg, "RRCReconfiguration"));
  if (!pdcp_packing_result.is_successful()) {
    logger.log_info("Requesting UE release. Cause: PDCP packing failed with {}",
                    pdcp_packing_result.get_failure_cause());
    on_ue_release_required(pdcp_packing_result.get_failure_cause());
    return ho_reconf_pdu;
  }

  ho_reconf_pdu = pdcp_packing_result.pop_pdu();

  // Log Tx message
  log_rrc_message(logger, Tx, ho_reconf_pdu, dl_dcch_msg, "DCCH DL");

  return ho_reconf_pdu;
}
