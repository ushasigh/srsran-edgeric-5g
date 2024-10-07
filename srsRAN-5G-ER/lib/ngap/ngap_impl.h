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

#include "ngap_connection_handler.h"
#include "ngap_context.h"
#include "ngap_error_indication_helper.h"
#include "procedures/ngap_transaction_manager.h"
#include "ue_context/ngap_ue_context.h"
#include "srsran/asn1/ngap/ngap.h"
#include "srsran/ngap/gateways/n2_connection_client.h"
#include "srsran/ngap/ngap.h"
#include "srsran/ngap/ngap_configuration.h"
#include "srsran/ngap/ngap_ue_radio_capability_management.h"
#include "srsran/support/executors/task_executor.h"
#include <memory>

namespace srsran {

namespace srs_cu_cp {

class ngap_impl final : public ngap_interface
{
public:
  ngap_impl(const ngap_configuration&          ngap_cfg_,
            ngap_cu_cp_notifier&               cu_cp_notifier_,
            ngap_cu_cp_du_repository_notifier& cu_cp_du_repository_notifier_,
            n2_connection_client&              n2_gateway,
            timer_manager&                     timers_,
            task_executor&                     ctrl_exec_);
  ~ngap_impl();

  bool
  update_ue_index(ue_index_t new_ue_index, ue_index_t old_ue_index, ngap_cu_cp_ue_notifier& new_ue_notifier) override;

  // ngap connection manager functions
  bool                             handle_amf_tnl_connection_request() override;
  async_task<void>                 handle_amf_disconnection_request() override;
  async_task<ngap_ng_setup_result> handle_ng_setup_request(const ngap_ng_setup_request& request) override;
  async_task<void>                 handle_ng_reset_message(const cu_cp_ng_reset& msg) override;

  // ngap_nas_message_handler
  void handle_initial_ue_message(const cu_cp_initial_ue_message& msg) override;
  void handle_ul_nas_transport_message(const cu_cp_ul_nas_transport& msg) override;

  // ngap_ue_radio_capability_management_handler
  void
  handle_tx_ue_radio_capability_info_indication_required(const ngap_ue_radio_capability_info_indication& msg) override;

  // ngap message handler functions
  void handle_message(const ngap_message& msg) override;
  void handle_connection_loss() override {}

  // ngap_control_message_handler
  async_task<bool> handle_ue_context_release_request(const cu_cp_ue_context_release_request& msg) override;
  async_task<ngap_handover_preparation_response>
       handle_handover_preparation_request(const ngap_handover_preparation_request& msg) override;
  void handle_inter_cu_ho_rrc_recfg_complete(const ue_index_t           ue_index,
                                             const nr_cell_global_id_t& cgi,
                                             const unsigned             tac) override;

  // ngap_statistics_handler
  size_t get_nof_ues() const override { return ue_ctxt_list.size(); }

  // ngap_ue_context_removal_handler
  void remove_ue_context(ue_index_t ue_index) override;

  ngap_message_handler&                        get_ngap_message_handler() override { return *this; }
  ngap_event_handler&                          get_ngap_event_handler() override { return *this; }
  ngap_connection_manager&                     get_ngap_connection_manager() override { return *this; }
  ngap_nas_message_handler&                    get_ngap_nas_message_handler() override { return *this; }
  ngap_ue_radio_capability_management_handler& get_ngap_ue_radio_cap_management_handler() override { return *this; }
  ngap_control_message_handler&                get_ngap_control_message_handler() override { return *this; }
  ngap_ue_control_manager&                     get_ngap_ue_control_manager() override { return *this; }
  ngap_statistics_handler&                     get_ngap_statistics_handler() override { return *this; }
  ngap_ue_context_removal_handler&             get_ngap_ue_context_removal_handler() override { return *this; }

private:
  class tx_pdu_notifier_with_logging final : public ngap_message_notifier
  {
  public:
    tx_pdu_notifier_with_logging(ngap_impl& parent_, std::unique_ptr<ngap_message_notifier> decorated_) :
      parent(parent_), decorated(std::move(decorated_))
    {
    }

    void on_new_message(const ngap_message& msg) override;

  private:
    ngap_impl&                             parent;
    std::unique_ptr<ngap_message_notifier> decorated;
  };

  /// \brief Notify about the reception of an initiating message.
  /// \param[in] msg The received initiating message.
  void handle_initiating_message(const asn1::ngap::init_msg_s& msg);

  /// \brief Notify about the reception of an DL NAS Transport message.
  /// \param[in] msg The received DL NAS Transport message.
  void handle_dl_nas_transport_message(const asn1::ngap::dl_nas_transport_s& msg);

  /// \brief Notify about the reception of an Initial Context Setup Request.
  /// \param[in] msg The received Initial Context Setup Request.
  void handle_initial_context_setup_request(const asn1::ngap::init_context_setup_request_s& request);

  /// \brief Notify about the reception of an PDU Session Resource Setup Request.
  /// \param[in] msg The received PDU Session Resource Setup Request.
  void handle_pdu_session_resource_setup_request(const asn1::ngap::pdu_session_res_setup_request_s& request);

  /// \brief Notify about the reception of an PDU Session Resource Modify Request.
  /// \param[in] msg The received PDU Session Resource Modify Request.
  void handle_pdu_session_resource_modify_request(const asn1::ngap::pdu_session_res_modify_request_s& request);

  /// \brief Notify about the reception of an PDU Session Resource Release Command.
  /// \param[in] msg The received PDU Session Resource Release Command.
  void handle_pdu_session_resource_release_command(const asn1::ngap::pdu_session_res_release_cmd_s& command);

  /// \brief Notify about the reception of an UE Context Release Command.
  /// \param[in] msg The received UE Context Release Command.
  void handle_ue_context_release_command(const asn1::ngap::ue_context_release_cmd_s& cmd);

  /// \brief Notify about the reception of a Paging message.
  /// \param[in] msg The received Paging message.
  void handle_paging(const asn1::ngap::paging_s& msg);

  /// \brief Notify about the reception of an Handover request message.
  /// \param[in] msg The received handover request message.
  void handle_handover_request(const asn1::ngap::ho_request_s& msg);

  /// \brief Notify about the reception of an Error Indication message.
  /// \param[in] msg The received Error Indication message.
  void handle_error_indication(const asn1::ngap::error_ind_s& msg);

  /// \brief Notify about the reception of a successful outcome message.
  /// \param[in] outcome The successful outcome message.
  void handle_successful_outcome(const asn1::ngap::successful_outcome_s& outcome);

  /// \brief Notify about the reception of an unsuccessful outcome message.
  /// \param[in] outcome The unsuccessful outcome message.
  void handle_unsuccessful_outcome(const asn1::ngap::unsuccessful_outcome_s& outcome);

  /// \brief Schedule the transmission of an Error Indication message on the UE task executor.
  /// \param[in] ue_index The index of the related UE.
  /// \param[in] cause The cause of the Error Indication.
  /// \param[in] amf_ue_id The AMF UE ID.
  void schedule_error_indication(ue_index_t ue_index, ngap_cause_t cause, std::optional<amf_ue_id_t> amf_ue_id = {});

  /// \brief Callback for the PDU Session Setup Timer expiration. Triggers the release of the UE.
  void on_pdu_session_setup_timer_expired(ue_index_t ue_index);

  /// \brief Log NGAP RX PDU.
  void log_rx_pdu(const ngap_message& msg);

  ngap_context_t context;

  srslog::basic_logger& logger;

  /// Repository of UE Contexts.
  ngap_ue_context_list ue_ctxt_list;

  std::unordered_map<ue_index_t, error_indication_request_t> stored_error_indications;

  ngap_cu_cp_notifier&               cu_cp_notifier;
  ngap_cu_cp_du_repository_notifier& cu_cp_du_repository_notifier;
  timer_manager&                     timers;
  task_executor&                     ctrl_exec;

  ngap_transaction_manager ev_mng;

  ngap_connection_handler conn_handler;

  std::unique_ptr<tx_pdu_notifier_with_logging> tx_pdu_notifier;
};

} // namespace srs_cu_cp

} // namespace srsran
