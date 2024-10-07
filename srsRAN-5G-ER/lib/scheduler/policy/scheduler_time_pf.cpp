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

#include "scheduler_time_pf.h"
#include "../support/csi_report_helpers.h"

#include <iostream>
#include <fstream>  // For file I/O
#include <string> 
#include <map>  // For std::to_string
#include "../../edgeric/edgeric.h"

// Ushasi
uint32_t cnt = 0;
std::map<uint16_t, uint32_t> pending_data;

using namespace srsran;

scheduler_time_pf::scheduler_time_pf(const scheduler_ue_expert_config& expert_cfg_) :
  fairness_coeff(std::get<time_pf_scheduler_expert_config>(expert_cfg_.strategy_cfg).pf_sched_fairness_coeff)
{
}

void scheduler_time_pf::dl_sched(ue_pdsch_allocator&          pdsch_alloc,
                                 const ue_resource_grid_view& res_grid,
                                 dl_ran_slice_candidate&      slice_candidate)
{ 
  
  // std::ofstream logfile2("ue_metrics_log.txt", std::ios_base::app); // Open log file in append mode
  //       if (logfile2.is_open()) {
  //           logfile2 << "TTI: " << static_cast<unsigned int>(cnt) << " ,from agent: " << edgeric::tti_cnt << std::endl;    
  //           logfile2.close();
  //       }      
  cnt = cnt + 1;
  // Clear the existing contents of the queue.
  dl_queue.clear();

  const slice_ue_repository& ues = slice_candidate.get_slice_ues();
  // Remove deleted users from history.
  for (auto it = ue_history_db.begin(); it != ue_history_db.end();) {
    if (not ues.contains(it->ue_index)) {
      ue_history_db.erase(it->ue_index);
    }
    ++it;
  }

  // Add new users to history db, and update DL priority queue.
  for (const auto& u : ues) {
    const ue_cell* ue_cc = u->find_cell(u->get_pcell().cell_index);
    if (ue_cc != nullptr) {
        float effective_cqi = ue_cc->link_adaptation_controller().get_effective_cqi();
        float effective_snr = ue_cc->channel_state_manager().get_pusch_snr();
        uint32_t dl_newtx_bytes = u->pending_dl_newtx_bytes();
        uint32_t ul_newtx_bytes = u->pending_ul_newtx_bytes();
        edgeric::set_cqi(static_cast<unsigned int>(ue_cc->rnti()), effective_cqi);
        edgeric::set_snr(static_cast<unsigned int>(ue_cc->rnti()), effective_snr);
        edgeric::set_ul_buffer(static_cast<unsigned int>(ue_cc->rnti()), ul_newtx_bytes);
        edgeric::set_dl_buffer(static_cast<unsigned int>(ue_cc->rnti()), dl_newtx_bytes);
    }
    
    if (not ue_history_db.contains(u->ue_index)) {
      // TODO: Consider other cells when carrier aggregation is supported.
      ue_history_db.emplace(u->ue_index, ue_ctxt{u->ue_index, u->get_pcell().cell_index, this});
    }
    ue_ctxt& ctxt = ue_history_db[u->ue_index];
    ctxt.compute_dl_prio(*u);
    dl_queue.push(&ctxt);
  }

  alloc_result alloc_result = {alloc_status::invalid_params};
  unsigned     rem_rbs      = slice_candidate.remaining_rbs();
  

  std::ofstream logfile("scheduling-final.txt", std::ios_base::app); // Open log file in append mode
        if (logfile.is_open()) {
            logfile << "TTI from agent: " << edgeric::tti_cnt << ", Remaining RBs: " << rem_rbs << std::endl;    
            logfile.close();
        }
  while (not dl_queue.empty() and rem_rbs > 0) {  
    ue_ctxt& ue = *dl_queue.top();
    if (alloc_result.status != alloc_status::skip_slot) {
      alloc_result = try_dl_alloc(ue, ues, pdsch_alloc, rem_rbs);
    }
    ue.save_dl_alloc(alloc_result.alloc_bytes);
    // Re-add the UE to the queue if scheduling of re-transmission fails so that scheduling of retransmission are
    // attempted again before scheduling new transmissions.
    if (ue.dl_retx_h != nullptr and alloc_result.status == alloc_status::invalid_params) {
      dl_queue.push(&ue);
    }
    dl_queue.pop();
    rem_rbs = slice_candidate.remaining_rbs();
  }
  
}

void scheduler_time_pf::ul_sched(ue_pusch_allocator&          pusch_alloc,
                                 const ue_resource_grid_view& res_grid,
                                 ul_ran_slice_candidate&      slice_candidate)
{
  // Clear the existing contents of the queue.
  ul_queue.clear();

  const slice_ue_repository& ues = slice_candidate.get_slice_ues();
  // Remove deleted users from history.
  for (auto it = ue_history_db.begin(); it != ue_history_db.end();) {
    if (not ues.contains(it->ue_index)) {
      ue_history_db.erase(it->ue_index);
    }
    ++it;
  }

  // Add new users to history db, and update UL priority queue.
  for (const auto& u : ues) {
    if (not ue_history_db.contains(u->ue_index)) {
      // TODO: Consider other cells when carrier aggregation is supported.
      ue_history_db.emplace(u->ue_index, ue_ctxt{u->ue_index, u->get_pcell().cell_index, this});
    }
    ue_ctxt& ctxt = ue_history_db[u->ue_index];
    ctxt.compute_ul_prio(*u, res_grid);
    ul_queue.push(&ctxt);
  }

  alloc_result alloc_result = {alloc_status::invalid_params};
  unsigned     rem_rbs      = slice_candidate.remaining_rbs();
  while (not ul_queue.empty() and rem_rbs > 0) {
    ue_ctxt& ue = *ul_queue.top();
    if (alloc_result.status != alloc_status::skip_slot) {
      alloc_result = try_ul_alloc(ue, ues, pusch_alloc, rem_rbs);
    }
    ue.save_ul_alloc(alloc_result.alloc_bytes);
    // Re-add the UE to the queue if scheduling of re-transmission fails so that scheduling of retransmission are
    // attempted again before scheduling new transmissions.
    if (ue.ul_retx_h != nullptr and alloc_result.status == alloc_status::invalid_params) {
      ul_queue.push(&ue);
    }
    ul_queue.pop();
    rem_rbs = slice_candidate.remaining_rbs();
  }
}

alloc_result scheduler_time_pf::try_dl_alloc(ue_ctxt&                   ctxt,
                                             const slice_ue_repository& ues,
                                             ue_pdsch_allocator&        pdsch_alloc,
                                             unsigned                   max_rbs)
{
  alloc_result   alloc_result = {alloc_status::invalid_params};
  ue_pdsch_grant grant{ues[ctxt.ue_index], ctxt.cell_index};
  // Prioritize reTx over newTx.
  if (ctxt.dl_retx_h != nullptr) {
    grant.h_id   = ctxt.dl_retx_h->id;
    alloc_result = pdsch_alloc.allocate_dl_grant(grant);
    if (alloc_result.status == alloc_status::success) {
      ctxt.dl_retx_h = nullptr;
    }
    // Return result here irrespective of the outcome so that reTxs of UEs are scheduled before scheduling newTxs of
    // UEs.
    return alloc_result;
  }

  if (ctxt.dl_newtx_h != nullptr) {
    grant.h_id                  = ctxt.dl_newtx_h->id;
    grant.recommended_nof_bytes = ctxt.dl_newtx_srb_pending_bytes > 0 ? ctxt.dl_newtx_srb_pending_bytes
                                                                      : ues[ctxt.ue_index]->pending_dl_newtx_bytes();
    grant.max_nof_rbs           = max_rbs;
    alloc_result                = pdsch_alloc.allocate_dl_grant(grant);
    if (alloc_result.status == alloc_status::success) {
      ctxt.dl_newtx_h = nullptr;
    }
    return alloc_result;
  }

  return {alloc_status::skip_ue};
}

alloc_result scheduler_time_pf::try_ul_alloc(ue_ctxt&                   ctxt,
                                             const slice_ue_repository& ues,
                                             ue_pusch_allocator&        pusch_alloc,
                                             unsigned                   max_rbs)
{
  alloc_result   alloc_result = {alloc_status::invalid_params};
  ue_pusch_grant grant{ues[ctxt.ue_index], ctxt.cell_index};
  // Prioritize reTx over newTx.
  if (ctxt.ul_retx_h != nullptr) {
    grant.h_id   = ctxt.ul_retx_h->id;
    alloc_result = pusch_alloc.allocate_ul_grant(grant);
    if (alloc_result.status == alloc_status::success) {
      ctxt.ul_retx_h = nullptr;
    }
    // Return result here irrespective of the outcome so that reTxs of UEs are scheduled before scheduling newTxs of
    // UEs.
    return alloc_result;
  }

  if (ctxt.ul_newtx_h != nullptr) {
    grant.h_id                  = ctxt.ul_newtx_h->id;
    grant.recommended_nof_bytes = ctxt.ul_newtx_srb_pending_bytes > 0 ? ctxt.ul_newtx_srb_pending_bytes
                                                                      : ues[ctxt.ue_index]->pending_ul_newtx_bytes();
    grant.max_nof_rbs           = max_rbs;
    alloc_result                = pusch_alloc.allocate_ul_grant(grant);
    if (alloc_result.status == alloc_status::success) {
      ctxt.ul_newtx_h = nullptr;
    }
    return alloc_result;
  }

  return {alloc_status::skip_ue};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void scheduler_time_pf::ue_ctxt::compute_dl_prio(const ue& u)
{
  dl_retx_h                  = nullptr;
  dl_newtx_h                 = nullptr;
  dl_prio                    = 0;
  dl_newtx_srb_pending_bytes = 0;
  const ue_cell* ue_cc       = u.find_cell(cell_index);
  if (ue_cc == nullptr or not ue_cc->is_active() or ue_cc->is_in_fallback_mode()) {
    return;
  }

  // Find DL HARQ with transport block containing SRB data to retransmit.
  const dl_harq_process* srb_retx_h = nullptr;
  for (unsigned i = 0; i != ue_cc->harqs.nof_dl_harqs(); ++i) {
    const dl_harq_process& h = ue_cc->harqs.dl_harq(i);
    if (h.has_pending_retx() and not h.last_alloc_params().is_fallback and
        h.last_alloc_params().tb[0]->contains_srb_data) {
      srb_retx_h = &h;
      break;
    }
  }

  // Calculate DL priority.
  dl_retx_h                  = srb_retx_h != nullptr ? srb_retx_h : ue_cc->harqs.find_pending_dl_retx();
  dl_newtx_h                 = ue_cc->harqs.find_empty_dl_harq();
  dl_newtx_srb_pending_bytes = u.pending_dl_srb_newtx_bytes();
  if (dl_retx_h != nullptr or (dl_newtx_h != nullptr and u.has_pending_dl_newtx_bytes())) {
    // NOTE: It does not matter whether it's a reTx or newTx since DL priority is computed based on estimated
    // instantaneous achievable rate to the average throughput of the user.
    // [Implementation-defined] We consider only the SearchSpace defined in UE dedicated configuration.
    const auto* ss_info = ue_cc->cfg().find_search_space(
        ue_cc->cfg().cfg_dedicated().init_dl_bwp.pdcch_cfg->search_spaces.back().get_id());
    if (ss_info == nullptr) {
      return;
    }
    span<const pdsch_time_domain_resource_allocation> pdsch_td_res_list = ss_info->pdsch_time_domain_list;
    // [Implementation-defined] We pick the first element since PDSCH time domain resource list is sorted in descending
    // order of nof. PDSCH symbols. And, we want to calculate estimate of instantaneous achievable rate with maximum
    // nof. PDSCH symbols.
    const pdsch_time_domain_resource_allocation& pdsch_td_cfg = pdsch_td_res_list.front();

    pdsch_config_params pdsch_cfg;
    switch (ss_info->get_dl_dci_format()) {
      case dci_dl_format::f1_0:
        pdsch_cfg = get_pdsch_config_f1_0_c_rnti(ue_cc->cfg().cell_cfg_common, &ue_cc->cfg(), pdsch_td_cfg);
        break;
      case dci_dl_format::f1_1:
        pdsch_cfg = get_pdsch_config_f1_1_c_rnti(
            ue_cc->cfg(), pdsch_td_cfg, ue_cc->channel_state_manager().get_nof_dl_layers());
        break;
      default:
        report_fatal_error("Unsupported PDCCH DCI DL format");
    }

    std::optional<sch_mcs_index> mcs = ue_cc->link_adaptation_controller().calculate_dl_mcs(pdsch_cfg.mcs_table);
    if (not mcs.has_value()) {
      // CQI is either 0, or > 15.
      dl_retx_h  = nullptr;
      dl_newtx_h = nullptr;
      return;
    }

    // Calculate DL PF priority.
    // NOTE: Estimated instantaneous DL rate is calculated assuming entire BWP CRBs are allocated to UE.
    const double estimated_rate   = ue_cc->get_estimated_dl_rate(pdsch_cfg, mcs.value(), ss_info->dl_crb_lims.length());
    const double current_avg_rate = dl_avg_rate();
    if (current_avg_rate != 0) {
      dl_prio = estimated_rate / pow(current_avg_rate, parent->fairness_coeff);
    } else {
      dl_prio = estimated_rate == 0 ? 0 : std::numeric_limits<double>::max();
    }
    // Log the DL priority for the UE
    // std::ofstream logfile("scheduling-final.txt", std::ios_base::app); // Open log file in append mode
    // if (logfile.is_open()) {
    //   logfile << "UE Index: " << static_cast<unsigned int>(u.crnti) << ", DL Priority: " << dl_prio << std::endl;
    //   logfile.close();
    // }
    return;
  }
  dl_newtx_h = nullptr;
}

void scheduler_time_pf::ue_ctxt::compute_ul_prio(const ue& u, const ue_resource_grid_view& res_grid)
{
  ul_retx_h                  = nullptr;
  ul_newtx_h                 = nullptr;
  ul_prio                    = 0;
  sr_ind_received            = false;
  ul_newtx_srb_pending_bytes = 0;
  const ue_cell* ue_cc       = u.find_cell(cell_index);
  if (ue_cc == nullptr or not ue_cc->is_active() or ue_cc->is_in_fallback_mode()) {
    return;
  }

  // Calculate UL priority.
  ul_retx_h                  = ue_cc->harqs.find_pending_ul_retx();
  ul_newtx_h                 = ue_cc->harqs.find_empty_ul_harq();
  sr_ind_received            = u.has_pending_sr();
  ul_newtx_srb_pending_bytes = u.pending_ul_srb_newtx_bytes();
  if (ul_retx_h != nullptr or (ul_newtx_h != nullptr and u.pending_ul_newtx_bytes() > 0)) {
    // NOTE: It does not matter whether it's a reTx or newTx since UL priority is computed based on estimated
    // instantaneous achievable rate to the average throughput of the user.
    // [Implementation-defined] We consider only the SearchSpace defined in UE dedicated configuration.
    const auto* ss_info = ue_cc->cfg().find_search_space(
        ue_cc->cfg().cfg_dedicated().init_dl_bwp.pdcch_cfg->search_spaces.back().get_id());
    if (ss_info == nullptr) {
      return;
    }
    span<const pusch_time_domain_resource_allocation> pusch_td_res_list = ss_info->pusch_time_domain_list;
    // [Implementation-defined] We pick the first element since PUSCH time domain resource list is sorted in descending
    // order of nof. PUSCH symbols. And, we want to calculate estimate of instantaneous achievable rate with maximum
    // nof. PUSCH symbols.
    const pusch_time_domain_resource_allocation& pusch_td_cfg = pusch_td_res_list.front();
    // [Implementation-defined] We assume nof. HARQ ACK bits is zero at PUSCH slot as a simplification in calculating
    // estimated instantaneous achievable rate.
    constexpr unsigned nof_harq_ack_bits  = 0;
    const bool         is_csi_report_slot = csi_helper::is_csi_reporting_slot(
        u.get_pcell().cfg().cfg_dedicated(), res_grid.get_pusch_slot(cell_index, pusch_td_cfg.k2));

    pusch_config_params pusch_cfg;
    switch (ss_info->get_ul_dci_format()) {
      case dci_ul_format::f0_0:
        pusch_cfg = get_pusch_config_f0_0_c_rnti(ue_cc->cfg().cell_cfg_common,
                                                 &ue_cc->cfg(),
                                                 ue_cc->cfg().cell_cfg_common.ul_cfg_common.init_ul_bwp,
                                                 pusch_td_cfg,
                                                 nof_harq_ack_bits,
                                                 is_csi_report_slot);
        break;
      case dci_ul_format::f0_1:
        pusch_cfg = get_pusch_config_f0_1_c_rnti(ue_cc->cfg(),
                                                 pusch_td_cfg,
                                                 ue_cc->channel_state_manager().get_nof_ul_layers(),
                                                 nof_harq_ack_bits,
                                                 is_csi_report_slot);
        break;
      default:
        report_fatal_error("Unsupported PDCCH DCI UL format");
    }

    sch_mcs_index mcs = ue_cc->link_adaptation_controller().calculate_ul_mcs(pusch_cfg.mcs_table);

    // Calculate UL PF priority.
    // NOTE: Estimated instantaneous UL rate is calculated assuming entire BWP CRBs are allocated to UE.
    const double estimated_rate   = ue_cc->get_estimated_ul_rate(pusch_cfg, mcs.value(), ss_info->ul_crb_lims.length());
    const double current_avg_rate = ul_avg_rate();
    if (current_avg_rate != 0) {
      ul_prio = estimated_rate / pow(current_avg_rate, parent->fairness_coeff);
    } else {
      ul_prio = estimated_rate == 0 ? 0 : std::numeric_limits<double>::max();
    }
    return;
  }
  ul_newtx_h = nullptr;
}

void scheduler_time_pf::ue_ctxt::save_dl_alloc(uint32_t alloc_bytes)
{
  if (dl_nof_samples < 1 / parent->exp_avg_alpha) {
    // Fast start before transitioning to exponential average.
    dl_avg_rate_ = dl_avg_rate_ + (alloc_bytes - dl_avg_rate_) / (dl_nof_samples + 1);
  } else {
    dl_avg_rate_ = (1 - parent->exp_avg_alpha) * dl_avg_rate_ + (parent->exp_avg_alpha) * alloc_bytes;
  }
  dl_nof_samples++;
}

void scheduler_time_pf::ue_ctxt::save_ul_alloc(uint32_t alloc_bytes)
{
  if (ul_nof_samples < 1 / parent->exp_avg_alpha) {
    // Fast start before transitioning to exponential average.
    ul_avg_rate_ = ul_avg_rate_ + (alloc_bytes - ul_avg_rate_) / (ul_nof_samples + 1);
  } else {
    ul_avg_rate_ = (1 - parent->exp_avg_alpha) * ul_avg_rate_ + (parent->exp_avg_alpha) * alloc_bytes;
  }
  ul_nof_samples++;
}

bool scheduler_time_pf::ue_dl_prio_compare::operator()(const scheduler_time_pf::ue_ctxt* lhs,
                                                       const scheduler_time_pf::ue_ctxt* rhs) const
{
  const bool is_lhs_retx = lhs->dl_retx_h != nullptr;
  const bool is_rhs_retx = rhs->dl_retx_h != nullptr;
  const bool is_lhs_srb_retx =
      lhs->dl_retx_h != nullptr and (lhs->dl_retx_h->last_alloc_params().tb[0]->contains_srb_data or
                                     (lhs->dl_retx_h->last_alloc_params().tb[1].has_value() and
                                      lhs->dl_retx_h->last_alloc_params().tb[1]->contains_srb_data));
  const bool is_rhs_srb_retx =
      rhs->dl_retx_h != nullptr and (rhs->dl_retx_h->last_alloc_params().tb[0]->contains_srb_data or
                                     (rhs->dl_retx_h->last_alloc_params().tb[1].has_value() and
                                      rhs->dl_retx_h->last_alloc_params().tb[1]->contains_srb_data));

  // First, prioritize UEs with SRB data re-transmissions.
  // SRB HARQ retransmission in one UE and not in other UE.
  if (is_lhs_srb_retx != is_rhs_srb_retx) {
    return is_rhs_srb_retx;
  }
  // SRB HARQ retransmission in both UEs.
  if (is_lhs_srb_retx) {
    return lhs->dl_prio < rhs->dl_prio;
  }
  // Second, prioritize UEs with SRB data new transmission.
  // SRB HARQ newTx in one UE and not in other UE.
  const bool lhs_has_dl_newtx_srb_pending_bytes = lhs->dl_newtx_srb_pending_bytes > 0;
  const bool rhs_has_dl_newtx_srb_pending_bytes = rhs->dl_newtx_srb_pending_bytes > 0;
  if (lhs_has_dl_newtx_srb_pending_bytes != rhs_has_dl_newtx_srb_pending_bytes) {
    return rhs_has_dl_newtx_srb_pending_bytes;
  }
  // SRB HARQ newTx in both UEs.
  if (lhs_has_dl_newtx_srb_pending_bytes) {
    return lhs->dl_prio < rhs->dl_prio;
  }
  // Third, prioritize UEs with DRB re-transmissions.
  // ReTx in one UE and not in other UE.
  if (is_lhs_retx != is_rhs_retx) {
    return is_rhs_retx;
  }
  // All other cases compare priorities.
  return lhs->dl_prio < rhs->dl_prio;
}

bool scheduler_time_pf::ue_ul_prio_compare::operator()(const scheduler_time_pf::ue_ctxt* lhs,
                                                       const scheduler_time_pf::ue_ctxt* rhs) const
{
  const bool is_lhs_retx = lhs->ul_retx_h != nullptr;
  const bool is_rhs_retx = rhs->ul_retx_h != nullptr;
  // First, prioritize UEs with pending SR.
  // SR indication in one UE and not in other UE.
  if (lhs->sr_ind_received != rhs->sr_ind_received) {
    return rhs->sr_ind_received;
  }
  // SR indication for both UEs.
  if (lhs->sr_ind_received) {
    return lhs->ul_prio < rhs->ul_prio;
  }
  // Second, prioritize UEs with SRB data new transmissions.
  // SRB data newTx in one UE and not in other UE.
  const bool lhs_has_ul_newtx_srb_pending_bytes = lhs->ul_newtx_srb_pending_bytes > 0;
  const bool rhs_has_ul_newtx_srb_pending_bytes = rhs->ul_newtx_srb_pending_bytes > 0;
  if (lhs_has_ul_newtx_srb_pending_bytes != rhs_has_ul_newtx_srb_pending_bytes) {
    return rhs_has_ul_newtx_srb_pending_bytes;
  }
  // SRB data newTx in both UEs.
  if (lhs_has_ul_newtx_srb_pending_bytes) {
    return lhs->ul_prio < rhs->ul_prio;
  }
  // Third, prioritize UEs with re-transmissions.
  // ReTx in one UE and not in other UE.
  if (is_lhs_retx != is_rhs_retx) {
    return is_rhs_retx;
  }
  // All other cases compare priorities.
  return lhs->ul_prio < rhs->ul_prio;
}
