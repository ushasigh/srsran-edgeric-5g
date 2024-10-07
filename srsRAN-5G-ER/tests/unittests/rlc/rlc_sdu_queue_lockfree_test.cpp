
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

#include "lib/rlc/rlc_bearer_logger.h"
#include "lib/rlc/rlc_sdu_queue_lockfree.h"
#include "srsran/adt/byte_buffer.h"
#include "srsran/ran/du_types.h"
#include "srsran/support/test_utils.h"

namespace srsran {

void queue_unqueue_test()
{
  rlc_bearer_logger   logger("RLC", {gnb_du_id_t::min, du_ue_index_t::MIN_DU_UE_INDEX, rb_id_t(drb_id_t::drb1), "DL"});
  test_delimit_logger delimiter{"RLC SDU queue unqueue test"};
  rlc_sdu_queue_lockfree tx_queue(4096, logger);

  // Write 1 SDU
  byte_buffer buf       = byte_buffer::create({0x00, 0x01}).value();
  rlc_sdu     write_sdu = {std::move(buf), 10};
  TESTASSERT(tx_queue.write(std::move(write_sdu)));

  // Check basic stats
  rlc_sdu_queue_lockfree::state_t state = tx_queue.get_state();
  TESTASSERT_EQ(1, state.n_sdus);
  TESTASSERT_EQ(2, state.n_bytes);

  // Read one SDU
  rlc_sdu read_sdu;
  TESTASSERT(tx_queue.read(read_sdu));

  // Check basic stats
  state = tx_queue.get_state();
  TESTASSERT_EQ(0, state.n_sdus);
  TESTASSERT_EQ(0, state.n_bytes);

  // Check SDU
  byte_buffer expected_msg = byte_buffer::create({0x00, 0x01}).value();
  TESTASSERT(read_sdu.pdcp_sn.has_value());
  TESTASSERT_EQ(10, read_sdu.pdcp_sn.value());
  TESTASSERT(expected_msg == read_sdu.buf);
}

void full_capacity_test()
{
  rlc_bearer_logger   logger("RLC", {gnb_du_id_t::min, du_ue_index_t::MIN_DU_UE_INDEX, rb_id_t(drb_id_t::drb1), "DL"});
  test_delimit_logger delimiter{"RLC SDU capacity test"};
  unsigned            capacity = 5;
  rlc_sdu_queue_lockfree tx_queue(capacity, logger);

  // Write Capacity + 1 SDUs
  for (uint32_t pdcp_sn = 0; pdcp_sn < capacity + 1; pdcp_sn++) {
    byte_buffer buf = {};
    TESTASSERT(buf.append(pdcp_sn));
    TESTASSERT(buf.append(pdcp_sn));
    rlc_sdu write_sdu = {std::move(buf), pdcp_sn};
    if (pdcp_sn != capacity) {
      TESTASSERT(tx_queue.write(std::move(write_sdu)) == true);
    } else {
      TESTASSERT(tx_queue.write(std::move(write_sdu)) == false);
    }
  }
  rlc_sdu_queue_lockfree::state_t state = tx_queue.get_state();
  TESTASSERT_EQ(capacity, state.n_sdus);
  TESTASSERT_EQ(2 * capacity, state.n_bytes);

  // Read all SDUs and try to read on SDU over capacity
  for (uint32_t pdcp_sn = 0; pdcp_sn < capacity + 1; pdcp_sn++) {
    byte_buffer expected_msg = {};
    TESTASSERT(expected_msg.append(pdcp_sn));
    TESTASSERT(expected_msg.append(pdcp_sn));
    rlc_sdu read_sdu = {};
    if (pdcp_sn != capacity) {
      TESTASSERT(tx_queue.read(read_sdu));
      TESTASSERT(expected_msg == read_sdu.buf);
    } else {
      TESTASSERT(false == tx_queue.read(read_sdu));
    }
  }

  state = tx_queue.get_state();
  TESTASSERT_EQ(0, state.n_sdus);
  TESTASSERT_EQ(0, state.n_bytes);
}

void discard_test()
{
  rlc_bearer_logger   logger("RLC", {gnb_du_id_t::min, du_ue_index_t::MIN_DU_UE_INDEX, rb_id_t(drb_id_t::drb1), "DL"});
  test_delimit_logger delimiter{"RLC SDU discard test"};
  unsigned            capacity = 10;
  unsigned            n_sdus   = capacity;
  rlc_sdu_queue_lockfree tx_queue(capacity, logger);

  // Fill SDU queue with SDUs
  for (uint32_t pdcp_sn = 0; pdcp_sn < n_sdus; pdcp_sn++) {
    byte_buffer buf = {};
    TESTASSERT(buf.append(pdcp_sn));
    TESTASSERT(buf.append(pdcp_sn));
    rlc_sdu write_sdu = {std::move(buf), pdcp_sn};
    TESTASSERT(tx_queue.write(std::move(write_sdu)) == true);
  }
  rlc_sdu_queue_lockfree::state_t state = tx_queue.get_state();
  TESTASSERT_EQ(n_sdus, state.n_sdus);
  TESTASSERT_EQ(2 * n_sdus, state.n_bytes);

  // Discard pdcp_sn 2 and 4
  TESTASSERT(tx_queue.try_discard(2));
  TESTASSERT(tx_queue.try_discard(4));

  // Try to discard non-existing pdcp_sn
  TESTASSERT(false == tx_queue.try_discard(16));

  // Double check correct number of SDUs and SDU bytes
  unsigned leftover_sdus = n_sdus - 2;
  state                  = tx_queue.get_state();
  TESTASSERT_EQ(leftover_sdus, state.n_sdus);
  TESTASSERT_EQ(leftover_sdus * 2, state.n_bytes);

  // Read SDUs
  for (uint32_t n = 0; n < leftover_sdus; n++) {
    rlc_sdu read_sdu = {};
    TESTASSERT(tx_queue.read(read_sdu));
  }
  state = tx_queue.get_state();
  TESTASSERT_EQ(0, state.n_sdus);
  TESTASSERT_EQ(0, state.n_bytes);
}

void discard_all_test()
{
  rlc_bearer_logger   logger("RLC", {gnb_du_id_t::min, du_ue_index_t::MIN_DU_UE_INDEX, rb_id_t(drb_id_t::drb1), "DL"});
  test_delimit_logger delimiter{"RLC SDU discard all test"};
  unsigned            capacity = 10;
  unsigned            n_sdus   = capacity / 2;
  rlc_sdu_queue_lockfree tx_queue(capacity, logger);

  // Fill SDU queue with SDUs
  for (uint32_t pdcp_sn = 0; pdcp_sn < n_sdus; pdcp_sn++) {
    byte_buffer buf = {};
    TESTASSERT(buf.append(pdcp_sn));
    TESTASSERT(buf.append(pdcp_sn));
    rlc_sdu write_sdu = {std::move(buf), pdcp_sn};
    TESTASSERT(tx_queue.write(std::move(write_sdu)) == true);
  }
  rlc_sdu_queue_lockfree::state_t state = tx_queue.get_state();
  TESTASSERT_EQ(n_sdus, state.n_sdus);
  TESTASSERT_EQ(2 * n_sdus, state.n_bytes);

  // Discard all SDUs
  for (uint32_t pdcp_sn = 0; pdcp_sn < n_sdus; pdcp_sn++) {
    TESTASSERT(tx_queue.try_discard(pdcp_sn));
  }

  state = tx_queue.get_state();
  TESTASSERT_EQ(0, state.n_sdus);
  TESTASSERT_EQ(0, state.n_bytes);

  // Read SDU
  {
    rlc_sdu read_sdu = {};
    TESTASSERT(tx_queue.read(read_sdu) == false);
  }
  state = tx_queue.get_state();
  TESTASSERT_EQ(0, state.n_sdus);
  TESTASSERT_EQ(0, state.n_bytes);
}
} // namespace srsran

int main()
{
  srslog::init();
  srslog::fetch_basic_logger("TEST", false).set_level(srslog::basic_levels::debug);
  srslog::fetch_basic_logger("RLC", false).set_level(srslog::basic_levels::debug);
  fprintf(stdout, "Testing RLC SDU queue\n");

  srsran::queue_unqueue_test();
  srsran::full_capacity_test();
  srsran::discard_test();
  srsran::discard_all_test();
}
