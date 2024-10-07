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

#include "iq_compression_none_impl.h"
#include "compressed_prb_packer.h"
#include "compressed_prb_unpacker.h"
#include "quantizer.h"
#include "srsran/srsvec/dot_prod.h"

using namespace srsran;
using namespace ofh;

void iq_compression_none_impl::compress(span<compressed_prb>         output,
                                        span<const cbf16_t>          input,
                                        const ru_compression_params& params)
{
  // Quantizer object.
  quantizer q(params.data_width);

  span<const bf16_t> float_samples(reinterpret_cast<const bf16_t*>(input.data()), input.size() * 2);

  unsigned in_sample_idx = 0;
  for (compressed_prb& c_prb : output) {
    // Auxiliary buffer used for float to int16_t conversion.
    std::array<int16_t, NOF_SUBCARRIERS_PER_RB * 2> conv_buffer;

    // Scale input IQ data to the range [-1: +1) and convert it to int16_t.
    q.to_fixed_point(conv_buffer, float_samples.subspan(in_sample_idx, NOF_SUBCARRIERS_PER_RB * 2), iq_scaling);

    compressed_prb_packer packer(c_prb);
    packer.pack(conv_buffer, params.data_width);

    in_sample_idx += (NOF_SUBCARRIERS_PER_RB * 2);
  }
}

void iq_compression_none_impl::decompress(span<cbf16_t>                output,
                                          span<const compressed_prb>   input,
                                          const ru_compression_params& params)
{
  // Quantizer object.
  quantizer q(params.data_width);

  unsigned out_idx = 0;
  for (const auto& c_prb : input) {
    compressed_prb_unpacker unpacker(c_prb);
    for (unsigned i = 0, read_pos = 0; i != NOF_SUBCARRIERS_PER_RB; ++i) {
      int16_t re = q.sign_extend(unpacker.unpack(read_pos, params.data_width));
      int16_t im = q.sign_extend(unpacker.unpack(read_pos + params.data_width, params.data_width));
      read_pos += (params.data_width * 2);
      output[out_idx++] = {q.to_float(re), q.to_float(im)};
    }
  }
}

void iq_compression_none_impl::log_post_quantization_rms(span<const int16_t> samples)
{
  if (SRSRAN_UNLIKELY(logger.debug.enabled() && !samples.empty())) {
    // Calculate and print RMS of quantized samples.
    float sum_squares = srsvec::dot_prod(samples, samples, 0);
    float rms         = std::sqrt(sum_squares / samples.size());
    if (std::isnormal(rms)) {
      logger.debug("Quantized IQ samples RMS value of '{}'", rms);
    }
  }
}
