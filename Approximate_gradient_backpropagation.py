

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1) Configuration
# ==============================================================================
MEMORY_BITS_PER_ROUTE = 4

APPROX_GRAD_ENABLE = True
APPROX_GRAD_TEMPERATURE = 1.0
APPROX_GRAD_SCALE = 1.0
APPROX_GRAD_ROUTE_CHUNK_SIZE = 256
APPROX_GRAD_ACCUM_FP32 = True


# ==============================================================================
# 2) Helper: pack binary bits into per-route integer codes
# ==============================================================================
def pack_binary_bits_to_codes(bits_btc: torch.Tensor, bits_per_route: int) -> torch.Tensor:
    """
    Convert binary bits into exact per-route integer codes.

    Args:
        bits_btc:
            Tensor of shape [B, T, C] with 0/1 values.

        bits_per_route:
            Number of binary bits assigned to each route.

    Returns:
        codes_btr:
            Tensor of shape [B, T, R], where each value is the exact
            integer code for one route.
    """
    batch_size, seq_len, channels = bits_btc.shape
    if channels % bits_per_route != 0:
        raise ValueError(
            f"channels={channels} must be divisible by bits_per_route={bits_per_route}"
        )

    num_routes = channels // bits_per_route
    x = bits_btc.view(batch_size, seq_len, num_routes, bits_per_route).to(torch.int32)

    out = torch.zeros((batch_size, seq_len, num_routes), dtype=torch.int32, device=bits_btc.device)
    for bit_idx in range(bits_per_route):
        out |= ((x[..., bit_idx] & 1) << bit_idx)
    return out


# ==============================================================================
# 3) Full flat-memory surrogate backward
# ==============================================================================
class ApproximateGradientSurrogateFunction(torch.autograd.Function):
    """
    Forward:
        Identity on `flat_memory`.

    Backward:
        1) Passes `flat_memory` gradients through unchanged.
        2) Adds a surrogate gradient to `route_logits` using exact
           bit-flip counterfactual lookups on the memory tables.

    Expected shapes:
        flat_memory:   [B, T, N * R * Dm]
        route_logits:  [B, T, C]
        route_codes:   [B, T, R]
    """

    @staticmethod
    def forward(
        ctx,
        flat_memory: torch.Tensor,
        route_logits: torch.Tensor,
        route_codes_btr: torch.Tensor,
        memory_module: nn.Module,
    ):
        ctx.save_for_backward(route_logits, route_codes_btr.to(torch.int16))
        ctx.memory_module = memory_module
        return flat_memory

    @staticmethod
    def backward(ctx, grad_flat_memory: torch.Tensor):
        (route_logits, route_codes_saved) = ctx.saved_tensors
        memory = ctx.memory_module

        if (grad_flat_memory is None) or (not APPROX_GRAD_ENABLE):
            return grad_flat_memory, None, None, None

        batch_size, seq_len, channels = route_logits.shape
        bits_per_route = int(memory.bits_per_route)
        num_routes = int(memory.num_routes)
        mem_dim = int(memory.mem_dim)
        alphabet_size = int(memory.alphabet_size)
        num_orders = int(len(memory.ngrams))

        if channels != num_routes * bits_per_route:
            raise ValueError(
                f"Shape mismatch: channels={channels}, "
                f"num_routes={num_routes}, bits_per_route={bits_per_route}"
            )

        device = route_logits.device
        accum_dtype = torch.float32 if APPROX_GRAD_ACCUM_FP32 else grad_flat_memory.dtype

        temperature = float(APPROX_GRAD_TEMPERATURE)
        grad_scale = float(APPROX_GRAD_SCALE)
        route_chunk_size = max(1, int(APPROX_GRAD_ROUTE_CHUNK_SIZE))

        grad_flat = grad_flat_memory.reshape(
            batch_size, seq_len, num_orders, num_routes, mem_dim
        ).to(accum_dtype)
        route_logits_btrm = route_logits.reshape(batch_size, seq_len, num_routes, bits_per_route)
        route_codes_btr = route_codes_saved.to(torch.long)

        grad_route_logits = torch.zeros(
            (batch_size, seq_len, num_routes, bits_per_route),
            device=device,
            dtype=accum_dtype,
        )
        full_route_ids = torch.arange(num_routes, device=device, dtype=torch.long)

        for route_start in range(0, num_routes, route_chunk_size):
            route_end = min(num_routes, route_start + route_chunk_size)
            route_count = route_end - route_start

            codes_chunk = route_codes_btr[:, :, route_start:route_end]  # [B, T, Rc]
            score_chunk = torch.zeros(
                (batch_size, seq_len, route_count, bits_per_route),
                device=device,
                dtype=accum_dtype,
            )

            for order_idx, ngram_order in enumerate(memory.ngrams):
                ngram_order = int(ngram_order)
                valid_len = seq_len - ngram_order + 1
                if valid_len <= 0:
                    continue

                table_weight = memory.tables[f"ngram_{ngram_order}"].weight.detach()
                route_vocab = alphabet_size ** ngram_order
                route_offsets = full_route_ids[route_start:route_end].view(1, 1, route_count) * route_vocab

                address = torch.zeros(
                    (batch_size, valid_len, route_count),
                    device=device,
                    dtype=torch.long,
                )
                stride = 1
                for step_idx in range(ngram_order):
                    address = address + codes_chunk[:, step_idx:step_idx + valid_len, :] * stride
                    stride *= alphabet_size

                grad_memory_valid = grad_flat[
                    :, ngram_order - 1 :, order_idx, route_start:route_end, :
                ]  # [B, L, Rc, Dm]

                for offset_in_ngram in range(ngram_order):
                    time_start = ngram_order - 1 - offset_in_ngram
                    time_end = time_start + valid_len
                    current_codes = codes_chunk[:, time_start:time_end, :]
                    exponent = ngram_order - 1 - offset_in_ngram

                    for bit_idx in range(bits_per_route):
                        delta = int((1 << bit_idx) * (alphabet_size ** exponent))

                        current_bit = ((current_codes >> bit_idx) & 1)
                        address_zero = address - current_bit * delta
                        address_one = address_zero + delta

                        index_zero = route_offsets + address_zero
                        index_one = route_offsets + address_one

                        value_zero = F.embedding(index_zero, table_weight)
                        value_one = F.embedding(index_one, table_weight)

                        if value_zero.dtype != accum_dtype:
                            value_zero = value_zero.to(accum_dtype)
                            value_one = value_one.to(accum_dtype)

                        diff = value_one - value_zero
                        score_chunk[:, time_start:time_end, :, bit_idx] += (
                            grad_memory_valid * diff
                        ).sum(dim=-1)

            logits_chunk = route_logits_btrm[:, :, route_start:route_end, :].to(accum_dtype)
            prob = torch.sigmoid(temperature * logits_chunk)
            slope = temperature * prob * (1.0 - prob)
            grad_route_logits[:, :, route_start:route_end, :] = grad_scale * slope * score_chunk

        grad_route_logits = grad_route_logits.reshape(batch_size, seq_len, channels).to(route_logits.dtype)

        return grad_flat_memory, grad_route_logits, None, None


# ==============================================================================
# 4) Chunk-level surrogate backward for streaming / route-chunk execution
# ==============================================================================
class ApproximateGradientChunkSurrogateFunction(torch.autograd.Function):
    """
    Forward:
        Identity on `memory_chunk`.

    Backward:
        1) Passes `memory_chunk` gradients through unchanged.
        2) Computes surrogate gradients only for the current
           (ngram_order, route_chunk) slice of `route_logits_chunk`.

    This allows autograd to accumulate surrogate gradients from multiple
    chunks back into the full `route_logits` tensor automatically.

    Expected shapes:
        memory_chunk:        [B, T, Rc, Dm]
        route_logits_chunk:  [B, T, Rc, M]
        route_codes_chunk:   [B, T, Rc]
    """

    @staticmethod
    def forward(
        ctx,
        memory_chunk: torch.Tensor,
        route_logits_chunk: torch.Tensor,
        route_codes_chunk: torch.Tensor,
        memory_module: nn.Module,
        ngram_order: int,
        route_offset: int,
    ):
        ctx.save_for_backward(route_logits_chunk, route_codes_chunk.to(torch.int16))
        ctx.memory_module = memory_module
        ctx.ngram_order = int(ngram_order)
        ctx.route_offset = int(route_offset)
        return memory_chunk

    @staticmethod
    def backward(ctx, grad_memory_chunk: torch.Tensor):
        route_logits_chunk, route_codes_saved = ctx.saved_tensors
        memory = ctx.memory_module
        ngram_order = int(ctx.ngram_order)
        route_offset = int(ctx.route_offset)

        if (grad_memory_chunk is None) or (not APPROX_GRAD_ENABLE):
            return grad_memory_chunk, None, None, None, None, None

        batch_size, seq_len, route_count, bits_per_route = route_logits_chunk.shape
        alphabet_size = int(memory.alphabet_size)

        device = route_logits_chunk.device
        accum_dtype = torch.float32 if APPROX_GRAD_ACCUM_FP32 else grad_memory_chunk.dtype

        temperature = float(APPROX_GRAD_TEMPERATURE)
        grad_scale = float(APPROX_GRAD_SCALE)

        route_codes_chunk = route_codes_saved.to(torch.long)
        score_chunk = torch.zeros(
            (batch_size, seq_len, route_count, bits_per_route),
            device=device,
            dtype=accum_dtype,
        )

        valid_len = seq_len - ngram_order + 1
        if valid_len > 0:
            table_weight = memory.tables[f"ngram_{ngram_order}"].weight.detach()
            route_vocab = alphabet_size ** ngram_order

            route_ids = torch.arange(
                route_offset,
                route_offset + route_count,
                device=device,
                dtype=torch.long,
            ).view(1, 1, route_count)
            route_offsets = route_ids * route_vocab

            address = torch.zeros(
                (batch_size, valid_len, route_count),
                device=device,
                dtype=torch.long,
            )
            stride = 1
            for step_idx in range(ngram_order):
                address = address + route_codes_chunk[:, step_idx:step_idx + valid_len, :] * stride
                stride *= alphabet_size

            grad_valid = grad_memory_chunk[:, ngram_order - 1 :, :, :].to(accum_dtype)

            for offset_in_ngram in range(ngram_order):
                time_start = ngram_order - 1 - offset_in_ngram
                time_end = time_start + valid_len
                current_codes = route_codes_chunk[:, time_start:time_end, :]
                exponent = ngram_order - 1 - offset_in_ngram

                for bit_idx in range(bits_per_route):
                    delta = int((1 << bit_idx) * (alphabet_size ** exponent))

                    current_bit = ((current_codes >> bit_idx) & 1)
                    address_zero = address - current_bit * delta
                    address_one = address_zero + delta

                    index_zero = route_offsets + address_zero
                    index_one = route_offsets + address_one

                    value_zero = F.embedding(index_zero, table_weight)
                    value_one = F.embedding(index_one, table_weight)

                    if value_zero.dtype != accum_dtype:
                        value_zero = value_zero.to(accum_dtype)
                        value_one = value_one.to(accum_dtype)

                    diff = value_one - value_zero
                    score_chunk[:, time_start:time_end, :, bit_idx] += (
                        grad_valid * diff
                    ).sum(dim=-1)

        logits_chunk = route_logits_chunk.to(accum_dtype)
        prob = torch.sigmoid(temperature * logits_chunk)
        slope = temperature * prob * (1.0 - prob)
        grad_logits_chunk = (grad_scale * slope * score_chunk).to(route_logits_chunk.dtype)

        return grad_memory_chunk, grad_logits_chunk, None, None, None, None
