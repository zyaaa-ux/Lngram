from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

def _tensor_assert(condition: Tensor, message: str) -> None:
    """Assert a scalar tensor without forcing a host/device synchronization."""
    if hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    else:  # pragma: no cover - compatibility with older PyTorch releases.
        torch._assert(condition, message)


@torch.no_grad()
def _normal_init(weight: Tensor, std: float) -> None:
    """Initialize in FP32, then cast, matching the reference implementation."""
    if std < 0.0:
        raise ValueError(f"initialization std must be non-negative, got {std}")
    if std == 0.0:
        weight.zero_()
        return
    if weight.dtype == torch.float32:
        nn.init.normal_(weight, mean=0.0, std=std)
        return
    temporary = torch.empty_like(weight, dtype=torch.float32)
    nn.init.normal_(temporary, mean=0.0, std=std)
    weight.copy_(temporary.to(weight.dtype))


class RMSNorm(nn.Module):
    """RMSNorm with an FP32 fallback for older PyTorch versions."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(
                hidden_states, (self.hidden_size,), self.weight, self.eps
            )
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return values.to(input_dtype) * self.weight


def build_lngram_norm(
    normalization: str,
    hidden_size: int,
    eps: float,
    device=None,
    dtype=None,
) -> nn.Module:
    """Build the same normalization family as the base decoder."""
    normalized_name = str(normalization).lower().replace("_", "")
    if normalized_name == "rmsnorm":
        return RMSNorm(hidden_size, eps=eps, device=device, dtype=dtype)
    if normalized_name == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=dtype)
    raise ValueError(
        f"Unsupported LNGram normalization {normalization!r}; expected RMSNorm or LayerNorm"
    )


def pack_bits_to_route_codes(bits: Tensor, bits_per_route: int) -> Tensor:
    """Pack [B,T,R*M] binary channels into exact int32 route codes [B,T,R]."""
    if bits.dim() != 3:
        raise ValueError(f"bits must be [B,T,C], got {tuple(bits.shape)}")
    batch, time, channels = bits.shape
    bits_per_route = int(bits_per_route)
    if bits_per_route <= 0 or bits_per_route > 30 or channels % bits_per_route:
        raise ValueError(
            f"channels={channels} must be divisible by bits_per_route in [1,30], "
            f"got {bits_per_route}"
        )
    routes = channels // bits_per_route
    grouped = bits.reshape(batch, time, routes, bits_per_route).to(torch.int32)
    positions = torch.arange(bits_per_route, device=bits.device, dtype=torch.int32)
    weights = torch.bitwise_left_shift(torch.ones_like(positions), positions)
    return torch.sum(grouped * weights.view(1, 1, 1, -1), dim=-1, dtype=torch.int32)


def route_code_storage_dtype(bits_per_route: int) -> torch.dtype:
    """Return the smallest signed/unsigned dtype that preserves every route code."""
    bits_per_route = int(bits_per_route)
    if bits_per_route <= 0 or bits_per_route > 16:
        raise ValueError(f"bits_per_route must be in [1, 16], got {bits_per_route}")
    if bits_per_route <= 8:
        return torch.uint8
    if bits_per_route <= 15:
        return torch.int16
    return torch.int32


def segment_ids_from_valid_tokens(valid_tokens: Tensor) -> Tuple[Tensor, Tensor]:
    """Build segment ids that reset n-gram/conv history across invalid runs.

    This is primarily used by Hugging Face non-cached evaluation/training with
    left- or right-padded batches.  A transition from invalid to valid starts a
    new segment, so the first real token never forms an n-gram with pad routes.
    """
    if valid_tokens.dim() != 2:
        raise ValueError(
            f"valid_tokens must be [B,T], got {tuple(valid_tokens.shape)}"
        )
    valid = valid_tokens.to(dtype=torch.bool)
    transitions = torch.ones_like(valid, dtype=torch.long)
    if valid.shape[1] > 1:
        transitions[:, 1:] = (valid[:, 1:] != valid[:, :-1]).to(torch.long)
    segment_ids = transitions.cumsum(dim=1) - 1
    return segment_ids, valid


def segment_ids_from_cu_seqlens(
    cu_seqlens: Optional[Tensor],
    batch: int,
    time: int,
    device: torch.device,
    valid_token_count: Optional[Tensor] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Create GPU-only packed-sequence ids and a valid-token mask.

    A THD-style packed representation may flatten all packed segments into the
    sequence dimension and therefore use batch size one.
    Refusing other shapes is safer than silently forming cross-sample n-grams.

    Some packed-data pipelines may include a final *padding segment* so
    that its last element equals the fixed sequence length.  In that case
    ``cu_seqlens[-1]`` is not the number of real tokens.  ``valid_token_count``
    accepts a scalar valid-token count supplied by the caller and prevents
    padding positions from participating in table
    lookup, surrogate gradients, route readout, or the causal convolution.
    A full [B,T] boolean/0-1 mask is also accepted for focused tests and callers.
    """
    positions = torch.arange(time, device=device, dtype=torch.long)

    def supplied_valid_mask() -> Optional[Tensor]:
        if valid_token_count is None:
            return None
        supplied = torch.as_tensor(valid_token_count, device=device)
        if supplied.numel() == 1:
            limit = supplied.to(dtype=torch.long).reshape(()).clamp(min=0, max=time)
            return (positions < limit).unsqueeze(0).expand(batch, -1)
        if supplied.numel() == batch * time:
            return supplied.reshape(batch, time).to(dtype=torch.bool)
        raise ValueError(
            "valid_token_count must be a scalar count or a [batch,time] mask; "
            f"got shape={tuple(supplied.shape)} for batch={batch}, time={time}"
        )

    explicit_valid = supplied_valid_mask()
    if cu_seqlens is None:
        return None, explicit_valid
    if batch != 1:
        raise ValueError(
            "LNGram packed-sequence support expects THD layout with batch size 1; "
            f"got batch={batch}"
        )
    cu = cu_seqlens.to(device=device, dtype=torch.long).reshape(-1)
    if cu.numel() < 2:
        raise ValueError("cu_seqlens must contain at least [0, sequence_length]")
    _tensor_assert(cu[0] == 0, "cu_seqlens must start at 0")
    _tensor_assert(
        torch.all(cu[1:] >= cu[:-1]),
        "cu_seqlens must be monotonically non-decreasing",
    )
    _tensor_assert(
        (cu[-1] >= 0) & (cu[-1] <= time),
        f"cu_seqlens final extent must be in [0, {time}]",
    )
    segment_ids = torch.bucketize(positions, cu[1:-1], right=True).unsqueeze(0)
    packed_extent = (positions < cu[-1]).unsqueeze(0)
    valid_tokens = packed_extent if explicit_valid is None else explicit_valid & packed_extent
    return segment_ids, valid_tokens


@dataclass
class LngramIncrementalState:
    """Side state that must travel with an attention KV cache."""

    route_history: Optional[Tensor] = None
    packed_route_state: Optional[Tensor] = None
    route_state_length: int = 0
    conv_history: Optional[Tensor] = None

    def index_select(self, indices: Tensor) -> "LngramIncrementalState":
        def select(value: Optional[Tensor]) -> Optional[Tensor]:
            return None if value is None else value.index_select(0, indices.to(value.device))

        return LngramIncrementalState(
            route_history=select(self.route_history),
            packed_route_state=select(self.packed_route_state),
            route_state_length=int(self.route_state_length),
            conv_history=select(self.conv_history),
        )


class DepthwiseCausalConv(nn.Module):
    """Zero-left-padded depthwise convolution with exact cached execution."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        dilation: int,
        eps: float,
        normalization: str = "RMSNorm",
        bias: bool = False,
        zero_init: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        self.norm = build_lngram_norm(
            normalization, hidden_size, eps, device=device, dtype=dtype
        )
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.act = nn.SiLU()
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def _segmented_forward(self, values: Tensor, segment_ids: Tensor) -> Tensor:
        """Depthwise convolution that resets at every packed-document boundary."""
        batch, time, channels = values.shape
        output = values.new_zeros(batch, time, channels)
        # Conv1d stores taps oldest -> newest.  This vectorized shift avoids a
        # host-side loop over packed segments and keeps boundary checks on GPU.
        for tap in range(self.kernel_size):
            lag = (self.kernel_size - 1 - tap) * self.dilation
            if lag == 0:
                shifted = values
                valid = torch.ones(
                    batch, time, 1, dtype=torch.bool, device=values.device
                )
            else:
                shifted = F.pad(values[:, :-lag, :], (0, 0, lag, 0))
                valid = torch.zeros(
                    batch, time, 1, dtype=torch.bool, device=values.device
                )
                valid[:, lag:, :] = (
                    segment_ids[:, lag:] == segment_ids[:, :-lag]
                ).unsqueeze(-1)
            weight = self.conv.weight[:, 0, tap].view(1, 1, channels)
            output = output + shifted * valid.to(shifted.dtype) * weight
        if self.conv.bias is not None:
            output = output + self.conv.bias.view(1, 1, channels)
        return self.act(output)

    def forward(self, values: Tensor, segment_ids: Optional[Tensor] = None) -> Tensor:
        if values.dim() != 3 or values.shape[-1] != self.hidden_size:
            raise ValueError(
                f"causal conv expects [B,T,{self.hidden_size}], got {tuple(values.shape)}"
            )
        normalized = self.norm(values)
        if segment_ids is not None:
            if tuple(segment_ids.shape) != tuple(values.shape[:2]):
                raise ValueError("segment_ids must match the [B,T] dimensions")
            return self._segmented_forward(normalized, segment_ids)
        conv_input = normalized.transpose(1, 2)
        if self.left_padding:
            conv_input = F.pad(conv_input, (self.left_padding, 0))
        return self.act(self.conv(conv_input)).transpose(1, 2)

    def forward_incremental(
        self, values: Tensor, history: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if values.dim() != 3 or values.shape[-1] != self.hidden_size:
            raise ValueError("incremental causal conv expects [B,T,H]")
        normalized = self.norm(values).transpose(1, 2)
        batch, channels, _ = normalized.shape
        if self.left_padding == 0:
            window = normalized
            new_history = normalized[:, :, :0]
        else:
            if history is None:
                history = normalized.new_zeros(batch, channels, self.left_padding)
            expected = (batch, channels, self.left_padding)
            if tuple(history.shape) != expected:
                raise ValueError(
                    f"conv history shape={tuple(history.shape)}, expected {expected}"
                )
            window = torch.cat((history.to(normalized), normalized), dim=-1)
            new_history = window[:, :, -self.left_padding :].contiguous()
        output = self.act(self.conv(window)).transpose(1, 2)
        if output.shape[1] != values.shape[1]:
            raise RuntimeError("incremental convolution returned the wrong sequence length")
        return output, new_history


def _aligned_ngram_address(codes: Tensor, order: int, bits: int) -> Tensor:
    """Return valid-window addresses [B,T-order+1,R], oldest digit first."""
    batch, time, routes = codes.shape
    valid_length = time - order + 1
    if valid_length <= 0:
        return codes.new_empty(batch, 0, routes)
    address = codes[:, :valid_length, :].clone()
    for exp in range(1, order):
        address.bitwise_or_(
            torch.bitwise_left_shift(codes[:, exp : exp + valid_length, :], exp * bits)
        )
    return address


class _LngramSurrogate(torch.autograd.Function):
    """Identity forward with exact or one-bit-counterfactual route gradients."""

    @staticmethod
    def forward(
        ctx,
        hard_memory: Tensor,
        q_logits: Tensor,
        route_codes: Tensor,
        table_weight: Tensor,
        valid_mask: Tensor,
        order: int,
        bits: int,
        route_start: int,
        alphabet_size: int,
        temperature: float,
        scale: float,
        exact: bool,
    ) -> Tensor:
        ctx.save_for_backward(
            q_logits,
            route_codes,
            table_weight,
            valid_mask,
            hard_memory.detach(),
        )
        ctx.order = int(order)
        ctx.bits = int(bits)
        ctx.route_start = int(route_start)
        ctx.alphabet_size = int(alphabet_size)
        ctx.temperature = float(temperature)
        ctx.scale = float(scale)
        ctx.exact = bool(exact)
        return hard_memory

    @staticmethod
    def backward(ctx, grad_memory: Optional[Tensor]):
        if grad_memory is None:
            return (None,) * 12
        q_logits, codes, table, valid_mask, hard_memory = ctx.saved_tensors
        order, bits = ctx.order, ctx.bits
        batch, time, routes, _ = q_logits.shape
        valid_length = time - order + 1
        grad_q = torch.zeros_like(q_logits, dtype=torch.float32)
        if valid_length > 0 and ctx.temperature != 0.0 and ctx.scale != 0.0:
            codes_index = codes.to(torch.long)
            address = _aligned_ngram_address(codes_index, order, bits)
            offsets = (
                torch.arange(
                    ctx.route_start,
                    ctx.route_start + routes,
                    device=codes.device,
                    dtype=torch.long,
                )
                * (ctx.alphabet_size ** order)
            ).view(1, 1, routes)
            indices = offsets + address
            aligned_valid = valid_mask[:, order - 1 :].unsqueeze(-1).unsqueeze(-1)
            grad_valid = (
                grad_memory[:, order - 1 :, :, :].float()
                * aligned_valid.to(torch.float32)
            )
            table_detached = table.detach()
            current_values = hard_memory[:, order - 1 :, :, :].float()
            current_dot = (current_values * grad_valid).sum(-1)
            probabilities = torch.sigmoid(ctx.temperature * q_logits.float())

            if ctx.exact:
                code_ids = torch.arange(
                    ctx.alphabet_size, device=codes.device, dtype=torch.long
                )
                code_bits = (
                    torch.bitwise_and(
                        torch.bitwise_right_shift(
                            code_ids[:, None],
                            torch.arange(bits, device=codes.device, dtype=torch.long)[None, :],
                        ),
                        1,
                    )
                    .bool()
                )
                for exp in range(order):
                    source_codes = codes_index[:, exp : exp + valid_length, :]
                    source_p = probabilities[:, exp : exp + valid_length, :, :]
                    base = indices - source_codes * (ctx.alphabet_size ** exp)
                    for code in range(ctx.alphabet_size):
                        row = base + code * (ctx.alphabet_size ** exp)
                        values = F.embedding(row, table_detached).float()
                        dot = (values * grad_valid).sum(-1)
                        bits_for_code = code_bits[code].view(1, 1, 1, bits)
                        code_probability = torch.where(
                            bits_for_code, source_p, 1.0 - source_p
                        ).prod(-1)
                        local = (
                            code_probability.unsqueeze(-1)
                            * (bits_for_code.to(source_p.dtype) - source_p)
                            * dot.unsqueeze(-1)
                        )
                        grad_q[:, exp : exp + valid_length, :, :].add_(
                            local,
                            alpha=ctx.scale * ctx.temperature,
                        )
            else:
                for exp in range(order):
                    for bit in range(bits):
                        mask = 1 << (exp * bits + bit)
                        flipped = torch.bitwise_xor(indices, mask)
                        flipped_values = F.embedding(flipped, table_detached).float()
                        flipped_dot = (flipped_values * grad_valid).sum(-1)
                        bit_is_one = torch.bitwise_and(indices, mask).ne(0)
                        sign = 1.0 - 2.0 * bit_is_one.to(torch.float32)
                        score = (flipped_dot - current_dot) * sign
                        grad_q[:, exp : exp + valid_length, :, bit].add_(score)
                slope = (
                    ctx.scale
                    * ctx.temperature
                    * probabilities
                    * (1.0 - probabilities)
                )
                grad_q.mul_(slope)

        # The first return preserves the ordinary hard-table gradient.  The
        # table argument itself is detached for counterfactual calculations.
        return (
            grad_memory,
            grad_q.to(q_logits.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _window_ngram_address(codes: Tensor, bits: int) -> Tensor:
    """Pack arbitrary-position n-gram windows into table addresses.

    ``codes`` is ``[B,T,O,R]`` and stores each window oldest-to-newest.  Unlike
    :func:`_aligned_ngram_address`, the time axis here is a set of CP-local
    query positions rather than one contiguous sequence.
    """
    if codes.dim() != 4:
        raise ValueError(f"window codes must be [B,T,O,R], got {tuple(codes.shape)}")
    address = codes[:, :, 0, :].clone()
    for exp in range(1, codes.shape[2]):
        address.bitwise_or_(
            torch.bitwise_left_shift(codes[:, :, exp, :], exp * int(bits))
        )
    return address


class _LngramWindowSurrogate(torch.autograd.Function):
    """LNGram route surrogate for non-contiguous CP-local target positions."""

    @staticmethod
    def forward(
        ctx,
        hard_memory: Tensor,
        q_windows: Tensor,
        code_windows: Tensor,
        table_weight: Tensor,
        valid_mask: Tensor,
        bits: int,
        route_start: int,
        alphabet_size: int,
        temperature: float,
        scale: float,
        exact: bool,
    ) -> Tensor:
        ctx.save_for_backward(
            q_windows,
            code_windows,
            table_weight,
            valid_mask,
            hard_memory.detach(),
        )
        ctx.bits = int(bits)
        ctx.route_start = int(route_start)
        ctx.alphabet_size = int(alphabet_size)
        ctx.temperature = float(temperature)
        ctx.scale = float(scale)
        ctx.exact = bool(exact)
        return hard_memory

    @staticmethod
    def backward(ctx, grad_memory: Optional[Tensor]):
        if grad_memory is None:
            return (None,) * 11
        q_windows, codes, table, valid_mask, hard_memory = ctx.saved_tensors
        bits = ctx.bits
        batch, time, order, routes, _ = q_windows.shape
        grad_q = torch.zeros_like(q_windows, dtype=torch.float32)
        if time > 0 and ctx.temperature != 0.0 and ctx.scale != 0.0:
            codes_index = codes.to(torch.long)
            address = _window_ngram_address(codes_index, bits)
            offsets = (
                torch.arange(
                    ctx.route_start,
                    ctx.route_start + routes,
                    device=codes.device,
                    dtype=torch.long,
                )
                * (ctx.alphabet_size ** order)
            ).view(1, 1, routes)
            indices = offsets + address
            grad_valid = grad_memory.float() * valid_mask.unsqueeze(-1).unsqueeze(-1)
            table_detached = table.detach()
            current_dot = (hard_memory.float() * grad_valid).sum(-1)
            probabilities = torch.sigmoid(ctx.temperature * q_windows.float())

            if ctx.exact:
                code_ids = torch.arange(
                    ctx.alphabet_size, device=codes.device, dtype=torch.long
                )
                code_bits = (
                    torch.bitwise_and(
                        torch.bitwise_right_shift(
                            code_ids[:, None],
                            torch.arange(bits, device=codes.device, dtype=torch.long)[None, :],
                        ),
                        1,
                    ).bool()
                )
                for exp in range(order):
                    source_codes = codes_index[:, :, exp, :]
                    source_p = probabilities[:, :, exp, :, :]
                    base = indices - source_codes * (ctx.alphabet_size ** exp)
                    for code in range(ctx.alphabet_size):
                        row = base + code * (ctx.alphabet_size ** exp)
                        values = F.embedding(row, table_detached).float()
                        dot = (values * grad_valid).sum(-1)
                        bits_for_code = code_bits[code].view(1, 1, 1, bits)
                        code_probability = torch.where(
                            bits_for_code, source_p, 1.0 - source_p
                        ).prod(-1)
                        local = (
                            code_probability.unsqueeze(-1)
                            * (bits_for_code.to(source_p.dtype) - source_p)
                            * dot.unsqueeze(-1)
                        )
                        grad_q[:, :, exp, :, :].add_(
                            local,
                            alpha=ctx.scale * ctx.temperature,
                        )
            else:
                for exp in range(order):
                    for bit in range(bits):
                        mask = 1 << (exp * bits + bit)
                        flipped = torch.bitwise_xor(indices, mask)
                        flipped_values = F.embedding(flipped, table_detached).float()
                        flipped_dot = (flipped_values * grad_valid).sum(-1)
                        bit_is_one = torch.bitwise_and(indices, mask).ne(0)
                        sign = 1.0 - 2.0 * bit_is_one.to(torch.float32)
                        score = (flipped_dot - current_dot) * sign
                        grad_q[:, :, exp, :, bit].add_(score)
                slope = (
                    ctx.scale
                    * ctx.temperature
                    * probabilities
                    * (1.0 - probabilities)
                )
                grad_q.mul_(slope)

        return (
            grad_memory,
            grad_q.to(q_windows.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@dataclass
class _LngramHaloPlan:
    """Static indices for one short CP halo exchange."""

    lags: Tuple[int, ...]
    send_local_indices: Tensor
    send_count: int
    max_send_count: int
    total_send_count: int
    slot_by_position: Tensor


class LngramContextParallelLayout:
    """Natural-position view of a zigzag context-parallel token layout.

    Only values within the requested causal lags are exchanged.  The full
    hidden sequence is never gathered or recomputed on every CP rank.
    """

    def __init__(
        self,
        *,
        cp_group,
        cp_rank: int,
        cp_size: int,
        cp_zigzag_perm: Tensor,
        cu_seqlens: Tensor,
        local_length: int,
        global_valid_tokens: Tensor,
    ):
        if cp_size <= 1:
            raise ValueError("LngramContextParallelLayout requires cp_size > 1")
        self.cp_group = cp_group
        self.cp_rank = int(cp_rank)
        self.cp_size = int(cp_size)
        self.local_length = int(local_length)
        self.perm = cp_zigzag_perm.to(dtype=torch.long).reshape(-1)
        self.global_length = int(self.perm.numel())
        if self.global_length != self.cp_size * self.local_length:
            raise ValueError(
                "cp_zigzag_perm length must equal cp_size * local_length; "
                f"got {self.global_length} != {self.cp_size} * {self.local_length}"
            )
        _tensor_assert(
            torch.all((self.perm >= 0) & (self.perm < self.global_length)),
            "cp_zigzag_perm contains an out-of-range global token position",
        )
        expected = torch.arange(
            self.global_length, device=self.perm.device, dtype=torch.long
        )
        rank_order = torch.arange(
            self.global_length, device=self.perm.device, dtype=torch.long
        )
        self.owner_by_position = torch.empty_like(rank_order)
        self.owner_by_position[self.perm] = torch.div(
            rank_order, self.local_length, rounding_mode="floor"
        )
        begin = self.cp_rank * self.local_length
        self.local_positions = self.perm[begin : begin + self.local_length]
        self.local_index_by_position = torch.full_like(rank_order, -1)
        self.local_index_by_position[self.local_positions] = torch.arange(
            self.local_length, device=self.perm.device, dtype=torch.long
        )

        cu = cu_seqlens.to(device=self.perm.device, dtype=torch.long).reshape(-1)
        if cu.numel() < 2:
            raise ValueError("CP LNGram requires full cu_seqlens")
        _tensor_assert(cu[0] == 0, "cu_seqlens must start at zero")
        _tensor_assert(cu[-1] == self.global_length, "cu_seqlens must span the full CP sequence")
        _tensor_assert(
            torch.all(cu[1:] >= cu[:-1]),
            "cu_seqlens must be monotonically non-decreasing",
        )
        self.segment_by_position = torch.bucketize(
            expected, cu[1:-1], right=True
        )
        valid = global_valid_tokens.to(device=self.perm.device, dtype=torch.bool).reshape(-1)
        if valid.numel() != self.global_length:
            raise ValueError(
                f"global_valid_tokens has {valid.numel()} entries, expected {self.global_length}"
            )
        self.global_valid_tokens = valid
        self.local_valid_tokens = valid[self.local_positions].unsqueeze(0)
        self._plans: Dict[Tuple[int, ...], _LngramHaloPlan] = {}

    def plan(self, lags: Sequence[int]) -> _LngramHaloPlan:
        lags_tuple = tuple(sorted({int(lag) for lag in lags if int(lag) > 0}))
        cached = self._plans.get(lags_tuple)
        if cached is not None:
            return cached
        needed = torch.zeros(
            self.global_length, device=self.perm.device, dtype=torch.bool
        )
        targets = torch.arange(
            self.global_length, device=self.perm.device, dtype=torch.long
        )
        for lag in lags_tuple:
            sources = targets - lag
            in_bounds = sources >= 0
            safe_sources = sources.clamp(min=0)
            cross_rank = (
                self.owner_by_position[safe_sources]
                != self.owner_by_position[targets]
            )
            same_segment = (
                self.segment_by_position[safe_sources]
                == self.segment_by_position[targets]
            )
            selected = safe_sources[in_bounds & cross_rank & same_segment]
            needed[selected] = True

        positions_by_rank = []
        counts = []
        for rank in range(self.cp_size):
            positions = torch.where(needed & (self.owner_by_position == rank))[0]
            positions_by_rank.append(positions)
            counts.append(int(positions.numel()))
        max_send_count = max(counts, default=0)
        total_send_count = sum(counts)
        slot_by_position = torch.full(
            (self.global_length,), -1, device=self.perm.device, dtype=torch.long
        )
        for rank, positions in enumerate(positions_by_rank):
            count = int(positions.numel())
            if count:
                slot_by_position[positions] = (
                    rank * max_send_count
                    + torch.arange(count, device=self.perm.device, dtype=torch.long)
                )
        send_positions = positions_by_rank[self.cp_rank]
        send_local_indices = self.local_index_by_position[send_positions]
        _tensor_assert(
            torch.all(send_local_indices >= 0),
            "CP halo send positions must be owned by the sending rank",
        )
        result = _LngramHaloPlan(
            lags=lags_tuple,
            send_local_indices=send_local_indices,
            send_count=counts[self.cp_rank],
            max_send_count=max_send_count,
            total_send_count=total_send_count,
            slot_by_position=slot_by_position,
        )
        self._plans[lags_tuple] = result
        return result

    def exchange(self, local_values: Tensor, plan: _LngramHaloPlan) -> Tensor:
        """Autograd-aware all-gather of only boundary values in ``plan``."""
        if local_values.dim() < 2 or local_values.shape[0] != 1:
            raise ValueError("CP LNGram currently requires [B=1,T,...] local values")
        if local_values.shape[1] != self.local_length:
            raise ValueError("local value length does not match the CP layout")
        feature_shape = tuple(local_values.shape[2:])
        if plan.max_send_count == 0:
            return local_values.new_empty((0, *feature_shape))
        selected = local_values[0].index_select(0, plan.send_local_indices)
        if plan.send_count < plan.max_send_count:
            selected = torch.cat(
                (
                    selected,
                    local_values.new_zeros(
                        (plan.max_send_count - plan.send_count, *feature_shape)
                    ),
                ),
                dim=0,
            )
        if selected.requires_grad:
            from torch.distributed.nn.functional import all_gather as differentiable_all_gather

            gathered = differentiable_all_gather(selected, group=self.cp_group)
        else:
            gathered = [torch.empty_like(selected) for _ in range(self.cp_size)]
            torch.distributed.all_gather(gathered, selected, group=self.cp_group)
        return torch.cat(tuple(gathered), dim=0)

    def fetch_lag(
        self,
        local_values: Tensor,
        gathered_values: Tensor,
        plan: _LngramHaloPlan,
        lag: int,
    ) -> Tuple[Tensor, Tensor]:
        """Fetch one causal lag and return values plus topology-valid mask."""
        lag = int(lag)
        if lag == 0:
            return local_values, torch.ones(
                (1, self.local_length),
                device=local_values.device,
                dtype=torch.bool,
            )
        sources = self.local_positions - lag
        in_bounds = sources >= 0
        safe_sources = sources.clamp(min=0)
        same_segment = (
            self.segment_by_position[safe_sources]
            == self.segment_by_position[self.local_positions]
        )
        topology_valid = in_bounds & same_segment
        source_owner = self.owner_by_position[safe_sources]
        local_source = source_owner == self.cp_rank
        local_indices = self.local_index_by_position[safe_sources].clamp(min=0)
        local_result = local_values.index_select(1, local_indices)
        slots = plan.slot_by_position[safe_sources].clamp(min=0)
        if gathered_values.shape[0]:
            remote_result = gathered_values.index_select(0, slots).unsqueeze(0)
        else:
            remote_result = torch.zeros_like(local_result)
        expand_shape = (1, self.local_length) + (1,) * (local_values.dim() - 2)
        result = torch.where(
            local_source.view(expand_shape), local_result, remote_result
        )
        result = result * topology_valid.view(expand_shape).to(result.dtype)
        remote_needed = topology_valid & ~local_source
        _tensor_assert(
            torch.all(~remote_needed | (plan.slot_by_position[safe_sources] >= 0)),
            "CP halo plan is missing a required remote predecessor",
        )
        return result, topology_valid.unsqueeze(0)


class _RouteNgramMemoryBase(nn.Module):
    """Hard route n-gram tables followed by a GQA route-memory readout."""

    # FP32 QK/PV is useful for cached readouts whose BF16 GEMM shape changes
    # between prefill and token decode.  Bound the temporary KV projection per
    # token so this numerical guard does not turn a large-route model into an
    # unbounded FP32 inference path.  Larger readouts retain the regular BF16
    # path and still benefit from the Conv1d-consistent incremental convolution.
    _INCREMENTAL_FP32_KV_BYTES_PER_TOKEN = 4 * 1024 * 1024
    _INCREMENTAL_FP32_WORKSPACE_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int = 4,
        num_routes: int = 16,
        memory_dim: int = 128,
        ngrams: Sequence[int] = (2, 3),
        initializer_range: float = 0.02,
        table_init_std_scale: float = 0.0,
        output_proj_init_std_scale: float = 1.0,
        readout_q_proj_init_std_scale: float = 1.0,
        readout_kv_proj_init_std_scale: float = 1.0,
        readout_num_heads: int = 16,
        readout_num_kv_heads: int = 2,
        readout_head_dim: int = 128,
        readout_attention_dropout: float = 0.0,
        rmsnorm_eps: float = 1e-6,
        normalization: str = "RMSNorm",
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        conv_bias: bool = False,
        conv_zero_init: bool = True,
        execution_mode: str = "full",
        auto_max_kv_mb: float = 384.0,
        stream_route_chunk_size: int = 64,
        readout_checkpoint_streaming: bool = False,
        q_surrogate_enable: bool = True,
        q_surrogate_mode: str = "approximate",
        q_surrogate_temperature: float = 1.0,
        q_surrogate_scale: float = 1.0,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bits_per_route = int(bits_per_route)
        self.num_routes = int(num_routes)
        self.memory_dim = int(memory_dim)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.bits_per_route <= 0 or self.bits_per_route > 16:
            raise ValueError(
                "bits_per_route must be in [1, 16]; larger code spaces make "
                "the exact surrogate and n-gram tables impractical"
            )
        if self.num_routes <= 0 or self.memory_dim <= 0:
            raise ValueError("num_routes and memory_dim must be positive")
        if float(table_init_std_scale) < 0.0:
            raise ValueError("table_init_std_scale must be non-negative")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= float(readout_attention_dropout) < 1.0:
            raise ValueError("readout_attention_dropout must be in [0, 1)")
        if float(auto_max_kv_mb) <= 0.0:
            raise ValueError("auto_max_kv_mb must be positive")
        if int(stream_route_chunk_size) <= 0:
            raise ValueError("stream_route_chunk_size must be positive")
        if float(q_surrogate_temperature) < 0.0:
            raise ValueError("q_surrogate_temperature must be non-negative")
        if int(conv_kernel_size) <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if conv_dilation is not None and int(conv_dilation) <= 0:
            raise ValueError("conv_dilation must be positive when supplied")
        self.ngrams = tuple(sorted({int(value) for value in ngrams}))
        if not self.ngrams or min(self.ngrams) < 1:
            raise ValueError(f"invalid n-gram orders: {self.ngrams}")
        self.max_ngram_order = max(self.ngrams)
        self.route_history_length = self.max_ngram_order - 1
        self.route_history_dtype = route_code_storage_dtype(self.bits_per_route)
        self.alphabet_size = 1 << self.bits_per_route
        self.route_logits_dim = self.num_routes * self.bits_per_route
        self.route_token_dim = len(self.ngrams) * self.memory_dim

        self.readout_num_heads = int(readout_num_heads)
        self.readout_num_kv_heads = int(readout_num_kv_heads)
        self.readout_head_dim = int(readout_head_dim)
        if (
            self.readout_num_heads <= 0
            or self.readout_num_kv_heads <= 0
            or self.readout_head_dim <= 0
        ):
            raise ValueError("readout head counts and head dimension must be positive")
        if self.readout_num_heads % self.readout_num_kv_heads:
            raise ValueError("readout_num_heads must be divisible by readout_num_kv_heads")
        if self.readout_num_heads * self.readout_head_dim != self.hidden_size:
            raise ValueError(
                "LNGram GQA must preserve the full query width: "
                "readout_num_heads * readout_head_dim must equal hidden_size; "
                f"got {self.readout_num_heads} * {self.readout_head_dim} != "
                f"{self.hidden_size}"
            )
        self.readout_groups = self.readout_num_heads // self.readout_num_kv_heads
        self.readout_inner_dim = self.readout_num_heads * self.readout_head_dim
        self.readout_kv_inner_dim = self.readout_num_kv_heads * self.readout_head_dim
        self.softmax_scale = self.readout_head_dim ** -0.5
        self.readout_attention_dropout = float(readout_attention_dropout)
        self.execution_mode = str(execution_mode).lower()
        if self.execution_mode not in {"full", "streaming", "auto"}:
            raise ValueError("execution_mode must be full, streaming, or auto")
        self.auto_max_kv_bytes = int(float(auto_max_kv_mb) * 1024 * 1024)
        self.stream_route_chunk_size = int(stream_route_chunk_size)
        self.readout_checkpoint_streaming = bool(readout_checkpoint_streaming)
        self.q_surrogate_enable = bool(q_surrogate_enable)
        self.q_surrogate_mode = str(q_surrogate_mode).lower()
        if self.q_surrogate_mode not in {"exact", "approximate"}:
            raise ValueError("q_surrogate_mode must be exact or approximate")
        self.q_surrogate_temperature = float(q_surrogate_temperature)
        self.q_surrogate_scale = float(q_surrogate_scale)
        max_rows = max(
            self.num_routes * (self.alphabet_size ** order)
            for order in self.ngrams
        )
        self.index_dtype = (
            torch.int32
            if max_rows - 1 <= torch.iinfo(torch.int32).max
            else torch.long
        )

        self.tables = nn.ModuleDict()
        for order in self.ngrams:
            rows = self.num_routes * (self.alphabet_size ** order)
            table = nn.Embedding(rows, self.memory_dim, device=device, dtype=dtype)
            _normal_init(table.weight, initializer_range * table_init_std_scale)
            self.tables[f"ngram_{order}"] = table
        self.lookup_dropout = nn.Dropout(float(dropout))

        self.readout_query_norm = build_lngram_norm(
            normalization, hidden_size, rmsnorm_eps, device=device, dtype=dtype
        )
        self.readout_memory_norm = build_lngram_norm(
            normalization, self.route_token_dim, rmsnorm_eps, device=device, dtype=dtype
        )
        self.readout_q_proj = nn.Linear(
            hidden_size, self.readout_inner_dim, bias=False, device=device, dtype=dtype
        )
        self.readout_kv_proj = nn.Linear(
            self.route_token_dim,
            2 * self.readout_kv_inner_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.readout_o_proj = nn.Linear(
            self.readout_inner_dim, hidden_size, bias=False, device=device, dtype=dtype
        )
        _normal_init(
            self.readout_q_proj.weight,
            initializer_range * readout_q_proj_init_std_scale,
        )
        _normal_init(
            self.readout_kv_proj.weight,
            initializer_range * readout_kv_proj_init_std_scale,
        )
        _normal_init(
            self.readout_o_proj.weight,
            initializer_range * output_proj_init_std_scale,
        )
        self.short_conv = DepthwiseCausalConv(
            hidden_size,
            kernel_size=conv_kernel_size,
            dilation=max(self.ngrams) if conv_dilation is None else int(conv_dilation),
            eps=rmsnorm_eps,
            normalization=normalization,
            bias=conv_bias,
            zero_init=conv_zero_init,
            device=device,
            dtype=dtype,
        )

    def quantize_route_logits(self, q_logits: Tensor) -> Tensor:
        if q_logits.shape[-1] != self.route_logits_dim:
            raise ValueError(
                f"route logits dim={q_logits.shape[-1]}, expected {self.route_logits_dim}"
            )
        return pack_bits_to_route_codes(q_logits > 0, self.bits_per_route)

    def _use_surrogate(self, q_logits: Optional[Tensor]) -> bool:
        return bool(
            self.q_surrogate_enable
            and self.training
            and torch.is_grad_enabled()
            and q_logits is not None
        )

    def _valid_mask(
        self,
        order: int,
        batch: int,
        time: int,
        device: torch.device,
        segment_ids: Optional[Tensor],
        valid_tokens: Optional[Tensor],
    ) -> Tensor:
        valid = torch.zeros(batch, time, dtype=torch.bool, device=device)
        if time < order:
            return valid
        suffix_valid = torch.ones(batch, time - order + 1, dtype=torch.bool, device=device)
        if segment_ids is not None:
            suffix_valid &= segment_ids[:, : time - order + 1] == segment_ids[:, order - 1 :]
        if valid_tokens is not None:
            suffix_valid &= valid_tokens[:, order - 1 :]
        valid[:, order - 1 :] = suffix_valid
        return valid

    def _lookup_route_chunk(
        self,
        route_codes: Tensor,
        q_logits_btrm: Optional[Tensor],
        route_start: int,
        route_end: int,
        segment_ids: Optional[Tensor],
        valid_tokens: Optional[Tensor],
        target_length: Optional[int] = None,
    ) -> Tensor:
        batch, time, _ = route_codes.shape
        codes = route_codes[:, :, route_start:route_end].to(self.index_dtype)
        q_chunk = (
            None if q_logits_btrm is None else q_logits_btrm[:, :, route_start:route_end, :]
        )
        use_surrogate = self._use_surrogate(q_logits_btrm)
        memories = []
        for order in self.ngrams:
            valid_mask = self._valid_mask(
                order, batch, time, codes.device, segment_ids, valid_tokens
            )
            valid_length = time - order + 1
            if valid_length <= 0:
                memory = self.tables[f"ngram_{order}"].weight.new_zeros(
                    batch, time, route_end - route_start, self.memory_dim
                )
            else:
                address = _aligned_ngram_address(codes, order, self.bits_per_route)
                offsets = (
                    torch.arange(
                        route_start,
                        route_end,
                        device=codes.device,
                        dtype=self.index_dtype,
                    )
                    * (self.alphabet_size ** order)
                ).view(1, 1, -1)
                values = self.tables[f"ngram_{order}"](offsets + address)
                memory = torch.cat(
                    (
                        values.new_zeros(batch, order - 1, route_end - route_start, self.memory_dim),
                        values,
                    ),
                    dim=1,
                )
                memory = memory * valid_mask.unsqueeze(-1).unsqueeze(-1).to(memory.dtype)
            if use_surrogate:
                if target_length is not None:
                    raise RuntimeError("cached inference cannot use surrogate gradients")
                memory = _LngramSurrogate.apply(
                    memory,
                    q_chunk,
                    codes,
                    self.tables[f"ngram_{order}"].weight,
                    valid_mask,
                    order,
                    self.bits_per_route,
                    route_start,
                    self.alphabet_size,
                    self.q_surrogate_temperature,
                    self.q_surrogate_scale,
                    self.q_surrogate_mode == "exact",
                )
            if target_length is not None:
                memory = memory[:, -int(target_length) :, :, :]
            memories.append(memory)
        return self.lookup_dropout(torch.cat(memories, dim=-1))

    def _project_query(self, hidden_states: Tensor) -> Tensor:
        batch, time, _ = hidden_states.shape
        query = self.readout_q_proj(self.readout_query_norm(hidden_states))
        return query.view(batch, time, self.readout_num_heads, self.readout_head_dim)

    def _project_memory(self, route_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        route_tokens = self.readout_memory_norm(route_tokens)
        kv = self.readout_kv_proj(route_tokens)
        key, value = kv.split(self.readout_kv_inner_dim, dim=-1)
        batch, time, routes, _ = key.shape
        key = key.view(
            batch, time, routes, self.readout_num_kv_heads, self.readout_head_dim
        ).permute(0, 1, 3, 2, 4)
        value = value.view(
            batch, time, routes, self.readout_num_kv_heads, self.readout_head_dim
        ).permute(0, 1, 3, 2, 4)
        return key, value

    def _project_memory_fp32_accumulation(
        self, route_tokens: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Project cached K/V with shape-stable FP32 accumulation.

        The projected tensors are quantized back to the activation dtype before
        attention.  This keeps the checkpoint/training dtype contract intact
        while avoiding BF16 GEMM reduction-order changes in cached inference.
        """
        route_tokens = self.readout_memory_norm(route_tokens)
        kv = F.linear(
            route_tokens.float(), self.readout_kv_proj.weight.float(), bias=None
        ).to(route_tokens.dtype)
        key, value = kv.split(self.readout_kv_inner_dim, dim=-1)
        batch, time, routes, _ = key.shape
        key = key.view(
            batch, time, routes, self.readout_num_kv_heads, self.readout_head_dim
        ).permute(0, 1, 3, 2, 4)
        value = value.view(
            batch, time, routes, self.readout_num_kv_heads, self.readout_head_dim
        ).permute(0, 1, 3, 2, 4)
        return key, value

    def _manual_gqa(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch, time, _, _ = query.shape
        grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * self.softmax_scale
        probabilities = torch.softmax(scores, dim=-1)
        if self.training and self.readout_attention_dropout:
            probabilities = F.dropout(
                probabilities, p=self.readout_attention_dropout, training=True
            )
        output = torch.matmul(probabilities.to(value.dtype), value)
        return output.reshape(
            batch, time, self.readout_num_heads, self.readout_head_dim
        )

    def _manual_gqa_fp32(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tensor:
        """Inference-only GQA with FP32 operands for both matrix products."""
        batch, time, _, _ = query.shape
        grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        scores = (
            torch.matmul(grouped.float(), key.transpose(-1, -2).float())
            * self.softmax_scale
        )
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.matmul(probabilities, value.float()).to(query.dtype)
        return output.reshape(
            batch, time, self.readout_num_heads, self.readout_head_dim
        )

    def _incremental_head_output(
        self, query: Tensor, route_tokens: Tensor
    ) -> Tensor:
        """Run cached readout without changing the training/full path."""
        fp32_bytes_per_token = (
            2
            * self.num_routes
            * self.readout_kv_inner_dim
            * 4  # torch.float32 element size
        )
        # The standard 2-group GQA kernel is already stable in BF16 and keeps
        # the original fast path.  The no-grouping and highly-grouped extremes
        # are more sensitive to prefill/decode shape changes, so use the
        # bounded FP32 path for those layouts only.
        if (
            self.readout_groups == 2
            or fp32_bytes_per_token > self._INCREMENTAL_FP32_KV_BYTES_PER_TOKEN
        ):
            key, value = self._project_memory(route_tokens)
            return self._manual_gqa(query, key, value)

        time_chunk = max(
            1,
            self._INCREMENTAL_FP32_WORKSPACE_BYTES
            // max(1, fp32_bytes_per_token),
        )
        outputs = []
        for start in range(0, query.shape[1], time_chunk):
            end = min(query.shape[1], start + time_chunk)
            key, value = self._project_memory_fp32_accumulation(
                route_tokens[:, start:end]
            )
            outputs.append(
                self._manual_gqa_fp32(query[:, start:end], key, value)
            )
        return torch.cat(outputs, dim=1)

    def _streaming_gqa(
        self,
        query: Tensor,
        route_codes: Tensor,
        q_logits_btrm: Optional[Tensor],
        segment_ids: Optional[Tensor],
        valid_tokens: Optional[Tensor],
        target_length: Optional[int] = None,
    ) -> Tensor:
        batch, time, _, _ = query.shape
        query_grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        state_shape = (batch, time, self.readout_num_kv_heads, self.readout_groups)
        running_max = query.new_full(state_shape, -math.inf, dtype=torch.float32)
        running_sum = query.new_zeros(state_shape, dtype=torch.float32)
        running_value = query.new_zeros(
            (*state_shape, self.readout_head_dim), dtype=torch.float32
        )
        checkpoint_chunks = bool(
            self.readout_checkpoint_streaming
            and self.training
            and torch.is_grad_enabled()
            and target_length is None
        )
        empty_logits = query.new_empty((0,))

        for start in range(0, self.num_routes, self.stream_route_chunk_size):
            end = min(self.num_routes, start + self.stream_route_chunk_size)
            q_arg = q_logits_btrm if q_logits_btrm is not None else empty_logits

            def chunk_step(
                running_max_arg: Tensor,
                running_sum_arg: Tensor,
                running_value_arg: Tensor,
                query_grouped_arg: Tensor,
                route_codes_arg: Tensor,
                q_logits_arg: Tensor,
                _start: int = start,
                _end: int = end,
                _has_q: bool = q_logits_btrm is not None,
            ) -> Tuple[Tensor, Tensor, Tensor]:
                route_tokens = self._lookup_route_chunk(
                    route_codes_arg,
                    q_logits_arg if _has_q else None,
                    _start,
                    _end,
                    segment_ids,
                    valid_tokens,
                    target_length=target_length,
                )
                key, value = self._project_memory(route_tokens)
                scores = (
                    torch.matmul(query_grouped_arg, key.transpose(-1, -2)).float()
                    * self.softmax_scale
                )
                chunk_max = scores.max(-1).values
                new_max = torch.maximum(running_max_arg, chunk_max).detach()
                old_scale = torch.exp(running_max_arg - new_max)
                exponentials = torch.exp(scores - new_max.unsqueeze(-1))
                new_sum = running_sum_arg * old_scale + exponentials.sum(-1)
                if self.training and self.readout_attention_dropout:
                    numerator = F.dropout(
                        exponentials,
                        p=self.readout_attention_dropout,
                        training=True,
                    )
                else:
                    numerator = exponentials
                chunk_value = torch.matmul(numerator, value.float())
                new_value = (
                    running_value_arg * old_scale.unsqueeze(-1) + chunk_value
                )
                return new_max, new_sum, new_value

            if checkpoint_chunks:
                running_max, running_sum, running_value = activation_checkpoint(
                    chunk_step,
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    route_codes,
                    q_arg,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                running_max, running_sum, running_value = chunk_step(
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    route_codes,
                    q_arg,
                )
        return (running_value / running_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)).to(
            query.dtype
        ).reshape(batch, time, self.readout_num_heads, self.readout_head_dim)

    def _resolve_mode(self, hidden_states: Tensor) -> str:
        if self.execution_mode != "auto":
            return self.execution_mode
        batch, time, _ = hidden_states.shape
        element_size = self.readout_kv_proj.weight.element_size()
        estimated = (
            batch
            * time
            * self.num_routes
            * (self.route_token_dim + 2 * self.readout_kv_inner_dim)
            * element_size
            * 2
        )
        return "full" if estimated <= self.auto_max_kv_bytes else "streaming"

    def _finish(
        self, head_output: Tensor, segment_ids: Optional[Tensor]
    ) -> Tensor:
        batch, time, _, _ = head_output.shape
        mixed = self.readout_o_proj(head_output.reshape(batch, time, self.readout_inner_dim))
        return mixed + self.short_conv(mixed, segment_ids=segment_ids)

    def forward(
        self,
        hidden_states: Tensor,
        route_codes_btr: Tensor,
        q_logits: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        valid_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must be [B,T,H]")
        batch, time, hidden = hidden_states.shape
        if hidden != self.hidden_size:
            raise ValueError(f"hidden size={hidden}, expected {self.hidden_size}")
        if tuple(route_codes_btr.shape) != (batch, time, self.num_routes):
            raise ValueError("route_codes_btr has an incompatible shape")
        q_logits_btrm = None
        if q_logits is not None:
            expected = (batch, time, self.route_logits_dim)
            if tuple(q_logits.shape) != expected:
                raise ValueError(f"q_logits shape={tuple(q_logits.shape)}, expected {expected}")
            q_logits_btrm = q_logits.reshape(
                batch, time, self.num_routes, self.bits_per_route
            )
        query = self._project_query(hidden_states)
        if self._resolve_mode(hidden_states) == "streaming":
            head_output = self._streaming_gqa(
                query,
                route_codes_btr,
                q_logits_btrm,
                segment_ids,
                valid_tokens,
            )
        else:
            tokens = self._lookup_route_chunk(
                route_codes_btr,
                q_logits_btrm,
                0,
                self.num_routes,
                segment_ids,
                valid_tokens,
            )
            key, value = self._project_memory(tokens)
            head_output = self._manual_gqa(query, key, value)
        output = self._finish(head_output, segment_ids)
        if valid_tokens is not None:
            output = output * valid_tokens.unsqueeze(-1).to(output.dtype)
        return output

    def _lookup_window_route_chunk(
        self,
        history_values: Dict[int, Tensor],
        history_topology_valid: Dict[int, Tensor],
        values_are_q_logits: bool,
        route_start: int,
        route_end: int,
        current_valid_tokens: Tensor,
    ) -> Tensor:
        """Lookup route memories for arbitrary CP-local natural positions."""
        sample = history_values[0]
        batch, time = sample.shape[:2]
        memories = []
        routes = route_end - route_start
        for order in self.ngrams:
            lag_order = tuple(range(order - 1, -1, -1))
            # In a contiguous segment, checking the oldest predecessor against
            # the current token is equivalent to checking every position in the
            # window.  This exactly matches _valid_mask's packed semantics.
            valid_mask = current_valid_tokens & history_topology_valid[order - 1]
            if values_are_q_logits:
                q_windows = torch.stack(
                    tuple(
                        history_values[lag]
                        .reshape(batch, time, self.num_routes, self.bits_per_route)[
                            :, :, route_start:route_end, :
                        ]
                        for lag in lag_order
                    ),
                    dim=2,
                )
                bit_weights = torch.bitwise_left_shift(
                    torch.ones(
                        self.bits_per_route,
                        device=sample.device,
                        dtype=torch.int32,
                    ),
                    torch.arange(
                        self.bits_per_route,
                        device=sample.device,
                        dtype=torch.int32,
                    ),
                )
                code_windows = torch.sum(
                    (q_windows > 0).to(torch.int32)
                    * bit_weights.view(1, 1, 1, 1, -1),
                    dim=-1,
                    dtype=torch.int32,
                )
            else:
                q_windows = None
                code_windows = torch.stack(
                    tuple(
                        history_values[lag][:, :, route_start:route_end]
                        for lag in lag_order
                    ),
                    dim=2,
                ).to(torch.int32)

            address = _window_ngram_address(
                code_windows.to(self.index_dtype), self.bits_per_route
            )
            offsets = (
                torch.arange(
                    route_start,
                    route_end,
                    device=sample.device,
                    dtype=self.index_dtype,
                )
                * (self.alphabet_size ** order)
            ).view(1, 1, routes)
            memory = self.tables[f"ngram_{order}"](offsets + address)
            memory = memory * valid_mask.unsqueeze(-1).unsqueeze(-1).to(memory.dtype)
            if q_windows is not None:
                memory = _LngramWindowSurrogate.apply(
                    memory,
                    q_windows,
                    code_windows,
                    self.tables[f"ngram_{order}"].weight,
                    valid_mask,
                    self.bits_per_route,
                    route_start,
                    self.alphabet_size,
                    self.q_surrogate_temperature,
                    self.q_surrogate_scale,
                    self.q_surrogate_mode == "exact",
                )
            memories.append(memory)
        return self.lookup_dropout(torch.cat(memories, dim=-1))

    def _context_parallel_head_output(
        self,
        hidden_states: Tensor,
        history_values: Dict[int, Tensor],
        history_topology_valid: Dict[int, Tensor],
        values_are_q_logits: bool,
        current_valid_tokens: Tensor,
    ) -> Tensor:
        query = self._project_query(hidden_states)
        if self._resolve_mode(hidden_states) == "full":
            route_tokens = self._lookup_window_route_chunk(
                history_values,
                history_topology_valid,
                values_are_q_logits,
                0,
                self.num_routes,
                current_valid_tokens,
            )
            key, value = self._project_memory(route_tokens)
            return self._manual_gqa(query, key, value)

        batch, time, _, _ = query.shape
        query_grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        state_shape = (batch, time, self.readout_num_kv_heads, self.readout_groups)
        running_max = query.new_full(state_shape, -math.inf, dtype=torch.float32)
        running_sum = query.new_zeros(state_shape, dtype=torch.float32)
        running_value = query.new_zeros(
            (*state_shape, self.readout_head_dim), dtype=torch.float32
        )
        checkpoint_chunks = bool(
            self.readout_checkpoint_streaming
            and self.training
            and torch.is_grad_enabled()
        )
        history_lags = tuple(sorted(history_values))
        history_args = tuple(history_values[lag] for lag in history_lags)

        for start in range(0, self.num_routes, self.stream_route_chunk_size):
            end = min(self.num_routes, start + self.stream_route_chunk_size)

            def chunk_step(
                running_max_arg: Tensor,
                running_sum_arg: Tensor,
                running_value_arg: Tensor,
                query_grouped_arg: Tensor,
                *history_args_inner: Tensor,
                _start: int = start,
                _end: int = end,
            ) -> Tuple[Tensor, Tensor, Tensor]:
                history_inner = dict(zip(history_lags, history_args_inner))
                route_tokens = self._lookup_window_route_chunk(
                    history_inner,
                    history_topology_valid,
                    values_are_q_logits,
                    _start,
                    _end,
                    current_valid_tokens,
                )
                key, value = self._project_memory(route_tokens)
                scores = (
                    torch.matmul(query_grouped_arg, key.transpose(-1, -2)).float()
                    * self.softmax_scale
                )
                chunk_max = scores.max(-1).values
                new_max = torch.maximum(running_max_arg, chunk_max).detach()
                old_scale = torch.exp(running_max_arg - new_max)
                exponentials = torch.exp(scores - new_max.unsqueeze(-1))
                new_sum = running_sum_arg * old_scale + exponentials.sum(-1)
                if self.training and self.readout_attention_dropout:
                    numerator = F.dropout(
                        exponentials,
                        p=self.readout_attention_dropout,
                        training=True,
                    )
                else:
                    numerator = exponentials
                chunk_value = torch.matmul(numerator, value.float())
                new_value = (
                    running_value_arg * old_scale.unsqueeze(-1) + chunk_value
                )
                return new_max, new_sum, new_value

            if checkpoint_chunks:
                running_max, running_sum, running_value = activation_checkpoint(
                    chunk_step,
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    *history_args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                running_max, running_sum, running_value = chunk_step(
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    *history_args,
                )
        return (
            running_value
            / running_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        ).to(query.dtype).reshape(
            batch, time, self.readout_num_heads, self.readout_head_dim
        )

    def forward_context_parallel(
        self,
        hidden_states: Tensor,
        route_codes_btr: Tensor,
        q_logits: Optional[Tensor],
        layout: LngramContextParallelLayout,
    ) -> Tensor:
        """Exact CP-sharded LNGram with differentiable short halo exchange."""
        if hidden_states.dim() != 3 or hidden_states.shape[0] != 1:
            raise ValueError("CP LNGram requires hidden_states shaped [1,T,H]")
        batch, time, hidden = hidden_states.shape
        if hidden != self.hidden_size or time != layout.local_length:
            raise ValueError("hidden states do not match the CP LNGram layout")
        if tuple(route_codes_btr.shape) != (batch, time, self.num_routes):
            raise ValueError("route_codes_btr has an incompatible CP-local shape")

        use_surrogate = self._use_surrogate(q_logits)
        route_lags = tuple(range(1, self.max_ngram_order))
        route_plan = layout.plan(route_lags)
        history_values: Dict[int, Tensor] = {}
        history_topology_valid: Dict[int, Tensor] = {
            0: torch.ones(
                (1, time), device=hidden_states.device, dtype=torch.bool
            )
        }
        if use_surrogate:
            expected = (batch, time, self.route_logits_dim)
            if q_logits is None or tuple(q_logits.shape) != expected:
                raise ValueError(f"q_logits shape must be {expected} for CP surrogate training")
            route_source = q_logits
            values_are_q_logits = True
        else:
            route_source = route_codes_btr.to(self.route_history_dtype)
            values_are_q_logits = False
        history_values[0] = route_source
        gathered_routes = layout.exchange(route_source, route_plan)
        for lag in route_lags:
            history_values[lag], history_topology_valid[lag] = layout.fetch_lag(
                route_source, gathered_routes, route_plan, lag
            )

        head_output = self._context_parallel_head_output(
            hidden_states,
            history_values,
            history_topology_valid,
            values_are_q_logits,
            layout.local_valid_tokens,
        )
        mixed = self.readout_o_proj(
            head_output.reshape(batch, time, self.readout_inner_dim)
        )

        conv = self.short_conv
        normalized = conv.norm(mixed)
        conv_lags = tuple(
            (conv.kernel_size - 1 - tap) * conv.dilation
            for tap in range(conv.kernel_size)
            if (conv.kernel_size - 1 - tap) * conv.dilation > 0
        )
        conv_plan = layout.plan(conv_lags)
        gathered_conv = layout.exchange(normalized, conv_plan)
        conv_output = torch.zeros_like(normalized)
        for tap in range(conv.kernel_size):
            lag = (conv.kernel_size - 1 - tap) * conv.dilation
            shifted, topology_valid = layout.fetch_lag(
                normalized, gathered_conv, conv_plan, lag
            )
            weight = conv.conv.weight[:, 0, tap].view(1, 1, self.hidden_size)
            conv_output = conv_output + (
                shifted * topology_valid.unsqueeze(-1).to(shifted.dtype) * weight
            )
        if conv.conv.bias is not None:
            conv_output = conv_output + conv.conv.bias.view(1, 1, self.hidden_size)
        output = mixed + conv.act(conv_output)
        output = output * layout.local_valid_tokens.unsqueeze(-1).to(output.dtype)

        route_width = (
            self.route_logits_dim if values_are_q_logits else self.num_routes
        )
        route_bytes = route_plan.max_send_count * route_width * route_source.element_size()
        conv_bytes = conv_plan.max_send_count * self.hidden_size * normalized.element_size()
        self._last_context_parallel_stats = {
            "cp_size": layout.cp_size,
            "local_tokens": layout.local_length,
            "global_tokens": layout.global_length,
            "route_halo_lags": route_plan.lags,
            "route_halo_tokens_sent": route_plan.send_count,
            "route_halo_tokens_padded": route_plan.max_send_count,
            "route_halo_tokens_global": route_plan.total_send_count,
            "conv_halo_lags": conv_plan.lags,
            "conv_halo_tokens_sent": conv_plan.send_count,
            "conv_halo_tokens_padded": conv_plan.max_send_count,
            "conv_halo_tokens_global": conv_plan.total_send_count,
            "per_rank_forward_send_bytes": int(route_bytes + conv_bytes),
            "full_sequence_reference_send_bytes": int(
                layout.local_length
                * (route_width * route_source.element_size() + self.hidden_size * normalized.element_size())
            ),
        }
        return output

    def _pack_suffix(self, codes: Tensor) -> Tuple[Tensor, int]:
        keep = min(codes.shape[1], self.max_ngram_order)
        packed = codes.new_zeros(codes.shape[0], codes.shape[2], dtype=torch.int32)
        suffix = codes[:, -keep:, :].to(torch.int32)
        first_digit = self.max_ngram_order - keep
        for offset in range(keep):
            packed.bitwise_or_(
                torch.bitwise_left_shift(
                    suffix[:, offset, :],
                    (first_digit + offset) * self.bits_per_route,
                )
            )
        return packed, keep

    def _advance_packed(
        self, packed: Optional[Tensor], current: Tensor, state_length: int
    ) -> Tuple[Tensor, int]:
        current = current.to(torch.int32)
        if packed is None:
            packed = torch.zeros_like(current)
        packed = torch.bitwise_right_shift(packed.to(current.device), self.bits_per_route)
        packed.bitwise_or_(
            torch.bitwise_left_shift(
                current, (self.max_ngram_order - 1) * self.bits_per_route
            )
        )
        return packed, min(int(state_length) + 1, self.max_ngram_order)

    def _lookup_packed(self, packed: Tensor, state_length: int) -> Tensor:
        batch, routes = packed.shape
        route_tokens = self.readout_kv_proj.weight.new_zeros(
            batch, 1, routes, self.route_token_dim
        )
        for order_index, order in enumerate(self.ngrams):
            if state_length < order:
                continue
            address = torch.bitwise_right_shift(
                packed, (self.max_ngram_order - order) * self.bits_per_route
            ).to(self.index_dtype)
            offsets = (
                torch.arange(routes, device=packed.device, dtype=self.index_dtype)
                * (self.alphabet_size ** order)
            ).view(1, routes)
            values = self.tables[f"ngram_{order}"](offsets + address)
            start = order_index * self.memory_dim
            route_tokens[:, 0, :, start : start + self.memory_dim] = values
        return self.lookup_dropout(route_tokens)

    @torch.no_grad()
    def forward_incremental(
        self,
        hidden_states: Tensor,
        route_codes_btr: Tensor,
        state: Optional[LngramIncrementalState] = None,
    ) -> Tuple[Tensor, LngramIncrementalState]:
        """Exact prompt/decode execution.  Surrogate gradients are training-only."""
        if self.training:
            raise RuntimeError("forward_incremental is inference-only; call eval() first")
        if state is None:
            state = LngramIncrementalState()
        batch, target_length, _ = hidden_states.shape
        if tuple(route_codes_btr.shape) != (batch, target_length, self.num_routes):
            raise ValueError("incremental route code shape mismatch")
        query = self._project_query(hidden_states)

        if target_length == 1:
            packed, state_length = self._advance_packed(
                state.packed_route_state,
                route_codes_btr[:, 0, :],
                state.route_state_length,
            )
            tokens = self._lookup_packed(packed, state_length)
            head_output = self._incremental_head_output(query, tokens)
            context = (
                route_codes_btr
                if state.route_history is None
                else torch.cat((state.route_history.to(route_codes_btr), route_codes_btr), dim=1)
            )
        else:
            context = (
                route_codes_btr
                if state.route_history is None
                else torch.cat((state.route_history.to(route_codes_btr), route_codes_btr), dim=1)
            )
            tokens = self._lookup_route_chunk(
                context,
                None,
                0,
                self.num_routes,
                None,
                None,
                target_length=target_length,
            )
            head_output = self._incremental_head_output(query, tokens)
            packed, suffix_length = self._pack_suffix(context)
            state_length = min(
                int(state.route_state_length) + target_length, self.max_ngram_order
            )
            if state.route_state_length == 0 and state.route_history is None:
                state_length = suffix_length

        mixed = self.readout_o_proj(
            head_output.reshape(batch, target_length, self.readout_inner_dim)
        )
        conv, conv_history = self.short_conv.forward_incremental(
            mixed, state.conv_history
        )
        if self.route_history_length:
            route_history = context[:, -self.route_history_length :, :].to(
                self.route_history_dtype
            ).contiguous()
        else:
            route_history = context[:, :0, :]
        new_state = LngramIncrementalState(
            route_history=route_history,
            packed_route_state=packed,
            route_state_length=int(state_length),
            conv_history=conv_history,
        )
        return mixed + conv, new_state


def estimate_lngram_v2_parameters_per_layer(
    hidden_size: int,
    bits_per_route: int,
    num_routes: int,
    memory_dim: int,
    ngrams: Sequence[int],
    readout_num_heads: int,
    readout_num_kv_heads: int,
    readout_head_dim: int,
    conv_kernel_size: int,
) -> int:
    alphabet = 1 << int(bits_per_route)
    table = sum(num_routes * (alphabet ** int(order)) * memory_dim for order in ngrams)
    route_dim = len(tuple(ngrams)) * memory_dim
    query_dim = readout_num_heads * readout_head_dim
    kv_dim = readout_num_kv_heads * readout_head_dim
    projections = hidden_size * query_dim + route_dim * 2 * kv_dim + query_dim * hidden_size
    norms = hidden_size + route_dim + hidden_size
    conv = hidden_size * conv_kernel_size
    router = hidden_size * num_routes * bits_per_route
    return int(table + projections + norms + conv + router)

class RouteNgramMemoryV2(_RouteNgramMemoryBase):
    """Route memory whose real entries compete with one fixed zero-value sink.

    Let ``s_i`` and ``v_i`` be the score and value of real route ``i`` and let
    ``b_0`` be the fixed sink logit.  The v2 readout is

    ``sum_i exp(s_i) v_i / (exp(b_0) + sum_j exp(s_j))``.

    No sink value tensor or trainable sink parameter is materialized.  The
    default ``sink_initial_active_mass=0.5`` chooses

    ``b_0 = log(num_routes * (1 - mass) / mass)``,

    so zero real logits assign half of the total probability mass to the real
    routes and half to the zero-valued sink.  ``b_0`` is a Python scalar derived
    from checkpointed configuration; it receives no gradient, optimizer state,
    weight decay, communication, or checkpoint tensor.
    """

    def __init__(self, *args, sink_initial_active_mass: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        mass = float(sink_initial_active_mass)
        if not math.isfinite(mass) or not 0.0 < mass < 1.0:
            raise ValueError("sink_initial_active_mass must be finite and in (0, 1)")
        self.sink_initial_active_mass = mass
        self.readout_sink_logit = math.log(
            self.num_routes * (1.0 - mass) / mass
        )

    def _real_route_probabilities(self, scores: Tensor) -> Tensor:
        """Return real-route probabilities including the sink in the denominator."""
        scores = scores.float()
        sink_logit = scores.new_tensor(self.readout_sink_logit)
        normalizer_max = torch.maximum(
            scores.amax(dim=-1, keepdim=True), sink_logit
        ).detach()
        real_exponentials = torch.exp(scores - normalizer_max)
        sink_exponential = torch.exp(sink_logit - normalizer_max)
        denominator = real_exponentials.sum(dim=-1, keepdim=True) + sink_exponential
        return real_exponentials / denominator

    def _manual_gqa(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch, time, _, _ = query.shape
        grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        scores = (
            torch.matmul(grouped, key.transpose(-1, -2)).float()
            * self.softmax_scale
        )
        probabilities = self._real_route_probabilities(scores)
        if self.training and self.readout_attention_dropout:
            probabilities = F.dropout(
                probabilities, p=self.readout_attention_dropout, training=True
            )
        output = torch.matmul(probabilities.to(value.dtype), value)
        return output.reshape(
            batch, time, self.readout_num_heads, self.readout_head_dim
        )

    def _manual_gqa_fp32(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tensor:
        """Inference-only FP32 GQA with the same v2 denominator."""
        batch, time, _, _ = query.shape
        grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        scores = (
            torch.matmul(grouped.float(), key.transpose(-1, -2).float())
            * self.softmax_scale
        )
        probabilities = self._real_route_probabilities(scores)
        output = torch.matmul(probabilities, value.float()).to(query.dtype)
        return output.reshape(
            batch, time, self.readout_num_heads, self.readout_head_dim
        )

    def _streaming_update(
        self,
        running_max: Tensor,
        running_sum: Tensor,
        running_value: Tensor,
        scores: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Add one real-route chunk to an online softmax seeded by the sink."""
        chunk_max = scores.max(dim=-1).values
        new_max = torch.maximum(running_max, chunk_max).detach()
        old_scale = torch.exp(running_max - new_max)
        exponentials = torch.exp(scores - new_max.unsqueeze(-1))
        new_sum = running_sum * old_scale + exponentials.sum(dim=-1)
        if self.training and self.readout_attention_dropout:
            numerator = F.dropout(
                exponentials,
                p=self.readout_attention_dropout,
                training=True,
            )
        else:
            numerator = exponentials
        chunk_value = torch.matmul(numerator, value.float())
        new_value = running_value * old_scale.unsqueeze(-1) + chunk_value
        return new_max, new_sum, new_value

    def _streaming_gqa(
        self,
        query: Tensor,
        route_codes: Tensor,
        q_logits_btrm: Optional[Tensor],
        segment_ids: Optional[Tensor],
        valid_tokens: Optional[Tensor],
        target_length: Optional[int] = None,
    ) -> Tensor:
        batch, time, _, _ = query.shape
        query_grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        state_shape = (batch, time, self.readout_num_kv_heads, self.readout_groups)

        # These three tensors already exist in v1 streaming.  Initializing the
        # max/sum as (b_0, 1) represents exp(b_0-b_0) in the denominator; the
        # zero numerator means no sink value allocation or matmul is needed.
        running_max = query.new_full(
            state_shape, self.readout_sink_logit, dtype=torch.float32
        )
        running_sum = query.new_ones(state_shape, dtype=torch.float32)
        running_value = query.new_zeros(
            (*state_shape, self.readout_head_dim), dtype=torch.float32
        )
        checkpoint_chunks = bool(
            self.readout_checkpoint_streaming
            and self.training
            and torch.is_grad_enabled()
            and target_length is None
        )
        empty_logits = query.new_empty((0,))

        for start in range(0, self.num_routes, self.stream_route_chunk_size):
            end = min(self.num_routes, start + self.stream_route_chunk_size)
            q_arg = q_logits_btrm if q_logits_btrm is not None else empty_logits

            def chunk_step(
                running_max_arg: Tensor,
                running_sum_arg: Tensor,
                running_value_arg: Tensor,
                query_grouped_arg: Tensor,
                route_codes_arg: Tensor,
                q_logits_arg: Tensor,
                _start: int = start,
                _end: int = end,
                _has_q: bool = q_logits_btrm is not None,
            ) -> Tuple[Tensor, Tensor, Tensor]:
                route_tokens = self._lookup_route_chunk(
                    route_codes_arg,
                    q_logits_arg if _has_q else None,
                    _start,
                    _end,
                    segment_ids,
                    valid_tokens,
                    target_length=target_length,
                )
                key, value = self._project_memory(route_tokens)
                scores = (
                    torch.matmul(query_grouped_arg, key.transpose(-1, -2)).float()
                    * self.softmax_scale
                )
                return self._streaming_update(
                    running_max_arg,
                    running_sum_arg,
                    running_value_arg,
                    scores,
                    value,
                )

            if checkpoint_chunks:
                running_max, running_sum, running_value = activation_checkpoint(
                    chunk_step,
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    route_codes,
                    q_arg,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                running_max, running_sum, running_value = chunk_step(
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    route_codes,
                    q_arg,
                )
        return (
            running_value
            / running_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        ).to(query.dtype).reshape(
            batch, time, self.readout_num_heads, self.readout_head_dim
        )

    def _context_parallel_head_output(
        self,
        hidden_states: Tensor,
        history_values: Dict[int, Tensor],
        history_topology_valid: Dict[int, Tensor],
        values_are_q_logits: bool,
        current_valid_tokens: Tensor,
    ) -> Tensor:
        query = self._project_query(hidden_states)
        if self._resolve_mode(hidden_states) == "full":
            route_tokens = self._lookup_window_route_chunk(
                history_values,
                history_topology_valid,
                values_are_q_logits,
                0,
                self.num_routes,
                current_valid_tokens,
            )
            key, value = self._project_memory(route_tokens)
            return self._manual_gqa(query, key, value)

        batch, time, _, _ = query.shape
        query_grouped = query.view(
            batch,
            time,
            self.readout_num_kv_heads,
            self.readout_groups,
            self.readout_head_dim,
        )
        state_shape = (batch, time, self.readout_num_kv_heads, self.readout_groups)
        running_max = query.new_full(
            state_shape, self.readout_sink_logit, dtype=torch.float32
        )
        running_sum = query.new_ones(state_shape, dtype=torch.float32)
        running_value = query.new_zeros(
            (*state_shape, self.readout_head_dim), dtype=torch.float32
        )
        checkpoint_chunks = bool(
            self.readout_checkpoint_streaming
            and self.training
            and torch.is_grad_enabled()
        )
        history_lags = tuple(sorted(history_values))
        history_args = tuple(history_values[lag] for lag in history_lags)

        for start in range(0, self.num_routes, self.stream_route_chunk_size):
            end = min(self.num_routes, start + self.stream_route_chunk_size)

            def chunk_step(
                running_max_arg: Tensor,
                running_sum_arg: Tensor,
                running_value_arg: Tensor,
                query_grouped_arg: Tensor,
                *history_args_inner: Tensor,
                _start: int = start,
                _end: int = end,
            ) -> Tuple[Tensor, Tensor, Tensor]:
                history_inner = dict(zip(history_lags, history_args_inner))
                route_tokens = self._lookup_window_route_chunk(
                    history_inner,
                    history_topology_valid,
                    values_are_q_logits,
                    _start,
                    _end,
                    current_valid_tokens,
                )
                key, value = self._project_memory(route_tokens)
                scores = (
                    torch.matmul(query_grouped_arg, key.transpose(-1, -2)).float()
                    * self.softmax_scale
                )
                return self._streaming_update(
                    running_max_arg,
                    running_sum_arg,
                    running_value_arg,
                    scores,
                    value,
                )

            if checkpoint_chunks:
                running_max, running_sum, running_value = activation_checkpoint(
                    chunk_step,
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    *history_args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                running_max, running_sum, running_value = chunk_step(
                    running_max,
                    running_sum,
                    running_value,
                    query_grouped,
                    *history_args,
                )
        return (
            running_value
            / running_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        ).to(query.dtype).reshape(
            batch, time, self.readout_num_heads, self.readout_head_dim
        )

# Compatibility alias for integrations that used the earlier helper name.
estimate_lngram_parameters_per_layer = estimate_lngram_v2_parameters_per_layer

__all__ = [
    "RouteNgramMemoryV2",
    "LngramIncrementalState",
    "LngramContextParallelLayout",
    "estimate_lngram_v2_parameters_per_layer",
    "estimate_lngram_parameters_per_layer",
]
