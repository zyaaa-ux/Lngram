



from __future__ import annotations

import math
import types
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LngramConfig:
    """Configuration for the latent route n-gram memory module."""

    enabled: bool = True
    target_layers: Tuple[int, ...] = (1, 11)

    bits_per_route: int = 4
    ngrams: Tuple[int, ...] = (2, 3)
    memory_dim: int = 16
    dropout: float = 0.0

    q_proj_init_std_scale: float = 1.0
    table_init_mode: str = "zeros"
    table_init_std_scale: float = 1.0
    output_proj_init_std_scale: float = 1.0

    q_surrogate_enable: bool = True
    q_surrogate_temp: float = 1.0
    q_surrogate_scale: float = 1.0
    q_surrogate_route_chunk_size: int = 256
    q_surrogate_accum_fp32: bool = True

    table_lr_multiplier: float = 5.0
    table_weight_decay: float = 0.0

    proj_chunk_size: Optional[int] = None

    conv_kernel_size: int = 4
    conv_dilation: Optional[int] = None
    conv_bias: bool = False
    conv_zero_init: bool = True

    def validate(self, hidden_size: int, num_hidden_layers: Optional[int] = None) -> None:
        if hidden_size % self.bits_per_route != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by bits_per_route={self.bits_per_route}."
            )
        if any(int(n) < 1 for n in self.ngrams):
            raise ValueError(f"All n-gram orders must be >= 1, got {self.ngrams}.")
        if self.table_init_mode not in {"zeros", "normal"}:
            raise ValueError(
                f"table_init_mode must be 'zeros' or 'normal', got {self.table_init_mode!r}."
            )
        if num_hidden_layers is not None:
            invalid = [idx for idx in self.target_layers if not (0 <= int(idx) < num_hidden_layers)]
            if invalid:
                raise ValueError(
                    f"target_layers contains out-of-range indices {invalid} for "
                    f"num_hidden_layers={num_hidden_layers}."
                )


def resolve_added_module_init_std(config) -> float:
    """Resolve a safe initializer std from a model config-like object."""
    std = float(getattr(config, "initializer_range", 0.02))
    return std if std > 0.0 else 0.02


@torch.no_grad()
def init_linear_weight_(weight: torch.Tensor, std: float) -> None:
    """Initialize a linear weight matrix with N(0, std)."""
    std = float(std)
    if std < 0.0:
        raise ValueError(f"Linear init std must be >= 0, got {std}.")
    if std == 0.0:
        weight.zero_()
        return

    tmp = torch.empty_like(weight, dtype=torch.float32)
    nn.init.normal_(tmp, mean=0.0, std=std)
    weight.copy_(tmp.to(dtype=weight.dtype))


@torch.no_grad()
def init_embedding_weight_(weight: torch.Tensor, mode: str, std: float) -> None:
    """Initialize an embedding table with zeros or N(0, std)."""
    mode = str(mode).strip().lower()
    std = float(std)

    if mode == "zeros":
        weight.zero_()
        return

    if mode == "normal":
        if std < 0.0:
            raise ValueError(f"Embedding init std must be >= 0, got {std}.")
        if std == 0.0:
            weight.zero_()
            return

        tmp = torch.empty_like(weight, dtype=torch.float32)
        nn.init.normal_(tmp, mean=0.0, std=std)
        weight.copy_(tmp.to(dtype=weight.dtype))
        return

    raise ValueError(f"Unknown table_init_mode: {mode}.")


def pack_bits_to_route_codes(bits_btc: torch.Tensor, bits_per_route: int) -> torch.Tensor:
    """
    Convert binary route bits into exact route codes.

    Args:
        bits_btc:
            Tensor of shape [batch, time, channels] with values in {0, 1}.
        bits_per_route:
            Number of binary channels grouped into one route code.

    Returns:
        Tensor of shape [batch, time, num_routes] with integer route codes.
    """
    batch, time, channels = bits_btc.shape
    if channels % bits_per_route != 0:
        raise ValueError(
            f"channels={channels} must be divisible by bits_per_route={bits_per_route}."
        )

    num_routes = channels // bits_per_route
    x = bits_btc.view(batch, time, num_routes, bits_per_route).to(torch.int32)

    out = torch.zeros((batch, time, num_routes), dtype=torch.int32, device=bits_btc.device)
    for bit_idx in range(bits_per_route):
        out |= ((x[..., bit_idx] & 1) << bit_idx)
    return out


def chunked_linear_lastdim(
    x: torch.Tensor,
    linear: nn.Linear,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """
    Apply an exactly equivalent linear projection in chunks along the input dimension.

    This is mathematically equivalent to `F.linear(x, linear.weight, linear.bias)`,
    but can reduce peak memory for large flattened suffix states.
    """
    if x.dim() < 2:
        raise ValueError(f"x must have dim >= 2, got shape={tuple(x.shape)}.")

    in_features = int(linear.in_features)
    out_features = int(linear.out_features)

    if x.shape[-1] != in_features:
        raise ValueError(
            f"Input last dim mismatch: got {x.shape[-1]}, expected {in_features}."
        )

    if chunk_size is None:
        return F.linear(x, linear.weight, linear.bias)

    chunk_size = int(chunk_size)
    if chunk_size <= 0 or chunk_size >= in_features:
        return F.linear(x, linear.weight, linear.bias)

    orig_prefix_shape = x.shape[:-1]
    x2d = x.reshape(-1, in_features)

    out = None
    for start in range(0, in_features, chunk_size):
        end = min(in_features, start + chunk_size)
        part = F.linear(x2d[:, start:end], linear.weight[:, start:end], bias=None)
        out = part if out is None else out + part

    if linear.bias is not None:
        out = out + linear.bias

    return out.view(*orig_prefix_shape, out_features)


class RMSNorm(nn.Module):
    """RMSNorm implementation used by the lngram module."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DepthwiseCausalConv(nn.Module):
    """
    Short-range depthwise causal convolution branch.

    The module returns only the convolutional branch output. Residual addition
    is handled by the caller.
    """

    def __init__(
        self,
        hidden_size: int,
        num_branches: int,
        kernel_size: int,
        dilation: int,
        eps: float,
        bias: bool = False,
        zero_init: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_branches = int(num_branches)

        channels = self.hidden_size * self.num_branches
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=int(kernel_size),
            groups=channels,
            padding=(int(kernel_size) - 1) * int(dilation),
            dilation=int(dilation),
            bias=bool(bias),
        )
        self.act = nn.SiLU()
        self.norms = nn.ModuleList(
            [RMSNorm(self.hidden_size, eps=eps) for _ in range(self.num_branches)]
        )

        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, time, branches, hidden_size].

        Returns:
            Tensor of shape [batch, time, branches, hidden_size].
        """
        if x.dim() != 4:
            raise ValueError(f"DepthwiseCausalConv expects [B, T, M, D], got {tuple(x.shape)}.")

        batch, time, num_branches, hidden_size = x.shape
        if num_branches != self.num_branches or hidden_size != self.hidden_size:
            raise ValueError(
                f"Shape mismatch: got [B={batch}, T={time}, M={num_branches}, D={hidden_size}], "
                f"expected M={self.num_branches}, D={self.hidden_size}."
            )

        normed = [self.norms[idx](x[:, :, idx, :]) for idx in range(num_branches)]
        x_cat = torch.cat(normed, dim=-1)

        y = self.conv(x_cat.transpose(1, 2))
        y = y[..., :time]
        y = self.act(y)

        y = y.transpose(1, 2).contiguous().view(batch, time, num_branches, hidden_size)
        return y


def _exact_local_surrogate_score_from_p(
    grad_valid: torch.Tensor,
    p_local: torch.Tensor,
    addr_base: torch.Tensor,
    table_weight: torch.Tensor,
    route_offsets: torch.Tensor,
    code_stride: int,
    temp: float,
    code_bits: torch.Tensor,
    accum_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Exact local surrogate gradient for a small discrete code space.

    This implementation assumes `alphabet_size = 2 ** bits_per_route`, which is
    typically 16 when `bits_per_route == 4`.
    """
    if grad_valid.dtype != accum_dtype:
        grad_valid = grad_valid.to(accum_dtype)
    if p_local.dtype != accum_dtype:
        p_local = p_local.to(accum_dtype)

    one_minus_p = 1.0 - p_local
    _, _, _, num_bits = p_local.shape
    num_codes = int(code_bits.shape[0])

    local_score = torch.zeros_like(p_local, dtype=accum_dtype)

    for code in range(num_codes):
        bits_code = code_bits[code].view(1, 1, 1, num_bits)
        prob_code = torch.where(bits_code.bool(), p_local, one_minus_p).prod(dim=-1)

        idx_code = route_offsets + addr_base + (int(code) * int(code_stride))
        emb_code = F.embedding(idx_code, table_weight)
        if emb_code.dtype != accum_dtype:
            emb_code = emb_code.to(accum_dtype)

        dot_code = (grad_valid * emb_code).sum(dim=-1)
        local_score = local_score + prob_code.unsqueeze(-1) * (bits_code - p_local) * dot_code.unsqueeze(-1)

    return float(temp) * local_score


class LngramQSurrogateFunction(torch.autograd.Function):
    """
    Forward uses hard route codes. Backward sends an exact local surrogate
    gradient to the route tokenizer logits.
    """

    @staticmethod
    def forward(
        ctx,
        flat_memory: torch.Tensor,
        q_logits: torch.Tensor,
        q_codes_btr: torch.Tensor,
        memory_module: "RouteNgramMemory",
    ):
        ctx.save_for_backward(q_logits, q_codes_btr.to(torch.int16))
        ctx.memory_module = memory_module
        return flat_memory

    @staticmethod
    def backward(ctx, grad_flat_memory: torch.Tensor):
        q_logits, q_codes_saved = ctx.saved_tensors
        memory = ctx.memory_module

        if (grad_flat_memory is None) or (not memory.q_surrogate_enable):
            return grad_flat_memory, None, None, None

        batch, time, channels = q_logits.shape
        bits_per_route = int(memory.bits_per_route)
        num_routes = int(memory.num_routes)
        memory_dim = int(memory.mem_dim)
        alphabet_size = int(memory.alphabet_size)
        num_orders = int(len(memory.ngrams))

        if channels != num_routes * bits_per_route:
            raise ValueError(
                f"channels={channels}, num_routes={num_routes}, bits_per_route={bits_per_route} mismatch."
            )

        device = q_logits.device
        accum_dtype = torch.float32 if memory.q_surrogate_accum_fp32 else grad_flat_memory.dtype

        temp = float(memory.q_surrogate_temp)
        grad_scale = float(memory.q_surrogate_scale)
        route_chunk = max(1, int(memory.q_surrogate_route_chunk_size))

        grad_flat = grad_flat_memory.reshape(batch, time, num_orders, num_routes, memory_dim)
        q_logits_btrm = q_logits.reshape(batch, time, num_routes, bits_per_route)
        q_codes_btr = q_codes_saved.to(torch.long)

        grad_q = torch.zeros((batch, time, num_routes, bits_per_route), device=device, dtype=accum_dtype)
        route_ids_full = torch.arange(num_routes, device=device, dtype=torch.long)

        bit_positions = torch.arange(bits_per_route, device=device, dtype=torch.long)
        code_bits = (
            (torch.arange(alphabet_size, device=device, dtype=torch.long).unsqueeze(-1)
             >> bit_positions.unsqueeze(0)) & 1
        ).to(accum_dtype)

        for route_start in range(0, num_routes, route_chunk):
            route_end = min(num_routes, route_start + route_chunk)
            route_count = route_end - route_start

            codes_chunk = q_codes_btr[:, :, route_start:route_end]
            q_chunk = q_logits_btrm[:, :, route_start:route_end, :]
            p_chunk = torch.sigmoid(temp * q_chunk.to(accum_dtype))

            grad_q_chunk = grad_q[:, :, route_start:route_end, :]

            for ngram_index, ngram_order in enumerate(memory.ngrams):
                ngram_order = int(ngram_order)
                valid_length = time - ngram_order + 1
                if valid_length <= 0:
                    continue

                table_weight = memory.tables[f"ngram_{ngram_order}"].weight.detach()
                route_vocab = alphabet_size ** ngram_order
                route_offsets = route_ids_full[route_start:route_end].view(1, 1, route_count) * route_vocab

                addr = torch.zeros((batch, valid_length, route_count), device=device, dtype=torch.long)
                stride = 1
                for exp in range(ngram_order):
                    addr = addr + codes_chunk[:, exp:exp + valid_length, :] * stride
                    stride *= alphabet_size

                grad_valid = grad_flat[:, ngram_order - 1:, ngram_index, route_start:route_end, :]
                if grad_valid.dtype != accum_dtype:
                    grad_valid = grad_valid.to(accum_dtype)

                for exp in range(ngram_order):
                    t_lo = exp
                    t_hi = exp + valid_length

                    current_codes = codes_chunk[:, t_lo:t_hi, :]
                    addr_base = addr - current_codes * (alphabet_size ** exp)

                    local_score = _exact_local_surrogate_score_from_p(
                        grad_valid=grad_valid,
                        p_local=p_chunk[:, t_lo:t_hi, :, :],
                        addr_base=addr_base,
                        table_weight=table_weight,
                        route_offsets=route_offsets,
                        code_stride=(alphabet_size ** exp),
                        temp=temp,
                        code_bits=code_bits,
                        accum_dtype=accum_dtype,
                    )

                    grad_q_chunk[:, t_lo:t_hi, :, :] += grad_scale * local_score

        grad_q = grad_q.reshape(batch, time, channels).to(q_logits.dtype)
        return grad_flat_memory, grad_q, None, None


class LngramQSurrogateChunkFunction(torch.autograd.Function):
    """
    Chunk-level exact local surrogate gradient used by the streaming lookup path.
    """

    @staticmethod
    def forward(
        ctx,
        emb_chunk: torch.Tensor,
        q_logits_chunk: torch.Tensor,
        q_codes_chunk: torch.Tensor,
        memory_module: "RouteNgramMemory",
        ngram_order: int,
        route_offset: int,
    ):
        ctx.save_for_backward(q_logits_chunk, q_codes_chunk.to(torch.int16))
        ctx.memory_module = memory_module
        ctx.ngram_order = int(ngram_order)
        ctx.route_offset = int(route_offset)
        return emb_chunk

    @staticmethod
    def backward(ctx, grad_emb_chunk: torch.Tensor):
        q_logits_chunk, q_codes_saved = ctx.saved_tensors
        memory = ctx.memory_module
        ngram_order = int(ctx.ngram_order)
        route_offset = int(ctx.route_offset)

        if (grad_emb_chunk is None) or (not memory.q_surrogate_enable):
            return grad_emb_chunk, None, None, None, None, None

        batch, time, route_count, bits_per_route = q_logits_chunk.shape
        alphabet_size = int(memory.alphabet_size)

        device = q_logits_chunk.device
        accum_dtype = torch.float32 if memory.q_surrogate_accum_fp32 else grad_emb_chunk.dtype

        temp = float(memory.q_surrogate_temp)
        grad_scale = float(memory.q_surrogate_scale)

        q_codes_chunk = q_codes_saved.to(torch.long)
        p_chunk = torch.sigmoid(temp * q_logits_chunk.to(accum_dtype))

        grad_q_chunk = torch.zeros((batch, time, route_count, bits_per_route), device=device, dtype=accum_dtype)

        valid_length = time - ngram_order + 1
        if valid_length > 0:
            table_weight = memory.tables[f"ngram_{ngram_order}"].weight.detach()
            route_vocab = alphabet_size ** ngram_order

            route_ids = torch.arange(
                route_offset,
                route_offset + route_count,
                device=device,
                dtype=torch.long,
            ).view(1, 1, route_count)
            route_offsets = route_ids * route_vocab

            bit_positions = torch.arange(bits_per_route, device=device, dtype=torch.long)
            code_bits = (
                (torch.arange(alphabet_size, device=device, dtype=torch.long).unsqueeze(-1)
                 >> bit_positions.unsqueeze(0)) & 1
            ).to(accum_dtype)

            addr = torch.zeros((batch, valid_length, route_count), device=device, dtype=torch.long)
            stride = 1
            for exp in range(ngram_order):
                addr = addr + q_codes_chunk[:, exp:exp + valid_length, :] * stride
                stride *= alphabet_size

            grad_valid = grad_emb_chunk[:, ngram_order - 1:, :, :]
            if grad_valid.dtype != accum_dtype:
                grad_valid = grad_valid.to(accum_dtype)

            for exp in range(ngram_order):
                t_lo = exp
                t_hi = exp + valid_length

                current_codes = q_codes_chunk[:, t_lo:t_hi, :]
                addr_base = addr - current_codes * (alphabet_size ** exp)

                local_score = _exact_local_surrogate_score_from_p(
                    grad_valid=grad_valid,
                    p_local=p_chunk[:, t_lo:t_hi, :, :],
                    addr_base=addr_base,
                    table_weight=table_weight,
                    route_offsets=route_offsets,
                    code_stride=(alphabet_size ** exp),
                    temp=temp,
                    code_bits=code_bits,
                    accum_dtype=accum_dtype,
                )

                grad_q_chunk[:, t_lo:t_hi, :, :] += grad_scale * local_score

        return grad_emb_chunk, grad_q_chunk.to(q_logits_chunk.dtype), None, None, None, None


class RouteNgramMemory(nn.Module):
    """
    Exact route-level n-gram memory with shared key/value projections across
    n-gram orders.
    """

    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int,
        memory_dim: int,
        ngrams: Tuple[int, ...] = (2, 3),
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        table_init_mode: str = "zeros",
        table_init_std_scale: float = 1.0,
        output_proj_init_std_scale: float = 1.0,
        rmsnorm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        conv_bias: bool = False,
        conv_zero_init: bool = True,
        proj_chunk_size: Optional[int] = None,
        q_surrogate_enable: bool = True,
        q_surrogate_temp: float = 1.0,
        q_surrogate_scale: float = 1.0,
        q_surrogate_route_chunk_size: int = 256,
        q_surrogate_accum_fp32: bool = True,
    ):
        super().__init__()
        if hidden_size % bits_per_route != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by bits_per_route={bits_per_route}."
            )
        if len(ngrams) == 0:
            raise ValueError("ngrams must not be empty.")
        if any(int(n) < 1 for n in ngrams):
            raise ValueError(f"All n-gram orders must be >= 1, got {ngrams}.")

        self.hidden_size = int(hidden_size)
        self.bits_per_route = int(bits_per_route)
        self.num_routes = self.hidden_size // self.bits_per_route
        self.alphabet_size = 1 << self.bits_per_route
        self.mem_dim = int(memory_dim)

        self.ngrams = tuple(sorted(set(int(n) for n in ngrams)))
        self.num_ngram_orders = len(self.ngrams)

        self.initializer_range = float(initializer_range)
        self.table_init_mode = str(table_init_mode)
        self.table_init_std_scale = float(table_init_std_scale)
        self.output_proj_init_std_scale = float(output_proj_init_std_scale)

        self.q_surrogate_enable = bool(q_surrogate_enable)
        self.q_surrogate_temp = float(q_surrogate_temp)
        self.q_surrogate_scale = float(q_surrogate_scale)
        self.q_surrogate_route_chunk_size = int(q_surrogate_route_chunk_size)
        self.q_surrogate_accum_fp32 = bool(q_surrogate_accum_fp32)

        self.proj_chunk_size = None if proj_chunk_size is None else int(proj_chunk_size)

        table_std = self.initializer_range * self.table_init_std_scale
        self.tables = nn.ModuleDict()
        for ngram_order in self.ngrams:
            vocab_size = self.num_routes * (self.alphabet_size ** ngram_order)
            emb = nn.Embedding(vocab_size, self.mem_dim)
            init_embedding_weight_(emb.weight, mode=self.table_init_mode, std=table_std)
            self.tables[f"ngram_{ngram_order}"] = emb

        self.lookup_dropout = nn.Dropout(float(dropout))

        self.suffix_flat_dim = self.num_routes * self.mem_dim
        self.flat_dim = self.num_ngram_orders * self.suffix_flat_dim

        proj_std = self.initializer_range * self.output_proj_init_std_scale

        self.value_proj = nn.Linear(self.suffix_flat_dim, self.hidden_size, bias=True)
        self.key_proj = nn.Linear(self.suffix_flat_dim, self.hidden_size, bias=True)

        init_linear_weight_(self.value_proj.weight, std=proj_std)
        init_linear_weight_(self.key_proj.weight, std=proj_std)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.key_proj.bias)

        self.query_norm = RMSNorm(self.hidden_size, eps=float(rmsnorm_eps))
        self.key_norm = RMSNorm(self.hidden_size, eps=float(rmsnorm_eps))

        dilation = int(conv_dilation) if conv_dilation is not None else int(max(self.ngrams))
        self.short_conv = DepthwiseCausalConv(
            hidden_size=self.hidden_size,
            num_branches=1,
            kernel_size=int(conv_kernel_size),
            dilation=dilation,
            eps=float(rmsnorm_eps),
            bias=bool(conv_bias),
            zero_init=bool(conv_zero_init),
        )

        self.inv_sqrt_hidden = 1.0 / math.sqrt(self.hidden_size)

    def _build_global_indices(
        self,
        route_codes_btr: torch.Tensor,
        ngram_order: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build global table indices for one n-gram order.

        Returns:
            global_idx: [B, T, R]
            valid:      [B, T, R]
        """
        if route_codes_btr.dtype != torch.long:
            route_codes_btr = route_codes_btr.long()

        batch, time, num_routes = route_codes_btr.shape
        device = route_codes_btr.device

        global_idx = torch.zeros((batch, time, num_routes), dtype=torch.long, device=device)
        valid = torch.zeros((batch, time, num_routes), dtype=torch.bool, device=device)

        if time < ngram_order:
            return global_idx, valid

        addr = torch.zeros((batch, time - ngram_order + 1, num_routes), dtype=torch.long, device=device)
        stride = 1
        for exp in range(ngram_order):
            addr = addr + route_codes_btr[:, exp:exp + (time - ngram_order + 1), :] * stride
            stride *= self.alphabet_size

        route_ids = torch.arange(0, num_routes, dtype=torch.long, device=device).view(1, 1, num_routes)
        route_vocab = self.alphabet_size ** ngram_order

        global_idx[:, ngram_order - 1:, :] = route_ids * route_vocab + addr
        valid[:, ngram_order - 1:, :] = True
        return global_idx, valid

    def lookup(self, route_codes_btr: torch.Tensor) -> torch.Tensor:
        """
        Full lookup path.

        Returns:
            Flat memory tensor of shape [B, T, num_orders * num_routes * memory_dim].
        """
        if route_codes_btr.dtype != torch.long:
            route_codes_btr = route_codes_btr.long()

        batch, time, num_routes = route_codes_btr.shape
        if num_routes != self.num_routes:
            raise ValueError(f"num_routes={num_routes}, expected {self.num_routes}.")

        table_acc_dtype = torch.float32
        memory_list = []

        for ngram_order in self.ngrams:
            table = self.tables[f"ngram_{ngram_order}"]
            global_idx, valid = self._build_global_indices(route_codes_btr, ngram_order=ngram_order)

            emb = table(global_idx).to(table_acc_dtype)
            emb = emb * valid.unsqueeze(-1).to(table_acc_dtype)
            memory_list.append(emb)

        memory_cat = torch.stack(memory_list, dim=2)
        memory_cat = self.lookup_dropout(memory_cat)

        flat = memory_cat.reshape(batch, time, self.flat_dim)
        return flat.to(dtype=self.value_proj.weight.dtype)

    def _reshape_flat_to_suffix_memory(self, flat_memory: torch.Tensor) -> torch.Tensor:
        """
        Reshape [B, T, num_orders * num_routes * memory_dim] to
        [B, T, num_orders, num_routes * memory_dim].
        """
        if flat_memory.dim() != 3:
            raise ValueError(f"flat_memory must be [B, T, F], got {tuple(flat_memory.shape)}.")
        if flat_memory.shape[-1] != self.flat_dim:
            raise ValueError(
                f"flat_memory dim mismatch: got {flat_memory.shape[-1]}, expected {self.flat_dim}."
            )

        batch, time, _ = flat_memory.shape
        return flat_memory.view(
            batch,
            time,
            self.num_ngram_orders,
            self.num_routes,
            self.mem_dim,
        ).reshape(batch, time, self.num_ngram_orders, self.suffix_flat_dim)

    def inject(self, hidden_states: torch.Tensor, flat_memory: torch.Tensor) -> torch.Tensor:
        """
        Fallback injection path that materializes the flat memory tensor.
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, C], got {tuple(hidden_states.shape)}.")
        if flat_memory.dim() != 3:
            raise ValueError(f"flat_memory must be [B, T, F], got {tuple(flat_memory.shape)}.")

        batch, time, channels = hidden_states.shape
        if channels != self.hidden_size:
            raise ValueError(
                f"hidden_size mismatch: got channels={channels}, expected {self.hidden_size}."
            )
        if flat_memory.shape[:2] != (batch, time):
            raise ValueError(
                f"Shape mismatch: hidden_states={tuple(hidden_states.shape)} "
                f"vs flat_memory={tuple(flat_memory.shape)}."
            )

        suffix_memory = self._reshape_flat_to_suffix_memory(flat_memory).to(dtype=self.value_proj.weight.dtype)

        value = chunked_linear_lastdim(suffix_memory, self.value_proj, self.proj_chunk_size)
        key = chunked_linear_lastdim(suffix_memory, self.key_proj, self.proj_chunk_size)

        query = self.query_norm(hidden_states).unsqueeze(2)
        key = self.key_norm(key)

        gate_logits = (query.to(torch.float32) * key.to(torch.float32)).sum(dim=-1) * self.inv_sqrt_hidden
        alpha = torch.sigmoid(gate_logits).to(value.dtype).unsqueeze(-1)

        mixed = (alpha * value).sum(dim=2)
        conv_out = self.short_conv(mixed.unsqueeze(2)).squeeze(2)
        return mixed + conv_out

    def _can_use_streaming_lookup_project(self) -> bool:
        if self.proj_chunk_size is None:
            return False

        if self.training and float(self.lookup_dropout.p) > 0.0:
            return False

        if self.proj_chunk_size % self.mem_dim != 0:
            raise ValueError(
                f"proj_chunk_size={self.proj_chunk_size} must be divisible by memory_dim={self.mem_dim}."
            )

        return True

    def _stream_route_chunk_size(self) -> int:
        return max(1, int(self.proj_chunk_size // self.mem_dim))

    def _proj_slice_for_route_chunk(self, route_start: int, route_end: int) -> Tuple[int, int]:
        start = route_start * self.mem_dim
        end = route_end * self.mem_dim
        return start, end

    def _lookup_single_ngram_route_chunk(
        self,
        route_codes_btr: torch.Tensor,
        ngram_order: int,
        route_start: int,
        route_end: int,
        out_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Lookup a single n-gram order for a chunk of route positions.

        Returns:
            Tensor of shape [B, T, route_chunk, memory_dim].
        """
        if route_codes_btr.dtype != torch.long:
            route_codes_btr = route_codes_btr.long()

        batch, time, num_routes = route_codes_btr.shape
        if num_routes != self.num_routes:
            raise ValueError(f"num_routes={num_routes}, expected {self.num_routes}.")
        if not (0 <= route_start < route_end <= self.num_routes):
            raise ValueError(
                f"Invalid route chunk: route_start={route_start}, route_end={route_end}, "
                f"num_routes={self.num_routes}."
            )

        route_count = route_end - route_start
        device = route_codes_btr.device
        table = self.tables[f"ngram_{ngram_order}"]

        out = torch.zeros((batch, time, route_count, self.mem_dim), dtype=out_dtype, device=device)

        if time < ngram_order:
            return out

        valid_length = time - ngram_order + 1
        codes_chunk = route_codes_btr[:, :, route_start:route_end]

        addr = torch.zeros((batch, valid_length, route_count), dtype=torch.long, device=device)
        stride = 1
        for exp in range(ngram_order):
            addr = addr + codes_chunk[:, exp:exp + valid_length, :] * stride
            stride *= self.alphabet_size

        route_ids = torch.arange(route_start, route_end, dtype=torch.long, device=device).view(1, 1, route_count)
        route_vocab = self.alphabet_size ** ngram_order
        global_idx = route_ids * route_vocab + addr

        out[:, ngram_order - 1:, :, :] = table(global_idx).to(out_dtype)
        return out

    def inject_from_route_codes(
        self,
        hidden_states: torch.Tensor,
        route_codes_btr: torch.Tensor,
        q_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Streaming lookup + direct projection accumulation path.

        Falls back to the full lookup path if chunked projection is disabled.
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, C], got {tuple(hidden_states.shape)}.")
        if route_codes_btr.dim() != 3:
            raise ValueError(f"route_codes_btr must be [B, T, R], got {tuple(route_codes_btr.shape)}.")

        batch, time, channels = hidden_states.shape
        num_routes = route_codes_btr.shape[-1]

        if channels != self.hidden_size:
            raise ValueError(
                f"hidden_size mismatch: got channels={channels}, expected {self.hidden_size}."
            )
        if num_routes != self.num_routes:
            raise ValueError(f"num_routes mismatch: got {num_routes}, expected {self.num_routes}.")

        if not self._can_use_streaming_lookup_project():
            flat_memory = self.lookup(route_codes_btr)
            if (q_logits is not None) and self.q_surrogate_enable:
                flat_memory = LngramQSurrogateFunction.apply(
                    flat_memory,
                    q_logits,
                    route_codes_btr,
                    self,
                )
            return self.inject(hidden_states, flat_memory)

        proj_dtype = self.value_proj.weight.dtype
        device = hidden_states.device
        route_chunk = self._stream_route_chunk_size()

        q_logits_btrm = None
        if (q_logits is not None) and self.q_surrogate_enable:
            if q_logits.shape != (batch, time, self.hidden_size):
                raise ValueError(
                    f"q_logits shape mismatch: got {tuple(q_logits.shape)}, "
                    f"expected {(batch, time, self.hidden_size)}."
                )
            q_logits_btrm = q_logits.view(batch, time, self.num_routes, self.bits_per_route)

        suffix_values = []
        suffix_keys = []

        for ngram_order in self.ngrams:
            value_sum = torch.zeros((batch, time, self.hidden_size), dtype=proj_dtype, device=device)
            key_sum = torch.zeros((batch, time, self.hidden_size), dtype=proj_dtype, device=device)

            for route_start in range(0, self.num_routes, route_chunk):
                route_end = min(self.num_routes, route_start + route_chunk)
                route_count = route_end - route_start

                emb_chunk = self._lookup_single_ngram_route_chunk(
                    route_codes_btr=route_codes_btr,
                    ngram_order=int(ngram_order),
                    route_start=int(route_start),
                    route_end=int(route_end),
                    out_dtype=torch.float32,
                ).to(dtype=proj_dtype)

                if q_logits_btrm is not None:
                    emb_chunk = LngramQSurrogateChunkFunction.apply(
                        emb_chunk,
                        q_logits_btrm[:, :, route_start:route_end, :],
                        route_codes_btr[:, :, route_start:route_end],
                        self,
                        int(ngram_order),
                        int(route_start),
                    )

                proj_start, proj_end = self._proj_slice_for_route_chunk(
                    route_start=route_start,
                    route_end=route_end,
                )

                emb_flat = emb_chunk.reshape(batch, time, route_count * self.mem_dim)
                value_sum = value_sum + F.linear(emb_flat, self.value_proj.weight[:, proj_start:proj_end], None)
                key_sum = key_sum + F.linear(emb_flat, self.key_proj.weight[:, proj_start:proj_end], None)

            if self.value_proj.bias is not None:
                value_sum = value_sum + self.value_proj.bias
            if self.key_proj.bias is not None:
                key_sum = key_sum + self.key_proj.bias

            suffix_values.append(value_sum)
            suffix_keys.append(key_sum)

        value = torch.stack(suffix_values, dim=2)
        key = torch.stack(suffix_keys, dim=2)

        query = self.query_norm(hidden_states).unsqueeze(2)
        key = self.key_norm(key)

        gate_logits = (query.to(torch.float32) * key.to(torch.float32)).sum(dim=-1) * self.inv_sqrt_hidden
        alpha = torch.sigmoid(gate_logits).to(value.dtype).unsqueeze(-1)

        mixed = (alpha * value).sum(dim=2)
        conv_out = self.short_conv(mixed.unsqueeze(2)).squeeze(2)
        return mixed + conv_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        route_codes_btr: torch.Tensor,
        q_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.inject_from_route_codes(
            hidden_states=hidden_states,
            route_codes_btr=route_codes_btr,
            q_logits=q_logits,
        )


def estimate_lngram_parameters_per_layer(
    hidden_size: int,
    bits_per_route: int,
    memory_dim: int,
    ngrams: Tuple[int, ...],
) -> int:
    """Estimate additional parameters introduced by one lngram-enabled layer."""
    if hidden_size % bits_per_route != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by bits_per_route={bits_per_route}."
        )

    num_routes = hidden_size // bits_per_route
    alphabet_size = 1 << bits_per_route

    q_proj_params = hidden_size * hidden_size
    table_params = num_routes * memory_dim * sum(alphabet_size ** int(n) for n in ngrams)

    suffix_flat_dim = num_routes * memory_dim
    value_proj_params = suffix_flat_dim * hidden_size + hidden_size
    key_proj_params = suffix_flat_dim * hidden_size + hidden_size

    query_norm_params = hidden_size
    key_norm_params = hidden_size

    conv_params = hidden_size * 4

    return (
        q_proj_params
        + table_params
        + value_proj_params
        + key_proj_params
        + query_norm_params
        + key_norm_params
        + conv_params
    )


def collect_lngram_table_parameter_names(model: nn.Module) -> set[str]:
    """Return parameter names belonging to lngram tables."""
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lngram_memory.tables." in name
    }


def build_lngram_optimizer_param_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    decay_parameter_names: Iterable[str],
    lngram_config: LngramConfig,
) -> list[dict]:
    """
    Build optimizer parameter groups with a dedicated rule for lngram tables.

    This keeps the original training policy:
    - lngram tables use `learning_rate * table_lr_multiplier`
    - lngram tables use `table_weight_decay`
    - all remaining parameters follow the standard decay / no-decay split
    """
    decay_parameter_names = set(decay_parameter_names)
    named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    table_names = collect_lngram_table_parameter_names(model)

    table_params = [param for name, param in named_parameters if name in table_names]
    decay_params = [
        param
        for name, param in named_parameters
        if (name not in table_names) and (name in decay_parameter_names)
    ]
    nodecay_params = [
        param
        for name, param in named_parameters
        if (name not in table_names) and (name not in decay_parameter_names)
    ]

    param_groups: list[dict] = []

    if table_params:
        param_groups.append(
            {
                "params": table_params,
                "lr": float(learning_rate) * float(lngram_config.table_lr_multiplier),
                "weight_decay": float(lngram_config.table_weight_decay),
            }
        )

    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )

    if nodecay_params:
        param_groups.append(
            {
                "params": nodecay_params,
                "lr": float(learning_rate),
                "weight_decay": 0.0,
            }
        )

    return param_groups


def patch_model_with_lngram(model: nn.Module, lngram_config: LngramConfig) -> nn.Module:
    """
    Add lngram modules to selected decoder layers without changing the base
    attention or MLP implementations.

    Assumptions:
    - `model.config.hidden_size` exists
    - `model.config.rms_norm_eps` exists or defaults to 1e-6
    - `model.model.layers` is an iterable of decoder layers
    - each patched layer exposes:
        * input_layernorm
        * self_attn
        * post_attention_layernorm
        * mlp

    The injection point is the pre-attention residual stream.
    """
    if not lngram_config.enabled:
        return model

    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise AttributeError("The provided model does not expose `model.layers` or `layers`.")

    hidden_size = int(getattr(model.config, "hidden_size"))
    lngram_config.validate(hidden_size=hidden_size, num_hidden_layers=len(layers))

    first_param = next(model.parameters())
    base_dtype = first_param.dtype
    base_device = first_param.device
    init_std = resolve_added_module_init_std(model.config)
    rmsnorm_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))

    active_layers = {int(idx) for idx in lngram_config.target_layers}

    for layer_index, layer in enumerate(layers):
        if layer_index not in active_layers:
            layer.lngram_q_proj = None
            layer.lngram_memory = None
            layer.lngram_enabled = False
            continue

        q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        init_linear_weight_(
            q_proj.weight,
            std=init_std * float(lngram_config.q_proj_init_std_scale),
        )
        layer.lngram_q_proj = q_proj.to(dtype=base_dtype, device=base_device)

        conv_dilation = (
            int(lngram_config.conv_dilation)
            if lngram_config.conv_dilation is not None
            else max(int(n) for n in lngram_config.ngrams)
        )

        lngram_memory = RouteNgramMemory(
            hidden_size=hidden_size,
            bits_per_route=lngram_config.bits_per_route,
            memory_dim=lngram_config.memory_dim,
            ngrams=tuple(int(n) for n in lngram_config.ngrams),
            dropout=lngram_config.dropout,
            initializer_range=init_std,
            table_init_mode=lngram_config.table_init_mode,
            table_init_std_scale=lngram_config.table_init_std_scale,
            output_proj_init_std_scale=lngram_config.output_proj_init_std_scale,
            rmsnorm_eps=rmsnorm_eps,
            conv_kernel_size=lngram_config.conv_kernel_size,
            conv_dilation=conv_dilation,
            conv_bias=lngram_config.conv_bias,
            conv_zero_init=lngram_config.conv_zero_init,
            proj_chunk_size=lngram_config.proj_chunk_size,
            q_surrogate_enable=lngram_config.q_surrogate_enable,
            q_surrogate_temp=lngram_config.q_surrogate_temp,
            q_surrogate_scale=lngram_config.q_surrogate_scale,
            q_surrogate_route_chunk_size=lngram_config.q_surrogate_route_chunk_size,
            q_surrogate_accum_fp32=lngram_config.q_surrogate_accum_fp32,
        )
        layer.lngram_memory = lngram_memory.to(dtype=base_dtype, device=base_device)
        layer.lngram_enabled = True

        def _forward_with_lngram(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
        ):
            if "padding_mask" in kwargs:
                warnings.warn(
                    "Passing `padding_mask` is deprecated. Please use `attention_mask` instead."
                )

            if (
                getattr(self, "lngram_enabled", False)
                and (getattr(self, "lngram_memory", None) is not None)
                and (getattr(self, "lngram_q_proj", None) is not None)
            ):
                base_states = self.input_layernorm(hidden_states)
                q_logits = self.lngram_q_proj(base_states)

                q_bits = (q_logits > 0).to(torch.int32)
                q_codes = pack_bits_to_route_codes(
                    q_bits,
                    lngram_config.bits_per_route,
                )

                lngram_injection = self.lngram_memory.inject_from_route_codes(
                    hidden_states=hidden_states,
                    route_codes_btr=q_codes,
                    q_logits=q_logits if lngram_config.q_surrogate_enable else None,
                ).to(hidden_states.dtype)

                hidden_states = hidden_states + lngram_injection

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual.to(hidden_states.dtype) + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual.to(hidden_states.dtype) + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs

        layer.forward = types.MethodType(_forward_with_lngram, layer)

    return model


__all__ = [
    "DepthwiseCausalConv",
    "LngramConfig",
    "LngramQSurrogateChunkFunction",
    "LngramQSurrogateFunction",
    "RMSNorm",
    "RouteNgramMemory",
    "build_lngram_optimizer_param_groups",
    "chunked_linear_lastdim",
    "collect_lngram_table_parameter_names",
    "estimate_lngram_parameters_per_layer",
    "init_embedding_weight_",
    "init_linear_weight_",
    "pack_bits_to_route_codes",
    "patch_model_with_lngram",
    "resolve_added_module_init_std",
]
