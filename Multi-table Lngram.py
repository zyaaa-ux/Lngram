


from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers import Trainer
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    from transformers.trainer_pt_utils import get_parameter_names
except ImportError:  # pragma: no cover
    Trainer = None
    ALL_LAYERNORM_LAYERS = []

    def get_parameter_names(*args, **kwargs):
        raise ImportError("transformers is required to use LngramTrainer.")


@dataclass
class LngramConfig:
    enabled: bool = True
    layers: tuple[int, ...] = (1, 11)
    bits_per_route: int = 4
    ngrams: tuple[int, ...] = (2, 3)
    num_subtables: int = 6
    mem_dim: int = 16
    dropout: float = 0.0

    q_proj_init_std_scale: float = 1.0
    table_init_mode: str = "zeros"
    table_init_std_scale: float = 1.0
    out_proj_init_std_scale: float = 1.0

    surrogate_enable: bool = True
    surrogate_temp: float = 1.0
    surrogate_scale: float = 1.0
    surrogate_route_chunk_size: int = 128
    surrogate_accum_fp32: bool = True

    surrogate_exact_enable: bool = True
    surrogate_exact_logit_threshold: float = 1.0
    surrogate_exact_entropy_threshold: float = -1.0
    surrogate_exact_max_positions: int = 4096

    fusion_temperature: float = 1.0
    null_logit: Optional[float] = None
    learnable_null_logit: bool = True
    null_logit_bound: Optional[float] = 8.0

    conv_kernel_size: int = 4
    conv_dilation: Optional[int] = None
    subtable_chunk_size: Optional[int] = None

    def normalized_layers(self) -> tuple[int, ...]:
        return tuple(sorted(set(int(index) for index in self.layers)))

    def normalized_ngrams(self) -> tuple[int, ...]:
        values = tuple(sorted(set(int(n) for n in self.ngrams)))
        if not values:
            raise ValueError("ngrams must not be empty.")
        if any(n <= 0 for n in values):
            raise ValueError(f"ngrams must be positive integers, got {values}.")
        return values


@dataclass
class LngramOptimizerConfig:
    table_lr_multiplier: float = 3.0
    table_weight_decay: float = 0.0
    null_logit_lr_multiplier: float = 0.25
    null_logit_weight_decay: float = 0.0


def coerce_lngram_config(config: LngramConfig | dict[str, Any] | Any) -> LngramConfig:
    if isinstance(config, LngramConfig):
        return config
    if isinstance(config, dict):
        return LngramConfig(**config)

    values = {}
    for field_name in LngramConfig.__dataclass_fields__:
        if hasattr(config, field_name):
            values[field_name] = getattr(config, field_name)
    return LngramConfig(**values)


def _resolve_added_module_init_std(config: Any) -> float:
    std = float(getattr(config, "initializer_range", 0.02))
    return std if std > 0.0 else 0.02


@torch.no_grad()
def _init_added_linear_weight_(weight: torch.Tensor, std: float) -> None:
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
def _init_added_embedding_weight_(weight: torch.Tensor, mode: str, std: float) -> None:
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

    raise ValueError(f"Unknown table_init_mode: {mode}")


def _pack_logits_btC_to_btr(
    logits_btC: torch.Tensor,
    bits_per_route: int,
    out_dtype: torch.dtype = torch.int16,
) -> torch.Tensor:
    if logits_btC.dim() != 3:
        raise ValueError(f"Expected [B, T, C], got {tuple(logits_btC.shape)}.")

    batch_size, seq_len, channels = logits_btC.shape
    if channels % bits_per_route != 0:
        raise ValueError(
            f"hidden_size={channels} is not divisible by bits_per_route={bits_per_route}."
        )

    num_routes = channels // bits_per_route
    x = logits_btC.view(batch_size, seq_len, num_routes, bits_per_route)
    bits = (x > 0).to(torch.int32)
    shifts = torch.arange(bits_per_route, device=logits_btC.device, dtype=torch.int32)
    shifts = shifts.view(1, 1, 1, bits_per_route)

    codes = ((bits & 1) << shifts).sum(dim=-1, dtype=torch.int32)
    return codes.to(out_dtype)


def _pack_logits_bstC_to_bstr(
    logits_bstC: torch.Tensor,
    bits_per_route: int,
    out_dtype: torch.dtype = torch.int16,
) -> torch.Tensor:
    if logits_bstC.dim() != 4:
        raise ValueError(f"Expected [B, S, T, C], got {tuple(logits_bstC.shape)}.")

    batch_size, num_subtables, seq_len, channels = logits_bstC.shape
    if channels % bits_per_route != 0:
        raise ValueError(
            f"hidden_size={channels} is not divisible by bits_per_route={bits_per_route}."
        )

    packed = _pack_logits_btC_to_btr(
        logits_bstC.reshape(batch_size * num_subtables, seq_len, channels),
        bits_per_route=bits_per_route,
        out_dtype=out_dtype,
    )
    return packed.view(batch_size, num_subtables, seq_len, -1)


def _project_multi_subtable_q_logits_chunk(
    routed_hidden_states: torch.Tensor,
    latent_q_weight_chunk: torch.Tensor,
) -> torch.Tensor:
    if routed_hidden_states.dim() != 3:
        raise ValueError(f"Expected [B, T, H], got {tuple(routed_hidden_states.shape)}.")
    if latent_q_weight_chunk.dim() != 3:
        raise ValueError(f"Expected [S, H, H], got {tuple(latent_q_weight_chunk.shape)}.")

    batch_size, seq_len, hidden_size = routed_hidden_states.shape
    subtable_count, out_dim, in_dim = latent_q_weight_chunk.shape
    if out_dim != hidden_size or in_dim != hidden_size:
        raise ValueError(
            "latent_q_weight_chunk must have shape [S, hidden_size, hidden_size]."
        )

    fused_weight = latent_q_weight_chunk.reshape(subtable_count * hidden_size, hidden_size)
    q_logits = F.linear(routed_hidden_states, fused_weight, bias=None)
    q_logits = q_logits.view(batch_size, seq_len, subtable_count, hidden_size)
    return q_logits.permute(0, 2, 1, 3).contiguous()


class LatentQSurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat_memory, q_logits, q_codes_btr, memory_module):
        ctx.save_for_backward(q_logits, q_codes_btr.to(torch.int16))
        ctx.memory_module = memory_module
        return flat_memory

    @staticmethod
    def backward(ctx, grad_flat_memory):
        q_logits, q_codes_saved = ctx.saved_tensors
        memory = ctx.memory_module

        if grad_flat_memory is None or not memory.surrogate_enable:
            return grad_flat_memory, None, None, None

        batch_size, seq_len, channels = q_logits.shape
        bits_per_route = int(memory.bits_per_route)
        num_routes = int(memory.num_routes)
        mem_dim = int(memory.mem_dim)
        alphabet_size = int(memory.alphabet_size)
        ngrams = tuple(int(n) for n in memory.ngrams)
        num_ngram_orders = len(ngrams)

        if channels != num_routes * bits_per_route:
            raise ValueError(f"C={channels}, R={num_routes}, bits={bits_per_route} mismatch.")

        device = q_logits.device
        accum_dtype = torch.float32 if memory.surrogate_accum_fp32 else grad_flat_memory.dtype
        temp = float(memory.surrogate_temp)
        grad_scale = float(memory.surrogate_scale)
        route_chunk = max(1, int(memory.surrogate_route_chunk_size))

        grad_flat = grad_flat_memory.reshape(
            batch_size,
            seq_len,
            num_ngram_orders,
            num_routes,
            mem_dim,
        ).to(accum_dtype)

        q_logits_btrm = q_logits.reshape(batch_size, seq_len, num_routes, bits_per_route)
        q_codes_btr = q_codes_saved.to(torch.long)

        grad_q = torch.zeros(
            (batch_size, seq_len, num_routes, bits_per_route),
            device=device,
            dtype=accum_dtype,
        )

        route_ids_full = torch.arange(num_routes, device=device, dtype=torch.long)
        bit_shifts = torch.arange(bits_per_route, device=device, dtype=torch.long)
        bit_values = (1 << bit_shifts).view(1, 1, 1, bits_per_route)

        table_weights = {
            n: memory.tables[f"ngram_{n}"].weight.detach()
            for n in ngrams
        }
        route_vocabs = {n: alphabet_size ** n for n in ngrams}

        for route_start in range(0, num_routes, route_chunk):
            route_end = min(num_routes, route_start + route_chunk)
            chunk_routes = route_end - route_start
            codes_chunk = q_codes_btr[:, :, route_start:route_end]
            score_chunk = torch.zeros(
                (batch_size, seq_len, chunk_routes, bits_per_route),
                device=device,
                dtype=accum_dtype,
            )

            route_offsets = {
                n: route_ids_full[route_start:route_end].view(1, 1, chunk_routes) * route_vocabs[n]
                for n in ngrams
            }

            for ngram_idx, ngram_order in enumerate(ngrams):
                valid_len = seq_len - ngram_order + 1
                if valid_len <= 0:
                    continue

                table_weight = table_weights[ngram_order]
                addr = torch.zeros((batch_size, valid_len, chunk_routes), device=device, dtype=torch.long)
                stride = 1
                for position_idx in range(ngram_order):
                    addr = addr + codes_chunk[:, position_idx : position_idx + valid_len, :] * stride
                    stride *= alphabet_size

                grad_mem_valid = grad_flat[:, ngram_order - 1 :, ngram_idx, route_start:route_end, :]

                for suffix_idx in range(ngram_order):
                    time_low = ngram_order - 1 - suffix_idx
                    time_high = time_low + valid_len
                    curr_codes = codes_chunk[:, time_low:time_high, :]
                    exponent = ngram_order - 1 - suffix_idx

                    deltas = (bit_values * (alphabet_size ** exponent)).view(1, 1, 1, bits_per_route)
                    curr_bits = (
                        (curr_codes.unsqueeze(-1) >> bit_shifts.view(1, 1, 1, bits_per_route)) & 1
                    )

                    addr0 = addr.unsqueeze(-1) - curr_bits * deltas
                    global_idx0 = route_offsets[ngram_order].unsqueeze(-1) + addr0
                    global_idx1 = global_idx0 + deltas

                    pair_idx = torch.stack([global_idx0, global_idx1], dim=-1)
                    pair_vals = F.embedding(pair_idx, table_weight)
                    if pair_vals.dtype != accum_dtype:
                        pair_vals = pair_vals.to(accum_dtype)

                    diff = pair_vals[..., 1, :] - pair_vals[..., 0, :]
                    score_chunk[:, time_low:time_high, :, :] += (
                        grad_mem_valid.unsqueeze(-2) * diff
                    ).sum(dim=-1)

            q_chunk = q_logits_btrm[:, :, route_start:route_end, :].to(accum_dtype)
            probability = torch.sigmoid(temp * q_chunk)
            slope = temp * probability * (1.0 - probability)
            grad_q[:, :, route_start:route_end, :] = grad_scale * slope * score_chunk

        grad_q = grad_q.reshape(batch_size, seq_len, channels).to(q_logits.dtype)
        return grad_flat_memory, grad_q, None, None


class _TensorWeightView:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight


class _PackedMultiSubtableBankSliceView:
    def __init__(self, packed_bank, sub_begin: int, sub_end: int, surrogate_enable: bool):
        self.hidden_size = int(packed_bank.hidden_size)
        self.bits_per_route = int(packed_bank.bits_per_route)
        self.num_routes = int(packed_bank.num_routes)
        self.alphabet_size = int(packed_bank.alphabet_size)
        self.mem_dim = int(packed_bank.mem_dim)
        self.ngrams = tuple(int(n) for n in packed_bank.ngrams)
        self.num_ngram_orders = int(packed_bank.num_ngram_orders)
        self.num_subtables = int(sub_end - sub_begin)

        self.surrogate_enable = bool(surrogate_enable)
        self.surrogate_temp = float(packed_bank.surrogate_temp)
        self.surrogate_scale = float(packed_bank.surrogate_scale)
        self.surrogate_accum_fp32 = bool(packed_bank.surrogate_accum_fp32)
        self.surrogate_route_chunk_size = int(packed_bank.surrogate_route_chunk_size)

        self.surrogate_exact_enable = bool(getattr(packed_bank, "surrogate_exact_enable", False))
        self.surrogate_exact_logit_threshold = float(
            getattr(packed_bank, "surrogate_exact_logit_threshold", 0.0)
        )
        self.surrogate_exact_entropy_threshold = float(
            getattr(packed_bank, "surrogate_exact_entropy_threshold", -1.0)
        )
        self.surrogate_exact_max_positions = int(
            getattr(packed_bank, "surrogate_exact_max_positions", 0)
        )

        self._weights = {
            int(n): packed_bank.tables[f"ngram_{int(n)}"][sub_begin:sub_end]
            for n in self.ngrams
        }
        self._table_vocab_sizes = {
            int(n): self._weights[int(n)].shape[1]
            for n in self.ngrams
        }

    def get_flat_weight(self, ngram_order: int, detach: bool = False) -> torch.Tensor:
        weight = self._weights[int(ngram_order)]
        if detach:
            weight = weight.detach()
        return weight.reshape(self.num_subtables * weight.shape[1], self.mem_dim)

    def subtable_offsets(self, ngram_order: int, device: torch.device) -> torch.Tensor:
        table_vocab_size = self._table_vocab_sizes[int(ngram_order)]
        return (
            torch.arange(self.num_subtables, device=device, dtype=torch.long)
            .view(1, self.num_subtables, 1, 1)
            * table_vocab_size
        )


class PackedMultiSubtableNgramTableBank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int,
        mem_dim: int,
        ngrams: Iterable[int] = (2, 3),
        num_subtables: int = 1,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        table_init_mode: str = "zeros",
        table_init_std_scale: float = 1.0,
        surrogate_enable: bool = True,
        surrogate_temp: float = 1.0,
        surrogate_scale: float = 1.0,
        surrogate_accum_fp32: bool = True,
        surrogate_route_chunk_size: int = 32,
        surrogate_exact_enable: bool = True,
        surrogate_exact_logit_threshold: float = 0.75,
        surrogate_exact_entropy_threshold: float = -1.0,
        surrogate_exact_max_positions: int = 8192,
    ):
        super().__init__()
        if hidden_size % bits_per_route != 0:
            raise ValueError("hidden_size must be divisible by bits_per_route.")
        if num_subtables <= 1:
            raise ValueError("PackedMultiSubtableNgramTableBank requires num_subtables > 1.")

        self.hidden_size = int(hidden_size)
        self.bits_per_route = int(bits_per_route)
        self.num_routes = self.hidden_size // self.bits_per_route
        self.alphabet_size = 1 << self.bits_per_route
        self.mem_dim = int(mem_dim)
        self.ngrams = tuple(sorted(set(int(n) for n in ngrams)))
        self.num_ngram_orders = len(self.ngrams)
        self.num_subtables = int(num_subtables)

        self.initializer_range = float(initializer_range)
        self.table_init_mode = str(table_init_mode)
        self.table_init_std_scale = float(table_init_std_scale)

        self.surrogate_enable = bool(surrogate_enable)
        self.surrogate_temp = float(surrogate_temp)
        self.surrogate_scale = float(surrogate_scale)
        self.surrogate_accum_fp32 = bool(surrogate_accum_fp32)
        self.surrogate_route_chunk_size = int(surrogate_route_chunk_size)

        self.surrogate_exact_enable = bool(surrogate_exact_enable)
        self.surrogate_exact_logit_threshold = float(surrogate_exact_logit_threshold)
        self.surrogate_exact_entropy_threshold = float(surrogate_exact_entropy_threshold)
        self.surrogate_exact_max_positions = int(surrogate_exact_max_positions)

        table_std = self.initializer_range * self.table_init_std_scale

        self.tables = nn.ParameterDict()
        self.route_vocab_by_ngram = {}
        self.table_vocab_by_ngram = {}

        for ngram_order in self.ngrams:
            route_vocab = self.alphabet_size ** int(ngram_order)
            table_vocab = self.num_routes * route_vocab

            weight = nn.Parameter(torch.empty(self.num_subtables, table_vocab, self.mem_dim))
            for sub_idx in range(self.num_subtables):
                _init_added_embedding_weight_(
                    weight[sub_idx],
                    mode=self.table_init_mode,
                    std=table_std,
                )
            self.tables[f"ngram_{int(ngram_order)}"] = weight

            self.route_vocab_by_ngram[int(ngram_order)] = route_vocab
            self.table_vocab_by_ngram[int(ngram_order)] = table_vocab

            self.register_buffer(
                f"_stride_ngram_{int(ngram_order)}",
                (
                    self.alphabet_size
                    ** torch.arange(int(ngram_order), dtype=torch.long)
                ).view(1, 1, 1, int(ngram_order)),
                persistent=False,
            )

        self.register_buffer(
            "_route_ids",
            torch.arange(self.num_routes, dtype=torch.long).view(1, 1, self.num_routes),
            persistent=False,
        )

        self.lookup_dropouts = nn.ModuleList(
            [nn.Dropout(float(dropout)) for _ in range(self.num_subtables)]
        )
        self.lookup_dropout_p = float(dropout)

        self.suffix_flat_dim = self.num_routes * self.mem_dim
        self.flat_dim = self.num_ngram_orders * self.suffix_flat_dim

    def get_stacked_weight(self, ngram_order: int, detach: bool = False) -> torch.Tensor:
        weight = self.tables[f"ngram_{int(ngram_order)}"]
        return weight.detach() if detach else weight

    def get_flat_weight(self, ngram_order: int, detach: bool = False) -> torch.Tensor:
        weight = self.get_stacked_weight(ngram_order, detach=detach)
        return weight.reshape(self.num_subtables * weight.shape[1], self.mem_dim)

    def subtable_offsets(self, ngram_order: int, device: torch.device) -> torch.Tensor:
        table_vocab_size = self.table_vocab_by_ngram[int(ngram_order)]
        return (
            torch.arange(self.num_subtables, device=device, dtype=torch.long)
            .view(1, self.num_subtables, 1, 1)
            * table_vocab_size
        )

    def make_slice_view(
        self,
        sub_begin: int,
        sub_end: int,
        surrogate_enable: bool = True,
    ) -> _PackedMultiSubtableBankSliceView:
        return _PackedMultiSubtableBankSliceView(
            self,
            sub_begin=sub_begin,
            sub_end=sub_end,
            surrogate_enable=surrogate_enable,
        )

    def _resolve_output_dtype(self, output_dtype: Optional[torch.dtype]) -> torch.dtype:
        if output_dtype is not None:
            return output_dtype
        first_weight = self.tables[f"ngram_{self.ngrams[0]}"]
        return first_weight.dtype

    def lookup_slice(
        self,
        route_codes_bstr: torch.Tensor,
        sub_begin: int = 0,
        sub_end: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if route_codes_bstr.dim() != 4:
            raise ValueError(f"Expected [B, S, T, R], got {tuple(route_codes_bstr.shape)}.")
        if route_codes_bstr.dtype != torch.long:
            route_codes_bstr = route_codes_bstr.to(torch.long)

        batch_size, subtable_chunk, seq_len, num_routes = route_codes_bstr.shape
        if num_routes != self.num_routes:
            raise ValueError(f"R={num_routes}, expected={self.num_routes}.")

        if sub_end is None:
            sub_end = sub_begin + subtable_chunk
        if subtable_chunk != (sub_end - sub_begin):
            raise ValueError(
                f"route_codes S={subtable_chunk}, requested slice is [{sub_begin}, {sub_end})."
            )

        target_dtype = self._resolve_output_dtype(output_dtype)

        flat_out = torch.zeros(
            (batch_size, subtable_chunk, seq_len, self.flat_dim),
            device=route_codes_bstr.device,
            dtype=target_dtype,
        )
        if seq_len == 0:
            return flat_out

        route_codes_flat = route_codes_bstr.reshape(batch_size * subtable_chunk, seq_len, num_routes)

        for n_idx, ngram_order in enumerate(self.ngrams):
            if seq_len < ngram_order:
                continue

            stride = getattr(self, f"_stride_ngram_{ngram_order}").to(route_codes_bstr.device)
            windows = route_codes_flat.unfold(dimension=1, size=ngram_order, step=1)
            addr = (windows * stride).sum(dim=-1)
            valid_len = addr.shape[1]

            addr = addr.view(batch_size, subtable_chunk, valid_len, num_routes)

            local_offsets = (
                torch.arange(subtable_chunk, device=route_codes_bstr.device, dtype=torch.long)
                .view(1, subtable_chunk, 1, 1)
                * self.table_vocab_by_ngram[ngram_order]
            )
            route_base = (
                self._route_ids.to(route_codes_bstr.device).view(1, 1, 1, num_routes)
                * self.route_vocab_by_ngram[ngram_order]
            )

            global_idx = addr + local_offsets + route_base

            weight_slice = self.tables[f"ngram_{ngram_order}"][sub_begin:sub_end]
            emb = F.embedding(
                global_idx,
                weight_slice.reshape(subtable_chunk * weight_slice.shape[1], self.mem_dim),
            )

            out_view = flat_out[
                ...,
                n_idx * self.suffix_flat_dim : (n_idx + 1) * self.suffix_flat_dim,
            ].view(batch_size, subtable_chunk, seq_len, num_routes, self.mem_dim)

            out_view[:, :, ngram_order - 1 :, :, :] = emb

        if self.training and self.lookup_dropout_p > 0.0:
            out_5d = flat_out.view(
                batch_size,
                subtable_chunk,
                seq_len,
                self.num_ngram_orders,
                self.num_routes,
                self.mem_dim,
            )
            dropped = []
            for local_idx in range(subtable_chunk):
                dropped.append(
                    self.lookup_dropouts[sub_begin + local_idx](out_5d[:, local_idx])
                )
            flat_out = torch.stack(dropped, dim=1).reshape(
                batch_size,
                subtable_chunk,
                seq_len,
                self.flat_dim,
            )

        return flat_out

    def lookup(
        self,
        route_codes_bstr: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return self.lookup_slice(
            route_codes_bstr,
            sub_begin=0,
            sub_end=self.num_subtables,
            output_dtype=output_dtype,
        )

    def lookup_ngram_valid_slice(
        self,
        route_codes_bstr: torch.Tensor,
        ngram_order: int,
        sub_begin: int = 0,
        sub_end: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        ngram_order = int(ngram_order)

        if route_codes_bstr.dim() != 4:
            raise ValueError(f"Expected [B, S, T, R], got {tuple(route_codes_bstr.shape)}.")
        if route_codes_bstr.dtype != torch.long:
            route_codes_bstr = route_codes_bstr.to(torch.long)

        batch_size, subtable_chunk, seq_len, num_routes = route_codes_bstr.shape
        if num_routes != self.num_routes:
            raise ValueError(f"R={num_routes}, expected={self.num_routes}.")

        if sub_end is None:
            sub_end = sub_begin + subtable_chunk
        if subtable_chunk != (sub_end - sub_begin):
            raise ValueError(
                f"route_codes S={subtable_chunk}, requested slice is [{sub_begin}, {sub_end})."
            )

        target_dtype = self._resolve_output_dtype(output_dtype)
        valid_len = seq_len - ngram_order + 1
        if valid_len <= 0:
            return torch.empty(
                (batch_size, subtable_chunk, 0, self.num_routes, self.mem_dim),
                device=route_codes_bstr.device,
                dtype=target_dtype,
            )

        route_codes_flat = route_codes_bstr.reshape(batch_size * subtable_chunk, seq_len, num_routes)
        stride = getattr(self, f"_stride_ngram_{ngram_order}").to(route_codes_bstr.device)
        windows = route_codes_flat.unfold(dimension=1, size=ngram_order, step=1)
        addr = (windows * stride).sum(dim=-1)
        addr = addr.view(batch_size, subtable_chunk, valid_len, num_routes)

        local_offsets = (
            torch.arange(subtable_chunk, device=route_codes_bstr.device, dtype=torch.long)
            .view(1, subtable_chunk, 1, 1)
            * self.table_vocab_by_ngram[ngram_order]
        )
        route_base = (
            self._route_ids.to(route_codes_bstr.device).view(1, 1, 1, num_routes)
            * self.route_vocab_by_ngram[ngram_order]
        )

        global_idx = addr + local_offsets + route_base
        weight_slice = self.tables[f"ngram_{ngram_order}"][sub_begin:sub_end]
        emb = F.embedding(
            global_idx,
            weight_slice.reshape(subtable_chunk * weight_slice.shape[1], self.mem_dim),
        )

        if emb.dtype != target_dtype:
            emb = emb.to(target_dtype)

        if self.training and self.lookup_dropout_p > 0.0:
            dropped = []
            for local_idx in range(subtable_chunk):
                dropped.append(
                    self.lookup_dropouts[sub_begin + local_idx](emb[:, local_idx])
                )
            emb = torch.stack(dropped, dim=1)

        return emb


class RouteExactNgramTableBank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int,
        mem_dim: int,
        ngrams: Iterable[int] = (2, 3),
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        table_init_mode: str = "zeros",
        table_init_std_scale: float = 1.0,
        surrogate_enable: bool = True,
        surrogate_temp: float = 1.0,
        surrogate_scale: float = 1.0,
        surrogate_accum_fp32: bool = True,
        surrogate_route_chunk_size: int = 32,
    ):
        super().__init__()
        if hidden_size % bits_per_route != 0:
            raise ValueError("hidden_size must be divisible by bits_per_route.")

        self.hidden_size = int(hidden_size)
        self.bits_per_route = int(bits_per_route)
        self.num_routes = self.hidden_size // self.bits_per_route
        self.alphabet_size = 1 << self.bits_per_route
        self.mem_dim = int(mem_dim)
        self.ngrams = tuple(sorted(set(int(n) for n in ngrams)))
        self.num_ngram_orders = len(self.ngrams)
        self.initializer_range = float(initializer_range)
        self.table_init_mode = str(table_init_mode)
        self.table_init_std_scale = float(table_init_std_scale)

        self.surrogate_enable = bool(surrogate_enable)
        self.surrogate_temp = float(surrogate_temp)
        self.surrogate_scale = float(surrogate_scale)
        self.surrogate_accum_fp32 = bool(surrogate_accum_fp32)
        self.surrogate_route_chunk_size = int(surrogate_route_chunk_size)

        table_std = self.initializer_range * self.table_init_std_scale
        self.tables = nn.ModuleDict()
        for ngram_order in self.ngrams:
            vocab_size = self.num_routes * (self.alphabet_size ** ngram_order)
            table = nn.Embedding(vocab_size, self.mem_dim)
            _init_added_embedding_weight_(table.weight, mode=self.table_init_mode, std=table_std)
            self.tables[f"ngram_{ngram_order}"] = table

        self.lookup_dropout = nn.Dropout(float(dropout))
        self.lookup_dropout_p = float(dropout)

        self.suffix_flat_dim = self.num_routes * self.mem_dim
        self.flat_dim = self.num_ngram_orders * self.suffix_flat_dim

    def _resolve_output_dtype(self, output_dtype: Optional[torch.dtype]) -> torch.dtype:
        if output_dtype is not None:
            return output_dtype
        first_table = self.tables[f"ngram_{self.ngrams[0]}"]
        return first_table.weight.dtype

    def _build_global_indices(self, route_codes_btr: torch.Tensor, n: int):
        if route_codes_btr.dtype != torch.long:
            route_codes_btr = route_codes_btr.long()

        batch_size, seq_len, num_routes = route_codes_btr.shape
        if num_routes != self.num_routes:
            raise ValueError(f"R={num_routes}, expected={self.num_routes}.")

        device = route_codes_btr.device
        global_idx = torch.zeros((batch_size, seq_len, num_routes), dtype=torch.long, device=device)
        valid = torch.zeros((batch_size, seq_len, num_routes), dtype=torch.bool, device=device)
        if seq_len < n:
            return global_idx, valid

        windows = route_codes_btr.unfold(dimension=1, size=n, step=1)
        strides = (self.alphabet_size ** torch.arange(n, device=device, dtype=torch.long)).view(1, 1, 1, n)
        addr = (windows * strides).sum(dim=-1)

        route_ids = torch.arange(0, num_routes, dtype=torch.long, device=device).view(1, 1, num_routes)
        route_vocab = self.alphabet_size ** n
        global_idx[:, n - 1 :, :] = route_ids * route_vocab + addr
        valid[:, n - 1 :, :] = True
        return global_idx, valid

    def lookup(self, route_codes_btr: torch.Tensor, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if route_codes_btr.dtype != torch.long:
            route_codes_btr = route_codes_btr.long()

        batch_size, seq_len, num_routes = route_codes_btr.shape
        if num_routes != self.num_routes:
            raise ValueError(f"R={num_routes}, expected={self.num_routes}.")

        target_dtype = self._resolve_output_dtype(output_dtype)

        memory_list = []
        for ngram_order in self.ngrams:
            table = self.tables[f"ngram_{ngram_order}"]
            global_idx, valid = self._build_global_indices(route_codes_btr, n=ngram_order)

            emb = F.embedding(global_idx, table.weight)
            if emb.dtype != target_dtype:
                emb = emb.to(target_dtype)
            emb = emb * valid.unsqueeze(-1).to(dtype=emb.dtype)
            memory_list.append(emb)

        memory_cat = torch.stack(memory_list, dim=2)

        if self.training and self.lookup_dropout_p > 0.0:
            memory_cat = self.lookup_dropout(memory_cat)

        return memory_cat.reshape(batch_size, seq_len, self.flat_dim)


class RMSNorm(nn.Module):
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


if RMSNorm not in ALL_LAYERNORM_LAYERS:
    ALL_LAYERNORM_LAYERS.append(RMSNorm)


class DepthwiseCausalConv(nn.Module):
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
        if x.dim() != 4:
            raise ValueError(f"DepthwiseCausalConv expects [B, T, M, D], got {tuple(x.shape)}.")

        batch_size, seq_len, num_branches, hidden_size = x.shape
        if num_branches != self.num_branches or hidden_size != self.hidden_size:
            raise ValueError(
                "Input shape does not match the configured branch count or hidden size."
            )

        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.norms[0].variance_epsilon)
        weight = torch.stack([norm.weight for norm in self.norms], dim=0).view(1, 1, num_branches, hidden_size)
        x_cat = (weight * x_norm.to(input_dtype)).reshape(batch_size, seq_len, num_branches * hidden_size)

        y = self.conv(x_cat.transpose(1, 2))
        y = y[..., :seq_len]
        y = self.act(y)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, num_branches, hidden_size)
        return y


class RouteExactNgramReadout(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int,
        mem_dim: int,
        ngrams: Iterable[int] = (2, 3),
        initializer_range: float = 0.02,
        out_proj_init_std_scale: float = 1.0,
        rmsnorm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        conv_bias: bool = False,
        conv_zero_init: bool = True,
    ):
        super().__init__()
        if hidden_size % bits_per_route != 0:
            raise ValueError("hidden_size must be divisible by bits_per_route.")

        self.hidden_size = int(hidden_size)
        self.bits_per_route = int(bits_per_route)
        self.num_routes = self.hidden_size // self.bits_per_route
        self.mem_dim = int(mem_dim)
        self.ngrams = tuple(sorted(set(int(n) for n in ngrams)))
        self.num_ngram_orders = len(self.ngrams)
        self.suffix_flat_dim = self.num_routes * self.mem_dim
        self.flat_dim = self.num_ngram_orders * self.suffix_flat_dim

        injector_std = float(initializer_range) * float(out_proj_init_std_scale)

        self.value_proj = nn.ModuleDict()
        self.key_proj = nn.ModuleDict()
        self._ngram_keys = tuple(f"ngram_{int(n)}" for n in self.ngrams)

        for proj_key in self._ngram_keys:
            value_proj = nn.Linear(self.suffix_flat_dim, self.hidden_size, bias=True)
            key_proj = nn.Linear(self.suffix_flat_dim, self.hidden_size, bias=True)

            _init_added_linear_weight_(value_proj.weight, std=injector_std)
            _init_added_linear_weight_(key_proj.weight, std=injector_std)
            nn.init.zeros_(value_proj.bias)
            nn.init.zeros_(key_proj.bias)

            self.value_proj[proj_key] = value_proj
            self.key_proj[proj_key] = key_proj

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

    def proj_dtype(self) -> torch.dtype:
        return self.value_proj[self._ngram_keys[0]].weight.dtype

    def _reshape_flat_to_suffix_memory(self, flat_memory: torch.Tensor) -> torch.Tensor:
        if flat_memory.dim() < 3:
            raise ValueError(f"Expected [..., T, C], got {tuple(flat_memory.shape)}.")
        if flat_memory.shape[-1] != self.flat_dim:
            raise ValueError(f"Expected last dim={self.flat_dim}, got {tuple(flat_memory.shape)}.")

        leading_shape = flat_memory.shape[:-2]
        seq_len = flat_memory.shape[-2]
        return flat_memory.view(
            *leading_shape,
            seq_len,
            self.num_ngram_orders,
            self.num_routes,
            self.mem_dim,
        ).reshape(
            *leading_shape,
            seq_len,
            self.num_ngram_orders,
            self.suffix_flat_dim,
        )

    def _build_valid_ngram_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device, dtype=torch.long)
        min_pos = torch.tensor(
            [int(n) - 1 for n in self.ngrams],
            device=device,
            dtype=torch.long,
        )
        return pos.view(seq_len, 1) >= min_pos.view(1, self.num_ngram_orders)

    def _get_stacked_vk_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        fused_weight = torch.stack(
            [
                torch.cat(
                    [self.value_proj[proj_key].weight, self.key_proj[proj_key].weight],
                    dim=0,
                )
                for proj_key in self._ngram_keys
            ],
            dim=0,
        )
        fused_bias = torch.stack(
            [
                torch.cat(
                    [self.value_proj[proj_key].bias, self.key_proj[proj_key].bias],
                    dim=0,
                )
                for proj_key in self._ngram_keys
            ],
            dim=0,
        )
        return fused_weight, fused_bias

    def prepare_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.query_norm(hidden_states).unsqueeze(2)

    def compute_mixed(
        self,
        flat_memory: torch.Tensor,
        query: torch.Tensor,
        return_confidence: bool = False,
    ):
        suffix_memory = self._reshape_flat_to_suffix_memory(flat_memory).to(dtype=self.proj_dtype())
        fused_weight, fused_bias = self._get_stacked_vk_params()

        vk = torch.matmul(
            suffix_memory.unsqueeze(-2),
            fused_weight.transpose(-1, -2),
        ).squeeze(-2)

        vk = vk + fused_bias.view(
            *([1] * (vk.dim() - 2)),
            self.num_ngram_orders,
            2 * self.hidden_size,
        )

        value, key = vk.split(self.hidden_size, dim=-1)
        key = self.key_norm(key)

        query_fp32 = query if query.dtype == torch.float32 else query.to(torch.float32)
        gate_logits = (query_fp32 * key.to(torch.float32)).sum(dim=-1)
        gate_logits = gate_logits * self.inv_sqrt_hidden

        alpha = torch.sigmoid(gate_logits).to(value.dtype).unsqueeze(-1)
        mixed = (alpha * value).sum(dim=-2)

        if not return_confidence:
            return mixed

        seq_len = flat_memory.shape[-2]
        valid_mask_tn = self._build_valid_ngram_mask(seq_len=seq_len, device=gate_logits.device)

        mask_shape = [1] * (gate_logits.dim() - 2) + [seq_len, self.num_ngram_orders]
        valid_mask = valid_mask_tn.view(*mask_shape)

        neg_large = torch.finfo(gate_logits.dtype).min
        masked_logits = gate_logits.masked_fill(~valid_mask, neg_large)
        confidence = torch.logsumexp(masked_logits, dim=-1)

        has_valid = valid_mask.any(dim=-1)
        confidence = torch.where(has_valid, confidence, torch.zeros_like(confidence))

        return mixed, confidence

    def finalize_output(self, mixed: torch.Tensor) -> torch.Tensor:
        conv_out = self.short_conv(mixed.unsqueeze(2)).squeeze(2)
        return mixed + conv_out

    def inject(
        self,
        hidden_states: torch.Tensor,
        flat_memory: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query is None:
            query = self.prepare_query(hidden_states)
        mixed = self.compute_mixed(flat_memory, query=query)
        return self.finalize_output(mixed)


class RouteExactNgramMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int,
        mem_dim: int,
        ngrams: Iterable[int] = (2, 3),
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        table_init_mode: str = "zeros",
        table_init_std_scale: float = 1.0,
        out_proj_init_std_scale: float = 1.0,
        rmsnorm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        conv_bias: bool = False,
        conv_zero_init: bool = True,
        surrogate_enable: bool = True,
        surrogate_temp: float = 1.0,
        surrogate_scale: float = 1.0,
        surrogate_accum_fp32: bool = True,
        surrogate_route_chunk_size: int = 32,
    ):
        super().__init__()
        shared_ngrams = tuple(int(n) for n in ngrams)
        self.surrogate_enable = bool(surrogate_enable)
        self.table_bank = RouteExactNgramTableBank(
            hidden_size=hidden_size,
            bits_per_route=bits_per_route,
            mem_dim=mem_dim,
            ngrams=shared_ngrams,
            dropout=dropout,
            initializer_range=initializer_range,
            table_init_mode=table_init_mode,
            table_init_std_scale=table_init_std_scale,
            surrogate_enable=surrogate_enable,
            surrogate_temp=surrogate_temp,
            surrogate_scale=surrogate_scale,
            surrogate_accum_fp32=surrogate_accum_fp32,
            surrogate_route_chunk_size=surrogate_route_chunk_size,
        )
        self.readout = RouteExactNgramReadout(
            hidden_size=hidden_size,
            bits_per_route=bits_per_route,
            mem_dim=mem_dim,
            ngrams=shared_ngrams,
            initializer_range=initializer_range,
            out_proj_init_std_scale=out_proj_init_std_scale,
            rmsnorm_eps=rmsnorm_eps,
            conv_kernel_size=conv_kernel_size,
            conv_dilation=conv_dilation,
            conv_bias=conv_bias,
            conv_zero_init=conv_zero_init,
        )

    def lookup(self, route_codes_btr: torch.Tensor) -> torch.Tensor:
        return self.table_bank.lookup(route_codes_btr, output_dtype=self.readout.proj_dtype())

    def inject(
        self,
        hidden_states: torch.Tensor,
        flat_memory: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.readout.inject(hidden_states, flat_memory, query=query)

    def inject_from_route_codes(
        self,
        hidden_states: torch.Tensor,
        route_codes_btr: torch.Tensor,
        q_logits: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_memory = self.lookup(route_codes_btr)
        if q_logits is not None and self.surrogate_enable:
            flat_memory = LatentQSurrogateFunction.apply(
                flat_memory,
                q_logits,
                route_codes_btr,
                self.table_bank,
            )
        return self.inject(hidden_states, flat_memory, query=query)

    def forward(
        self,
        hidden_states: torch.Tensor,
        route_codes_btr: torch.Tensor,
        q_logits: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.inject_from_route_codes(
            hidden_states,
            route_codes_btr,
            q_logits=q_logits,
            query=query,
        )


class LngramInjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int = 4,
        mem_dim: int = 8,
        ngrams: Iterable[int] = (2, 3),
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        table_init_mode: str = "zeros",
        table_init_std_scale: float = 1.0,
        out_proj_init_std_scale: float = 1.0,
        q_proj_init_std_scale: float = 1.0,
        rmsnorm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        surrogate_enable: bool = True,
        surrogate_temp: float = 1.0,
        surrogate_scale: float = 1.0,
        surrogate_accum_fp32: bool = True,
        surrogate_route_chunk_size: int = 32,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bits_per_route = int(bits_per_route)
        self.surrogate_enable = bool(surrogate_enable)

        self.latent_q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        _init_added_linear_weight_(
            self.latent_q_proj.weight,
            std=float(initializer_range) * float(q_proj_init_std_scale),
        )

        self.memory = RouteExactNgramMemory(
            hidden_size=hidden_size,
            bits_per_route=bits_per_route,
            mem_dim=mem_dim,
            ngrams=tuple(int(n) for n in ngrams),
            dropout=dropout,
            initializer_range=initializer_range,
            table_init_mode=table_init_mode,
            table_init_std_scale=table_init_std_scale,
            out_proj_init_std_scale=out_proj_init_std_scale,
            rmsnorm_eps=rmsnorm_eps,
            conv_kernel_size=conv_kernel_size,
            conv_dilation=conv_dilation,
            surrogate_enable=surrogate_enable,
            surrogate_temp=surrogate_temp,
            surrogate_scale=surrogate_scale,
            surrogate_accum_fp32=surrogate_accum_fp32,
            surrogate_route_chunk_size=surrogate_route_chunk_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routed_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if routed_hidden_states is None:
            routed_hidden_states = hidden_states

        q_logits = self.latent_q_proj(routed_hidden_states)
        q_codes = _pack_logits_btC_to_btr(
            q_logits,
            self.bits_per_route,
            out_dtype=torch.int16,
        )

        query = self.memory.readout.prepare_query(hidden_states)
        return self.memory.inject_from_route_codes(
            hidden_states=hidden_states,
            route_codes_btr=q_codes,
            q_logits=q_logits if self.surrogate_enable else None,
            query=query,
        )


class MultiSubtableValidNgramSurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, suffix_memory_valid, q_logits, q_codes_bstr, packed_bank, ngram_order):
        ctx.save_for_backward(q_logits, q_codes_bstr.to(torch.int16))
        ctx.packed_bank = packed_bank
        ctx.ngram_order = int(ngram_order)
        return suffix_memory_valid

    @staticmethod
    def backward(ctx, grad_suffix_memory_valid):
        q_logits, q_codes_saved = ctx.saved_tensors
        memory = ctx.packed_bank
        ngram_order = int(ctx.ngram_order)

        if grad_suffix_memory_valid is None or not memory.surrogate_enable:
            return grad_suffix_memory_valid, None, None, None, None

        batch_size, num_subtables, seq_len, channels = q_logits.shape
        bits_per_route = int(memory.bits_per_route)
        num_routes = int(memory.num_routes)
        mem_dim = int(memory.mem_dim)
        alphabet_size = int(memory.alphabet_size)

        if channels != num_routes * bits_per_route:
            raise ValueError(f"C={channels}, R={num_routes}, bits={bits_per_route} mismatch.")

        valid_len = seq_len - ngram_order + 1
        if valid_len <= 0:
            return grad_suffix_memory_valid, torch.zeros_like(q_logits), None, None, None

        if grad_suffix_memory_valid.shape[:3] != (batch_size, num_subtables, valid_len):
            raise ValueError(
                "grad_suffix_memory_valid has an unexpected shape for the selected ngram order."
            )

        device = q_logits.device
        accum_dtype = (
            torch.float32 if memory.surrogate_accum_fp32 else grad_suffix_memory_valid.dtype
        )
        temp = float(memory.surrogate_temp)
        grad_scale = float(memory.surrogate_scale)
        route_chunk = max(1, int(memory.surrogate_route_chunk_size))

        grad_valid = grad_suffix_memory_valid.reshape(
            batch_size,
            num_subtables,
            valid_len,
            num_routes,
            mem_dim,
        ).to(accum_dtype)

        q_logits_bstrm = q_logits.reshape(
            batch_size,
            num_subtables,
            seq_len,
            num_routes,
            bits_per_route,
        )
        q_codes_bstr = q_codes_saved.to(torch.long)

        grad_q = torch.zeros(
            (batch_size, num_subtables, seq_len, num_routes, bits_per_route),
            device=device,
            dtype=accum_dtype,
        )

        route_ids_full = torch.arange(num_routes, device=device, dtype=torch.long)
        bit_shifts = torch.arange(bits_per_route, device=device, dtype=torch.long)
        bit_values = (1 << bit_shifts).view(1, 1, 1, 1, bits_per_route)

        flat_weight = memory.get_flat_weight(ngram_order, detach=True)
        sub_off = memory.subtable_offsets(ngram_order, device=device)
        route_vocab = alphabet_size ** ngram_order

        for route_start in range(0, num_routes, route_chunk):
            route_end = min(num_routes, route_start + route_chunk)
            chunk_routes = route_end - route_start

            codes_chunk = q_codes_bstr[:, :, :, route_start:route_end]
            score_chunk = torch.zeros(
                (batch_size, num_subtables, seq_len, chunk_routes, bits_per_route),
                device=device,
                dtype=accum_dtype,
            )

            route_offsets = (
                route_ids_full[route_start:route_end].view(1, 1, 1, chunk_routes) * route_vocab
            )

            addr = torch.zeros(
                (batch_size, num_subtables, valid_len, chunk_routes),
                device=device,
                dtype=torch.long,
            )
            stride = 1
            for position_idx in range(ngram_order):
                addr = addr + codes_chunk[:, :, position_idx : position_idx + valid_len, :] * stride
                stride *= alphabet_size

            grad_mem_valid = grad_valid[:, :, :, route_start:route_end, :]

            for suffix_idx in range(ngram_order):
                time_low = ngram_order - 1 - suffix_idx
                time_high = time_low + valid_len
                curr_codes = codes_chunk[:, :, time_low:time_high, :]
                exponent = ngram_order - 1 - suffix_idx

                deltas = (bit_values * (alphabet_size ** exponent)).view(
                    1, 1, 1, 1, bits_per_route
                )
                curr_bits = (
                    (curr_codes.unsqueeze(-1) >> bit_shifts.view(1, 1, 1, 1, bits_per_route)) & 1
                )

                addr0 = addr.unsqueeze(-1) - curr_bits * deltas
                global_idx0 = sub_off.unsqueeze(-1) + route_offsets.unsqueeze(-1) + addr0
                global_idx1 = global_idx0 + deltas

                pair_idx = torch.stack([global_idx0, global_idx1], dim=-1)
                pair_vals = F.embedding(pair_idx, flat_weight)
                if pair_vals.dtype != accum_dtype:
                    pair_vals = pair_vals.to(accum_dtype)

                diff = pair_vals[..., 1, :] - pair_vals[..., 0, :]
                score_chunk[:, :, time_low:time_high, :, :] += (
                    grad_mem_valid.unsqueeze(-2) * diff
                ).sum(dim=-1)

            q_chunk = q_logits_bstrm[:, :, :, route_start:route_end, :].to(accum_dtype)
            probability = torch.sigmoid(temp * q_chunk)
            slope = temp * probability * (1.0 - probability)

            grad_q[:, :, :, route_start:route_end, :] = grad_scale * slope * score_chunk

        grad_q = grad_q.reshape(batch_size, num_subtables, seq_len, channels).to(q_logits.dtype)
        return grad_suffix_memory_valid, grad_q, None, None, None


class MultiSubtableLngramInjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bits_per_route: int = 4,
        mem_dim: int = 8,
        ngrams: Iterable[int] = (2, 3),
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        table_init_mode: str = "zeros",
        table_init_std_scale: float = 1.0,
        out_proj_init_std_scale: float = 1.0,
        q_proj_init_std_scale: float = 1.0,
        rmsnorm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        conv_dilation: Optional[int] = None,
        surrogate_enable: bool = True,
        surrogate_temp: float = 1.0,
        surrogate_scale: float = 1.0,
        surrogate_accum_fp32: bool = True,
        surrogate_route_chunk_size: int = 32,
        num_subtables: int = 1,
        subtable_chunk_size: Optional[int] = None,
        fusion_temperature: float = 1.0,
        null_logit: float = 0.0,
        learnable_null_logit: bool = True,
        null_logit_bound: Optional[float] = 8.0,
        surrogate_exact_enable: bool = True,
        surrogate_exact_logit_threshold: float = 0.75,
        surrogate_exact_entropy_threshold: float = -1.0,
        surrogate_exact_max_positions: int = 8192,
    ):
        super().__init__()
        if num_subtables <= 1:
            raise ValueError("MultiSubtableLngramInjector requires num_subtables > 1.")
        if fusion_temperature <= 0.0:
            raise ValueError(f"fusion_temperature must be > 0, got {fusion_temperature}.")
        if not math.isfinite(float(null_logit)):
            raise ValueError(f"null_logit must be finite, got {null_logit}.")

        self.hidden_size = int(hidden_size)
        self.bits_per_route = int(bits_per_route)
        self.num_subtables = int(num_subtables)
        self.surrogate_enable = bool(surrogate_enable)
        self.shared_ngrams = tuple(int(n) for n in ngrams)
        self.fusion_temperature = float(fusion_temperature)
        self.learnable_null_logit = bool(learnable_null_logit)

        if isinstance(null_logit_bound, str) and null_logit_bound.strip().lower() in ("none", "null"):
            null_logit_bound = None

        if null_logit_bound is None:
            self.null_logit_bound = None
        else:
            self.null_logit_bound = float(null_logit_bound)
            if self.null_logit_bound <= 0.0:
                raise ValueError(
                    f"null_logit_bound must be > 0 or None, got {null_logit_bound}."
                )

        raw_null_logit = float(null_logit)
        if self.null_logit_bound is not None:
            bound = float(self.null_logit_bound)
            ratio = raw_null_logit / bound
            ratio = max(min(ratio, 0.999), -0.999)
            raw_null_logit = bound * math.atanh(ratio)

        null_tensor = torch.tensor(raw_null_logit, dtype=torch.float32)
        if self.learnable_null_logit:
            self.null_logit = nn.Parameter(null_tensor)
        else:
            self.register_buffer("null_logit", null_tensor, persistent=True)

        if subtable_chunk_size is None:
            subtable_chunk_size = min(self.num_subtables, 8)
        self.subtable_chunk_size = int(subtable_chunk_size)
        if self.subtable_chunk_size <= 0:
            raise ValueError("subtable_chunk_size must be > 0.")

        self.latent_q_weight = nn.Parameter(
            torch.empty(self.num_subtables, self.hidden_size, self.hidden_size)
        )
        q_std = float(initializer_range) * float(q_proj_init_std_scale)
        for sub_idx in range(self.num_subtables):
            _init_added_linear_weight_(self.latent_q_weight[sub_idx], std=q_std)

        self.packed_table_bank = PackedMultiSubtableNgramTableBank(
            hidden_size=hidden_size,
            bits_per_route=bits_per_route,
            mem_dim=mem_dim,
            ngrams=self.shared_ngrams,
            num_subtables=num_subtables,
            dropout=dropout,
            initializer_range=initializer_range,
            table_init_mode=table_init_mode,
            table_init_std_scale=table_init_std_scale,
            surrogate_enable=surrogate_enable,
            surrogate_temp=surrogate_temp,
            surrogate_scale=surrogate_scale,
            surrogate_accum_fp32=surrogate_accum_fp32,
            surrogate_route_chunk_size=surrogate_route_chunk_size,
            surrogate_exact_enable=surrogate_exact_enable,
            surrogate_exact_logit_threshold=surrogate_exact_logit_threshold,
            surrogate_exact_entropy_threshold=surrogate_exact_entropy_threshold,
            surrogate_exact_max_positions=surrogate_exact_max_positions,
        )

        self.subtable_surrogate_enable = [bool(surrogate_enable) for _ in range(self.num_subtables)]

        self.readout = RouteExactNgramReadout(
            hidden_size=hidden_size,
            bits_per_route=bits_per_route,
            mem_dim=mem_dim,
            ngrams=self.shared_ngrams,
            initializer_range=initializer_range,
            out_proj_init_std_scale=out_proj_init_std_scale,
            rmsnorm_eps=rmsnorm_eps,
            conv_kernel_size=conv_kernel_size,
            conv_dilation=conv_dilation,
            conv_bias=False,
            conv_zero_init=True,
        )

    def _effective_null_logit(self) -> torch.Tensor:
        x = self.null_logit.to(torch.float32)
        if self.null_logit_bound is not None:
            bound = float(self.null_logit_bound)
            x = bound * torch.tanh(x / bound)
        return x

    def _apply_valid_ngram_surrogate(
        self,
        suffix_memory_valid: torch.Tensor,
        q_logits: torch.Tensor,
        q_codes: torch.Tensor,
        sub_begin: int,
        sub_end: int,
        ngram_order: int,
    ) -> torch.Tensor:
        if not self.surrogate_enable:
            return suffix_memory_valid

        chunk_surrogate_flags = self.subtable_surrogate_enable[sub_begin:sub_end]
        if all(chunk_surrogate_flags) and self.packed_table_bank.surrogate_enable:
            return MultiSubtableValidNgramSurrogateFunction.apply(
                suffix_memory_valid,
                q_logits,
                q_codes,
                self.packed_table_bank.make_slice_view(
                    sub_begin=sub_begin,
                    sub_end=sub_end,
                    surrogate_enable=True,
                ),
                int(ngram_order),
            )

        local_out = []
        for local_idx, sub_idx in enumerate(range(sub_begin, sub_end)):
            branch_valid = suffix_memory_valid[:, local_idx : local_idx + 1]
            if self.subtable_surrogate_enable[sub_idx]:
                branch_valid = MultiSubtableValidNgramSurrogateFunction.apply(
                    branch_valid,
                    q_logits[:, local_idx : local_idx + 1],
                    q_codes[:, local_idx : local_idx + 1],
                    self.packed_table_bank.make_slice_view(
                        sub_begin=sub_idx,
                        sub_end=sub_idx + 1,
                        surrogate_enable=True,
                    ),
                    int(ngram_order),
                )
            local_out.append(branch_valid)
        return torch.cat(local_out, dim=1)

    def _build_branch_proj_cache(self, proj_dtype: torch.dtype):
        cache = {}
        for ngram_order in self.shared_ngrams:
            proj_key = f"ngram_{int(ngram_order)}"
            value_proj = self.readout.value_proj[proj_key]
            key_proj = self.readout.key_proj[proj_key]

            fused_weight = torch.cat([value_proj.weight, key_proj.weight], dim=0)
            fused_bias = torch.cat([value_proj.bias, key_proj.bias], dim=0)

            value_bias = value_proj.bias.view(1, 1, 1, self.hidden_size).to(proj_dtype)
            key_bias = self.readout.key_norm(
                key_proj.bias.view(1, 1, 1, self.hidden_size).to(proj_dtype)
            )

            cache[int(ngram_order)] = {
                "fused_weight": fused_weight,
                "fused_bias": fused_bias,
                "value_bias": value_bias,
                "key_bias_fp32": key_bias.to(torch.float32),
            }
        return cache

    def _streaming_softmax_update_slice(
        self,
        running_max: torch.Tensor,
        running_denom: torch.Tensor,
        running_num: torch.Tensor,
        branch_logits: torch.Tensor,
        branch_values: torch.Tensor,
        time_start: int,
    ) -> None:
        if branch_logits.numel() == 0:
            return

        if branch_logits.dim() != 3:
            raise ValueError(f"branch_logits expected [B, G, L], got {tuple(branch_logits.shape)}.")
        if branch_values.dim() != 4:
            raise ValueError(f"branch_values expected [B, G, L, H], got {tuple(branch_values.shape)}.")
        if branch_values.shape[:3] != branch_logits.shape:
            raise ValueError("branch_values must match branch_logits on the first three dimensions.")

        if branch_logits.dtype != torch.float32:
            branch_logits = branch_logits.to(torch.float32)
        if self.fusion_temperature != 1.0:
            branch_logits = branch_logits / self.fusion_temperature
        if branch_values.dtype != torch.float32:
            branch_values = branch_values.to(torch.float32)

        length = int(branch_logits.shape[2])
        if length <= 0:
            return

        time_slice = slice(time_start, time_start + length)
        prev_max = running_max[:, time_slice].clone()
        prev_denom = running_denom[:, time_slice].clone()
        prev_num = running_num[:, time_slice, :].clone()

        chunk_max = branch_logits.max(dim=1).values
        new_max = torch.maximum(prev_max, chunk_max)

        old_scale = torch.exp(prev_max - new_max)
        chunk_scale = torch.exp(branch_logits - new_max.unsqueeze(1))

        new_denom = prev_denom * old_scale + chunk_scale.sum(dim=1)
        new_num = prev_num * old_scale.unsqueeze(-1) + (
            chunk_scale.unsqueeze(-1) * branch_values
        ).sum(dim=1)

        running_max[:, time_slice] = new_max
        running_denom[:, time_slice] = new_denom
        running_num[:, time_slice, :] = new_num

    def _stream_one_chunk(
        self,
        routed_hidden_states: torch.Tensor,
        shared_query_fp32: torch.Tensor,
        proj_dtype: torch.dtype,
        proj_cache,
        running_max: torch.Tensor,
        running_denom: torch.Tensor,
        running_num: torch.Tensor,
        sub_begin: int,
        sub_end: int,
    ) -> None:
        q_logits = _project_multi_subtable_q_logits_chunk(
            routed_hidden_states,
            self.latent_q_weight[sub_begin:sub_end],
        )
        q_codes = _pack_logits_bstC_to_bstr(
            q_logits,
            self.bits_per_route,
            out_dtype=torch.int16,
        )

        batch_size, subtable_chunk, seq_len, _ = q_logits.shape

        for ngram_order in self.shared_ngrams:
            ngram_order = int(ngram_order)
            prefix_len = min(seq_len, ngram_order - 1)
            cache = proj_cache[ngram_order]

            if prefix_len > 0:
                value_prefix = cache["value_bias"].expand(
                    batch_size,
                    subtable_chunk,
                    prefix_len,
                    self.hidden_size,
                )
                gate_prefix = (
                    shared_query_fp32[:, :, :prefix_len, :] * cache["key_bias_fp32"]
                ).sum(dim=-1) * self.readout.inv_sqrt_hidden
                gate_prefix = gate_prefix.expand(batch_size, subtable_chunk, prefix_len)

                self._streaming_softmax_update_slice(
                    running_max=running_max,
                    running_denom=running_denom,
                    running_num=running_num,
                    branch_logits=gate_prefix,
                    branch_values=value_prefix,
                    time_start=0,
                )

            suffix_valid = self.packed_table_bank.lookup_ngram_valid_slice(
                q_codes,
                ngram_order=ngram_order,
                sub_begin=sub_begin,
                sub_end=sub_end,
                output_dtype=proj_dtype,
            )

            valid_len = int(suffix_valid.shape[2])
            if valid_len <= 0:
                continue

            suffix_valid = suffix_valid.reshape(
                batch_size,
                subtable_chunk,
                valid_len,
                self.readout.suffix_flat_dim,
            )
            suffix_valid = self._apply_valid_ngram_surrogate(
                suffix_memory_valid=suffix_valid,
                q_logits=q_logits,
                q_codes=q_codes,
                sub_begin=sub_begin,
                sub_end=sub_end,
                ngram_order=ngram_order,
            )

            vk_valid = F.linear(
                suffix_valid,
                cache["fused_weight"],
                cache["fused_bias"],
            )
            value_valid, key_valid = vk_valid.split(self.hidden_size, dim=-1)
            key_valid = self.readout.key_norm(key_valid)

            gate_valid = (
                shared_query_fp32[:, :, prefix_len:, :] * key_valid.to(torch.float32)
            ).sum(dim=-1) * self.readout.inv_sqrt_hidden

            self._streaming_softmax_update_slice(
                running_max=running_max,
                running_denom=running_denom,
                running_num=running_num,
                branch_logits=gate_valid,
                branch_values=value_valid,
                time_start=prefix_len,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routed_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if routed_hidden_states is None:
            routed_hidden_states = hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        if seq_len == 0:
            return hidden_states.new_zeros(hidden_states.shape)

        shared_query = self.readout.query_norm(hidden_states).unsqueeze(1)
        shared_query_fp32 = (
            shared_query if shared_query.dtype == torch.float32 else shared_query.to(torch.float32)
        )

        proj_dtype = self.readout.proj_dtype()
        proj_cache = self._build_branch_proj_cache(proj_dtype=proj_dtype)
        chunk = min(self.num_subtables, self.subtable_chunk_size)

        running_max = torch.full(
            (batch_size, seq_len),
            fill_value=-float("inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        running_denom = torch.zeros(
            (batch_size, seq_len),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        running_num = torch.zeros(
            (batch_size, seq_len, self.hidden_size),
            device=hidden_states.device,
            dtype=torch.float32,
        )

        for sub_begin in range(0, self.num_subtables, chunk):
            sub_end = min(self.num_subtables, sub_begin + chunk)
            self._stream_one_chunk(
                routed_hidden_states=routed_hidden_states,
                shared_query_fp32=shared_query_fp32,
                proj_dtype=proj_dtype,
                proj_cache=proj_cache,
                running_max=running_max,
                running_denom=running_denom,
                running_num=running_num,
                sub_begin=sub_begin,
                sub_end=sub_end,
            )

        null_score = (
            self._effective_null_logit().to(device=hidden_states.device, dtype=torch.float32)
            / float(self.fusion_temperature)
        )
        null_score_bt = null_score.reshape(1, 1).expand_as(running_max)

        combined_max = torch.maximum(running_max, null_score_bt)
        real_scale = torch.exp(running_max - combined_max)
        null_scale = torch.exp(null_score_bt - combined_max)

        total_denom = running_denom * real_scale + null_scale
        total_num = running_num * real_scale.unsqueeze(-1)

        mixed = total_num / total_denom.clamp_min(1e-20).unsqueeze(-1)
        mixed = mixed.to(hidden_states.dtype)
        return self.readout.finalize_output(mixed)


def build_lngram_injector(
    hidden_size: int,
    config: LngramConfig | dict[str, Any] | Any,
    initializer_range: float = 0.02,
    rmsnorm_eps: float = 1e-6,
) -> nn.Module:
    cfg = coerce_lngram_config(config)
    ngrams = cfg.normalized_ngrams()
    conv_dilation = cfg.conv_dilation if cfg.conv_dilation is not None else max(int(n) for n in ngrams)

    common_kwargs = dict(
        hidden_size=int(hidden_size),
        bits_per_route=int(cfg.bits_per_route),
        mem_dim=int(cfg.mem_dim),
        ngrams=ngrams,
        dropout=float(cfg.dropout),
        initializer_range=float(initializer_range),
        table_init_mode=str(cfg.table_init_mode),
        table_init_std_scale=float(cfg.table_init_std_scale),
        out_proj_init_std_scale=float(cfg.out_proj_init_std_scale),
        q_proj_init_std_scale=float(cfg.q_proj_init_std_scale),
        rmsnorm_eps=float(rmsnorm_eps),
        conv_kernel_size=int(cfg.conv_kernel_size),
        conv_dilation=conv_dilation,
        surrogate_enable=bool(cfg.surrogate_enable),
        surrogate_temp=float(cfg.surrogate_temp),
        surrogate_scale=float(cfg.surrogate_scale),
        surrogate_accum_fp32=bool(cfg.surrogate_accum_fp32),
        surrogate_route_chunk_size=int(cfg.surrogate_route_chunk_size),
    )

    if int(cfg.num_subtables) > 1:
        num_real_branches = int(cfg.num_subtables) * len(ngrams)
        if cfg.null_logit is None:
            null_logit = float(cfg.fusion_temperature) * math.log(max(1, num_real_branches))
        else:
            null_logit = float(cfg.null_logit)

        return MultiSubtableLngramInjector(
            num_subtables=int(cfg.num_subtables),
            subtable_chunk_size=(
                None if cfg.subtable_chunk_size is None else int(cfg.subtable_chunk_size)
            ),
            fusion_temperature=float(cfg.fusion_temperature),
            null_logit=float(null_logit),
            learnable_null_logit=bool(cfg.learnable_null_logit),
            null_logit_bound=cfg.null_logit_bound,
            surrogate_exact_enable=bool(cfg.surrogate_exact_enable),
            surrogate_exact_logit_threshold=float(cfg.surrogate_exact_logit_threshold),
            surrogate_exact_entropy_threshold=float(cfg.surrogate_exact_entropy_threshold),
            surrogate_exact_max_positions=int(cfg.surrogate_exact_max_positions),
            **common_kwargs,
        )

    return LngramInjector(**common_kwargs)


def estimate_lngram_params_per_layer(
    hidden_size: int,
    bits_per_route: int,
    mem_dim: int,
    ngrams: tuple[int, ...],
    num_subtables: int = 1,
    conv_kernel_size: int = 4,
    conv_bias: bool = False,
    return_breakdown: bool = False,
):
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}.")
    if bits_per_route <= 0:
        raise ValueError(f"bits_per_route must be > 0, got {bits_per_route}.")
    if mem_dim <= 0:
        raise ValueError(f"mem_dim must be > 0, got {mem_dim}.")
    if hidden_size % bits_per_route != 0:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by bits_per_route={bits_per_route}."
        )
    if int(num_subtables) < 1:
        raise ValueError(f"num_subtables must be >= 1, got {num_subtables}.")
    if int(conv_kernel_size) <= 0:
        raise ValueError(f"conv_kernel_size must be > 0, got {conv_kernel_size}.")

    ngrams = tuple(sorted(set(int(n) for n in ngrams)))
    if len(ngrams) == 0:
        raise ValueError("ngrams must not be empty.")
    if any(n <= 0 for n in ngrams):
        raise ValueError(f"ngrams must be positive integers, got {ngrams}.")

    num_subtables = int(num_subtables)
    hidden_size = int(hidden_size)
    mem_dim = int(mem_dim)
    num_routes = hidden_size // int(bits_per_route)
    alphabet_size = 1 << int(bits_per_route)
    num_ngram_orders = len(ngrams)

    q_proj_params = num_subtables * hidden_size * hidden_size

    table_params_by_ngram = {
        f"ngram_{n}": num_subtables * (num_routes * (alphabet_size ** n) * mem_dim)
        for n in ngrams
    }
    tables_total = sum(table_params_by_ngram.values())

    suffix_flat_dim = num_routes * mem_dim
    per_ngram_value_proj = suffix_flat_dim * hidden_size + hidden_size
    per_ngram_key_proj = suffix_flat_dim * hidden_size + hidden_size

    value_proj_total = num_ngram_orders * per_ngram_value_proj
    key_proj_total = num_ngram_orders * per_ngram_key_proj

    query_norm_params = hidden_size
    key_norm_params = hidden_size

    conv_weight_params = hidden_size * int(conv_kernel_size)
    conv_bias_params = hidden_size if bool(conv_bias) else 0
    short_conv_params = conv_weight_params + conv_bias_params

    total = (
        q_proj_params
        + tables_total
        + value_proj_total
        + key_proj_total
        + query_norm_params
        + key_norm_params
        + short_conv_params
    )

    if not return_breakdown:
        return total

    breakdown = {
        "q_proj_total": q_proj_params,
        "tables_total": tables_total,
        "value_proj_total": value_proj_total,
        "key_proj_total": key_proj_total,
        "query_norm_params": query_norm_params,
        "key_norm_params": key_norm_params,
        "short_conv_params": short_conv_params,
        "per_subtable_q": hidden_size * hidden_size,
        "per_subtable_tables": tables_total // num_subtables,
        "suffix_flat_dim": suffix_flat_dim,
        "num_routes": num_routes,
        "alphabet_size": alphabet_size,
        "num_ngrams": num_ngram_orders,
    }
    breakdown.update(table_params_by_ngram)
    return total, breakdown


def summarize_lngram_budget(
    model: nn.Module,
    config: LngramConfig | dict[str, Any] | Any,
) -> dict[str, Any]:
    cfg = coerce_lngram_config(config)
    layers = cfg.normalized_layers()
    per_layer = estimate_lngram_params_per_layer(
        hidden_size=int(getattr(model.config, "hidden_size")),
        bits_per_route=int(cfg.bits_per_route),
        mem_dim=int(cfg.mem_dim),
        ngrams=cfg.normalized_ngrams(),
        num_subtables=int(cfg.num_subtables),
        conv_kernel_size=int(cfg.conv_kernel_size),
        conv_bias=False,
    )
    total_added = per_layer * len(layers)
    total_params = sum(param.numel() for param in model.parameters())

    return {
        "layers": layers,
        "num_subtables": int(cfg.num_subtables),
        "mem_dim": int(cfg.mem_dim),
        "added_params_per_layer": int(per_layer),
        "added_params_total": int(total_added),
        "model_params_total": int(total_params),
    }


def _wrap_decoder_layer_forward(layer: nn.Module) -> None:
    if not hasattr(layer, "_original_forward_without_lngram"):
        layer._original_forward_without_lngram = layer.forward

    def _forward_with_lngram(self, hidden_states: torch.Tensor, *args, **kwargs):
        if getattr(self, "lngram_enabled", False) and getattr(self, "lngram", None) is not None:
            if not hasattr(self, "input_layernorm"):
                raise AttributeError(
                    "Patched layers must expose `input_layernorm` so the L-ngram "
                    "router sees the same normalized activations as the base attention block."
                )

            routed_hidden_states = self.input_layernorm(hidden_states)
            delta = self.lngram(
                hidden_states=hidden_states,
                routed_hidden_states=routed_hidden_states,
            ).to(hidden_states.dtype)
            hidden_states = hidden_states + delta

        return self._original_forward_without_lngram(hidden_states, *args, **kwargs)

    layer.forward = types.MethodType(_forward_with_lngram, layer)


def patch_model_with_lngram(
    model: nn.Module,
    config: LngramConfig | dict[str, Any] | Any,
    layers: Optional[list[nn.Module]] = None,
    verbose: bool = False,
) -> nn.Module:
    cfg = coerce_lngram_config(config)
    if not cfg.enabled:
        return model

    if layers is None:
        try:
            layers = list(model.model.layers)
        except AttributeError as exc:
            raise AttributeError(
                "Could not resolve model.model.layers. Pass `layers=` explicitly for custom architectures."
            ) from exc

    if not layers:
        raise ValueError("No decoder layers were provided for patching.")

    if not hasattr(model, "config") or not hasattr(model.config, "hidden_size"):
        raise AttributeError("The model config must expose `hidden_size`.")

    base_param = next(model.parameters())
    hidden_size = int(model.config.hidden_size)
    init_std = _resolve_added_module_init_std(model.config)
    rmsnorm_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))

    if hidden_size % int(cfg.bits_per_route) != 0:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by bits_per_route={cfg.bits_per_route}."
        )

    active_layers = set(cfg.normalized_layers())

    for layer_index, layer in enumerate(layers):
        if layer_index not in active_layers:
            layer.lngram = None
            layer.lngram_enabled = False
            if hasattr(layer, "_original_forward_without_lngram"):
                layer.forward = layer._original_forward_without_lngram
            continue

        injector = build_lngram_injector(
            hidden_size=hidden_size,
            config=cfg,
            initializer_range=init_std,
            rmsnorm_eps=rmsnorm_eps,
        ).to(dtype=base_param.dtype, device=base_param.device)

        layer.lngram = injector
        layer.lngram_enabled = True
        _wrap_decoder_layer_forward(layer)

    model.lngram_config = cfg

    if verbose:
        summary = summarize_lngram_budget(model, cfg)
        print(f"[Lngram] enabled layers: {summary['layers']}")
        print(
            f"[Lngram] num_subtables={summary['num_subtables']}, "
            f"mem_dim={summary['mem_dim']}, "
            f"added_params_per_layer={summary['added_params_per_layer'] / 1e6:.3f}M"
        )

    return model


if Trainer is not None:

    class LngramTrainer(Trainer):
        def __init__(self, *args, lngram_optimizer_config: Optional[LngramOptimizerConfig] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.lngram_optimizer_config = lngram_optimizer_config or LngramOptimizerConfig()

        def _build_optimizer_grouped_parameters(self):
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = {
                name for name in decay_parameters
                if not name.endswith(".bias")
            }

            named_parameters = [
                (name, param)
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ]

            table_names = {
                name for name, _ in named_parameters
                if (".lngram." in name) and (".tables." in name)
            }
            q_names = {
                name for name, _ in named_parameters
                if (".lngram." in name) and (
                    name.endswith(".latent_q_proj.weight")
                    or name.endswith(".latent_q_weight")
                )
            }
            null_logit_names = {
                name for name, _ in named_parameters
                if (".lngram." in name) and name.endswith(".null_logit")
            }

            special_names = table_names | q_names | null_logit_names

            table_params = [param for name, param in named_parameters if name in table_names]
            q_nodecay_params = [param for name, param in named_parameters if name in q_names]
            null_logit_params = [param for name, param in named_parameters if name in null_logit_names]

            decay_params = [
                param for name, param in named_parameters
                if (name not in special_names) and (name in decay_parameters)
            ]
            nodecay_params = [
                param for name, param in named_parameters
                if (name not in special_names) and (name not in decay_parameters)
            ]

            cfg = self.lngram_optimizer_config
            optimizer_grouped_parameters = []

            if table_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": table_params,
                        "lr": self.args.learning_rate * float(cfg.table_lr_multiplier),
                        "weight_decay": float(cfg.table_weight_decay),
                        "group_name": "lngram_tables",
                    }
                )

            if q_nodecay_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": q_nodecay_params,
                        "lr": self.args.learning_rate,
                        "weight_decay": 0.0,
                        "group_name": "lngram_q",
                    }
                )

            if null_logit_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": null_logit_params,
                        "lr": self.args.learning_rate * float(cfg.null_logit_lr_multiplier),
                        "weight_decay": float(cfg.null_logit_weight_decay),
                        "group_name": "lngram_null_logit",
                    }
                )

            if decay_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": decay_params,
                        "lr": self.args.learning_rate,
                        "weight_decay": self.args.weight_decay,
                        "group_name": "base_decay",
                    }
                )

            if nodecay_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": nodecay_params,
                        "lr": self.args.learning_rate,
                        "weight_decay": 0.0,
                        "group_name": "base_no_decay",
                    }
                )

            seen_param_ids = set()
            for group in optimizer_grouped_parameters:
                for param in group["params"]:
                    param_id = id(param)
                    if param_id in seen_param_ids:
                        raise RuntimeError(
                            f"Parameter appears in multiple optimizer groups: {group.get('group_name')}"
                        )
                    seen_param_ids.add(param_id)

            return optimizer_grouped_parameters

        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer

            optimizer_grouped_parameters = self._build_optimizer_grouped_parameters()

            try:
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                    self.args,
                    self.model,
                )
            except TypeError:
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                    self.args,
                )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters,
                **optimizer_kwargs,
            )

            if getattr(self.args, "process_index", 0) == 0:
                print("[Optimizer] parameter groups:")
                for index, group in enumerate(optimizer_grouped_parameters):
                    num_params = sum(param.numel() for param in group["params"])
                    print(
                        f"  group={index:02d} "
                        f"name={group.get('group_name', 'unnamed')} "
                        f"params={num_params / 1e6:.3f}M "
                        f"lr={group['lr']:.6e} "
                        f"wd={group['weight_decay']}"
                    )

            return self.optimizer

else:  # pragma: no cover

    class LngramTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError("transformers is required to use LngramTrainer.")


__all__ = [
    "LngramConfig",
    "LngramOptimizerConfig",
    "RMSNorm",
    "DepthwiseCausalConv",
    "RouteExactNgramTableBank",
    "RouteExactNgramReadout",
    "RouteExactNgramMemory",
    "LngramInjector",
    "MultiSubtableLngramInjector",
    "build_lngram_injector",
    "estimate_lngram_params_per_layer",
    "summarize_lngram_budget",
    "patch_model_with_lngram",
    "LngramTrainer",
]

