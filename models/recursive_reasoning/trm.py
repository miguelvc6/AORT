from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True

    forward_dtype: str = "bfloat16"

    # Legacy TRM switch
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    no_ACT_continue: bool = True  # No continue ACT loss, only use halt sigmoid

    # AORT routing config
    operator_routing: bool = False
    routing_mode: str = "learned_soft"
    schedule_type: str = "mlp_then_attn"
    router_pool: str = "mean"
    router_hidden_size: int = 256
    router_temperature: float = 1.0
    router_entropy_weight: float = 1e-3
    router_balance_weight: float = 0.0


class TinyRecursiveReasoningModel_ACTV1MLPOperator(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, puzzle_emb_len: int) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.mlp_t = SwiGLU(
            hidden_size=config.seq_len + puzzle_emb_len,
            expansion=config.expansion,
        )

    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        del cos_sin
        hidden_states = hidden_states.transpose(1, 2)
        out = self.mlp_t(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states.transpose(1, 2)


class TinyRecursiveReasoningModel_ACTV1AttentionOperator(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )

    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        return rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )


class TinyRecursiveReasoningModel_ACTV1Router(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        if config.router_pool != "mean":
            raise ValueError(f"Unsupported router_pool={config.router_pool!r}. Only 'mean' is implemented.")

        self.step_embedding = nn.Embedding(config.halt_max_steps, config.hidden_size)
        self.input_proj = nn.Linear(3 * config.hidden_size, config.router_hidden_size)
        self.output_proj = nn.Linear(config.router_hidden_size, 2)

        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor, step_indices: torch.Tensor) -> torch.Tensor:
        pooled_H = z_H.mean(dim=1).to(torch.float32)
        pooled_L = z_L.mean(dim=1).to(torch.float32)
        step_indices = torch.clamp(step_indices.to(torch.long), max=self.step_embedding.num_embeddings - 1)
        step_embed = self.step_embedding(step_indices)

        router_input = torch.cat((pooled_H, pooled_L, step_embed), dim=-1)
        hidden = F.gelu(self.input_proj(router_input))
        return self.output_proj(hidden)


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, puzzle_emb_len: int) -> None:
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        if self.config.operator_routing or self.config.mlp_t:
            self.mlp_operator = TinyRecursiveReasoningModel_ACTV1MLPOperator(config, puzzle_emb_len)
        if self.config.operator_routing or not self.config.mlp_t:
            self.attn_operator = TinyRecursiveReasoningModel_ACTV1AttentionOperator(config)

        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def _apply_operator(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[CosSin],
        route_probs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.config.operator_routing:
            if self.config.mlp_t:
                return self.mlp_operator(hidden_states, cos_sin)  # type: ignore[attr-defined]
            return self.attn_operator(hidden_states, cos_sin)  # type: ignore[attr-defined]

        if route_probs is None:
            raise ValueError("route_probs must be provided when operator_routing is enabled.")

        mlp_hidden = self.mlp_operator(hidden_states, cos_sin)  # type: ignore[attr-defined]
        attn_hidden = self.attn_operator(hidden_states, cos_sin)  # type: ignore[attr-defined]
        mlp_weight = route_probs[:, 0].to(hidden_states.dtype).view(-1, 1, 1)
        attn_weight = route_probs[:, 1].to(hidden_states.dtype).view(-1, 1, 1)
        return mlp_weight * mlp_hidden + attn_weight * attn_hidden

    def forward(
        self,
        cos_sin: Optional[CosSin],
        hidden_states: torch.Tensor,
        route_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self._apply_operator(hidden_states, cos_sin=cos_sin, route_probs=route_probs)
        out = self.mlp(hidden_states)
        return rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        route_probs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, route_probs=route_probs, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self._validate_routing_config()

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[
                TinyRecursiveReasoningModel_ACTV1Block(self.config, self.puzzle_emb_len)
                for _i in range(self.config.L_layers)
            ]
        )

        if self.config.operator_routing and self.config.routing_mode == "learned_soft":
            self.router = TinyRecursiveReasoningModel_ACTV1Router(self.config)

        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore[arg-type]

    def _validate_routing_config(self) -> None:
        if not self.config.operator_routing:
            return

        valid_modes = {"fixed_mlp", "fixed_attn", "fixed_schedule", "learned_soft"}
        if self.config.routing_mode not in valid_modes:
            raise ValueError(f"Unsupported routing_mode={self.config.routing_mode!r}.")
        if self.config.schedule_type != "mlp_then_attn":
            raise ValueError(f"Unsupported schedule_type={self.config.schedule_type!r}.")
        if self.config.router_temperature <= 0:
            raise ValueError("router_temperature must be positive.")

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _fixed_route_probs(self, step_indices: torch.Tensor) -> torch.Tensor:
        batch_size = step_indices.shape[0]
        probs = torch.zeros((batch_size, 2), device=step_indices.device, dtype=torch.float32)

        if self.config.routing_mode == "fixed_mlp":
            probs[:, 0] = 1.0
        elif self.config.routing_mode == "fixed_attn":
            probs[:, 1] = 1.0
        elif self.config.routing_mode == "fixed_schedule":
            mlp_steps = (self.config.halt_max_steps + 1) // 2
            use_mlp = step_indices < mlp_steps
            probs[:, 0] = use_mlp.to(torch.float32)
            probs[:, 1] = (~use_mlp).to(torch.float32)
        else:
            raise ValueError(f"Unsupported fixed routing mode {self.config.routing_mode!r}.")

        return probs

    def _route(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        step_indices: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.config.operator_routing:
            return None, {}

        if self.config.routing_mode == "learned_soft":
            router_logits = self.router(z_H, z_L, step_indices)  # type: ignore[attr-defined]
            router_probs = F.softmax(router_logits / self.config.router_temperature, dim=-1)
        else:
            router_probs = self._fixed_route_probs(step_indices)
            router_logits = torch.where(
                router_probs > 0,
                torch.zeros_like(router_probs),
                torch.full_like(router_probs, -1e9),
            )

        router_entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1)
        return router_probs, {
            "router_logits": router_logits,
            "router_probs": router_probs,
            "router_entropy": router_entropy,
            "router_loop_idx": step_indices.to(torch.int32),
        }

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
        step_indices: torch.Tensor,
    ) -> Tuple[
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        seq_info = {
            "cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        }

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        route_probs, routing_outputs = self._route(carry.z_H, carry.z_L, step_indices)

        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, route_probs=route_probs, **seq_info)
                z_H = self.L_level(z_H, z_L, route_probs=route_probs, **seq_info)

        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, route_probs=route_probs, **seq_info)
        z_H = self.L_level(z_H, z_L, route_probs=route_probs, **seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), routing_outputs


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        step_indices = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), routing_outputs = self.inner(
            new_inner_carry,
            new_current_data,
            step_indices=step_indices,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **routing_outputs,
        }

        with torch.no_grad():
            new_steps = step_indices + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            use_adaptive = self.training and self.config.act_enabled and (self.config.halt_max_steps > 1)
            if use_adaptive:
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(
                        new_inner_carry,
                        new_current_data,
                        step_indices=new_steps,
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return TinyRecursiveReasoningModel_ACTV1Carry(
            new_inner_carry,
            new_steps,
            halted,
            new_current_data,
        ), outputs
