import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

from moe_infinity.models.policy_utils import drive_expert_policy

try:
    import nvtx
except ImportError:
    class _NvtxStub:
        @staticmethod
        def annotate(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    nvtx = _NvtxStub()


class _QwenMoEBase(nn.Module):
    def _init_runtime_handles(self) -> None:
        self.lib = None
        self.expert_executor = None
        self.layer_id = None

    def _prepare_expert_route(self, hidden_states):
        router_logits = self.gate(hidden_states)
        router_mask, routing_weights_mask = self.lib.topk_softmax(router_logits)
        return router_logits, router_mask, routing_weights_mask

    def _dispatch_sparse(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, router_mask, routing_weights_mask = (
            self._prepare_expert_route(hidden_states)
        )
        selected_experts = torch.topk(
            routing_weights_mask, self.top_k, dim=-1
        ).indices
        expert_index = selected_experts.reshape(
            batch_size, sequence_length, self.top_k
        )
        drive_expert_policy(self, expert_index)

        self.expert_executor.dispatch_local(
            self.layer_id, hidden_states, router_mask, routing_weights_mask
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()
        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)
        return final_hidden_states, router_logits


class Qwen3MoEBlock(_QwenMoEBase):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )
        self._init_runtime_handles()

    @nvtx.annotate("Qwen3MoEBlock", color="blue")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._dispatch_sparse(hidden_states)


class Qwen2MoEBlock(_QwenMoEBase):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.experts = nn.ModuleList(
            [
                Qwen2MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = Qwen2MoeMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self._init_runtime_handles()

    @nvtx.annotate("Qwen2MoEBlock", color="blue")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        final_hidden_states, _router_logits = self._dispatch_sparse(hidden_states)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        shared_expert_output = torch.sigmoid(
            self.shared_expert_gate(hidden_states_reshaped)
        ) * shared_expert_output
        shared_expert_output = shared_expert_output.view(
            batch_size, sequence_length, hidden_dim
        ).to(final_hidden_states.dtype)
        return final_hidden_states + shared_expert_output
