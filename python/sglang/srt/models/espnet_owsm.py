# SPDX-License-Identifier: Apache-2.0
"""OWSM SGLang execution bridge; model implementation belongs to ESPnet.

The initial correctness implementation recomputes the native decoder prefix.
SGLang retains ownership of request scheduling, sampling and returned logprobs.
"""

from espnet2.s2t.rl.sglang_model import ESPnetOWSMForCausalLM

EntryClass = ESPnetOWSMForCausalLM
