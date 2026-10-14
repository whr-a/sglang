# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Register the ESPnet library's Bagpiper inference architecture."""
from espnet2.speechlm.model.speechlm.rl.bagpiper.sglang_model import ESPnetBagpiperForCausalLM

EntryClass = ESPnetBagpiperForCausalLM
