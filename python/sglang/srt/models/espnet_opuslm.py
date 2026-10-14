# SPDX-License-Identifier: Apache-2.0
"""OpusLM execution bridge; the model implementation belongs to ESPnet."""

from espnet2.speechlm.rl.sglang_model import (
    OpusLMForConditionalGeneration,
    OpusLMDialogueForConditionalGeneration,
)

EntryClass = [OpusLMForConditionalGeneration, OpusLMDialogueForConditionalGeneration]
