#pragma once

#include "common.h"

#include <cstdint>
#include <list>
#include <vector>

// delphi: context checkpoint policy of the server slots (docs/delphi/usage/server-checkpoints.md).
//
// A slot keeps its checkpoints in a list ordered by n_tokens. These helpers hold the decisions that do not need the slot or the
// contexts, so they can be tested alone.

// Pick the checkpoint that a full slot drops to make room for a new one at pos_new.
//
// fifo drops the oldest checkpoint. variance drops the interior checkpoint whose removal leaves the most uniform spacing, that is
// the smallest variance of the gaps between neighbours, counting the new checkpoint as the last point. The oldest checkpoint is
// never an interior one. pos holds the n_tokens of the checkpoints in order and must not be empty. Returns an index into pos.
size_t server_ckpt_victim_index(const std::vector<int64_t> & pos, int64_t pos_new, common_ckpt_evict policy);

std::list<common_prompt_checkpoint>::iterator server_ckpt_pick_victim(
        std::list<common_prompt_checkpoint> & ckpts, int64_t pos_new, common_ckpt_evict policy);
