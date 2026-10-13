#pragma once

#include "chat.h"
#include "common.h"
#include "json.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// delphi: context checkpoints at the ends of chat messages (docs/delphi/usage/server-checkpoints.md).
//
// The next request of a conversation has the same prompt up to where its new messages start, so the end of a message is a good
// place for a checkpoint. Finding it takes two steps:
//   1. server_turn_params() renders the chat template for prefixes of the message list and notes, in the request parameters, the
//      byte offset where each prefix ends in the prompt text. It needs the templates and runs in the HTTP thread.
//   2. server_turn_boundaries() tokenizes those prefixes and keeps the positions where the prefix tokens are the first tokens of
//      the prompt. It needs the vocab and the prompt tokens.
// The slot cuts its batches at these positions and takes a checkpoint there, see server-context.cpp.

// a place for a checkpoint in the prompt of a task
struct server_turn_boundary {
    int32_t pos      = 0;     // token position of the first token after the message
    bool    required = false; // the end of the system prompt or of the last turn: kept even when a checkpoint is close
};

// which prefix of the message list to render
struct server_turn_pick {
    size_t index    = 0;     // the prefix is the messages [0, index]
    bool   required = false;
};

// Choose the message ends that are worth a render. roles and est_len (estimated tokens) have one entry per message. The last
// message is never chosen: its end is the end of the prompt, which the ordinary checkpoints near the prompt end cover.
// Required are the end of the leading system messages and the end of the last assistant message. A few more are chosen, about
// step tokens apart, the latest first. Returns the picks in message order, at most 5.
std::vector<server_turn_pick> server_turn_pick_messages(const std::vector<std::string> & roles, const std::vector<int64_t> & est_len, int32_t step);

// Step 1. Adds turn_ckpt_offsets and turn_ckpt_required (parallel arrays) to llama_params, or nothing when no prefix of this
// request is usable. A prefix is usable if its render is the start of the prompt text, which a template that changes the
// earlier messages once a later one is added does not give.
void server_turn_params(common_json & llama_params, const common_chat_templates * tmpls,
                        const common_chat_templates_inputs & inputs, const std::string & prompt, int32_t step);

// Step 2. Reads what step 1 left in data. A prefix gives a boundary if its tokens (tokenized as the prompt is) are the first
// tokens of the prompt. Non-required boundaries are at least step tokens apart and from the start. A required boundary needs
// to be step / 4 tokens from the start, as a checkpoint before that saves little.
std::vector<server_turn_boundary> server_turn_boundaries(const common_json & data, const llama_vocab * vocab,
                                                         const std::string & prompt, const llama_tokens & tokens, int32_t step);

// the boundary at token position pos, or nullptr
const server_turn_boundary * server_turn_find(const std::vector<server_turn_boundary> & turns, int64_t pos);
