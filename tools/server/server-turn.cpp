#include "server-turn.h"

#include "log.h"

#include <algorithm>
#include <exception>

// the most extra prefixes one request renders, and the most offsets read back from the parameters
static constexpr size_t SERVER_TURN_MAX_EXTRA   = 3;
static constexpr size_t SERVER_TURN_MAX_OFFSETS = 16;

std::vector<server_turn_pick> server_turn_pick_messages(const std::vector<std::string> & roles, const std::vector<int64_t> & est_len, int32_t step) {
    std::vector<server_turn_pick> picks;

    const size_t n = roles.size();
    if (step <= 0 || n < 2 || est_len.size() != n) {
        return picks;
    }

    // the candidates are the messages [0, n - 2]
    const size_t n_cand = n - 1;

    // the end of the leading system messages
    bool   has_sys = false;
    size_t sys_end = 0;
    for (size_t k = 0; k < n_cand && (roles[k] == "system" || roles[k] == "developer"); k++) {
        has_sys = true;
        sys_end = k;
    }

    // the end of the last assistant message
    bool   has_turn = false;
    size_t turn_end = 0;
    for (size_t k = n_cand; k-- > 0;) {
        if (roles[k] == "assistant") {
            has_turn = true;
            turn_end = k;
            break;
        }
    }

    // estimated token position of the end of each candidate
    std::vector<int64_t> end_pos(n_cand);
    int64_t sum = 0;
    for (size_t k = 0; k < n_cand; k++) {
        sum += est_len[k];
        end_pos[k] = sum;
    }

    std::vector<size_t> chosen;
    if (has_sys) {
        picks.push_back({ sys_end, true });
        chosen.push_back(sys_end);
    }
    if (has_turn && !(has_sys && turn_end == sys_end)) {
        picks.push_back({ turn_end, true });
        chosen.push_back(turn_end);
    }

    // More ends, the latest first. The estimate is rough, so they only have to be half a step from the others.
    // The exact positions are checked later, in server_turn_boundaries().
    size_t n_extra = 0;
    for (size_t k = n_cand; k-- > 0 && n_extra < SERVER_TURN_MAX_EXTRA;) {
        if (end_pos[k] < step / 2) {
            break; // the earlier ones are closer to the start
        }

        bool far = true;
        for (const size_t c : chosen) {
            const int64_t d = end_pos[k] > end_pos[c] ? end_pos[k] - end_pos[c] : end_pos[c] - end_pos[k];
            if (c == k || d < step / 2) {
                far = false;
                break;
            }
        }
        if (far) {
            picks.push_back({ k, false });
            chosen.push_back(k);
            n_extra++;
        }
    }

    std::sort(picks.begin(), picks.end(), [](const server_turn_pick & a, const server_turn_pick & b) {
        return a.index < b.index;
    });

    return picks;
}

// the size of a message in tokens, counting 3 bytes per token
static int64_t server_turn_est_len(const common_chat_msg & msg) {
    size_t n = msg.content.size() + msg.reasoning_content.size() + msg.tool_name.size();
    for (const auto & part : msg.content_parts) {
        n += part.text.size();
    }
    for (const auto & call : msg.tool_calls) {
        n += call.name.size() + call.arguments.size();
    }
    return (int64_t) (n / 3 + 8);
}

void server_turn_params(common_json & llama_params, const common_chat_templates * tmpls,
                        const common_chat_templates_inputs & inputs, const std::string & prompt, int32_t step) {
    // a prompt with fewer bytes than the step has fewer tokens than that too
    if (step <= 0 || inputs.messages.size() < 2 || prompt.size() < (size_t) step) {
        return;
    }

    std::vector<std::string> roles;
    std::vector<int64_t>     est_len;
    for (const auto & msg : inputs.messages) {
        if (msg.contains_media()) {
            return; // the media tokens are not in the text
        }
        roles.push_back(msg.role);
        est_len.push_back(server_turn_est_len(msg));
    }

    const auto picks = server_turn_pick_messages(roles, est_len, step);
    if (picks.empty()) {
        return;
    }

    // the same request, cut after a message. The longest prefix goes first, then the list only gets shorter.
    common_chat_templates_inputs sub = inputs;
    sub.add_generation_prompt  = false;
    sub.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    sub.grammar.clear();
    sub.json_schema.clear();
    sub.render_only = true;

    std::vector<size_t> offsets;
    std::vector<int>    required;

    for (auto it = picks.rbegin(); it != picks.rend(); ++it) {
        sub.messages.erase(sub.messages.begin() + (std::ptrdiff_t) (it->index + 1), sub.messages.end());

        std::string text;
        try {
            text = common_chat_templates_apply(tmpls, sub).prompt;
        } catch (const std::exception & e) {
            LOG_DBG("%s: no render of the prefix after message %zu: %s\n", __func__, it->index, e.what());
            continue;
        }

        // the render has to be the start of the prompt text, and shorter than the one before
        if (text.empty() || text.size() >= prompt.size() || prompt.compare(0, text.size(), text) != 0) {
            LOG_DBG("%s: the prefix after message %zu is not the start of the prompt\n", __func__, it->index);
            continue;
        }
        if (!offsets.empty() && text.size() >= offsets.back()) {
            continue;
        }

        offsets.push_back(text.size());
        required.push_back(it->required ? 1 : 0);
    }

    if (offsets.empty()) {
        return;
    }

    std::reverse(offsets.begin(),  offsets.end());
    std::reverse(required.begin(), required.end());

    llama_params["turn_ckpt_offsets"]  = offsets;
    llama_params["turn_ckpt_required"] = required;
}

std::vector<server_turn_boundary> server_turn_boundaries(const common_json & data, const llama_vocab * vocab,
                                                         const std::string & prompt, const llama_tokens & tokens, int32_t step) {
    std::vector<server_turn_boundary> out;

    if (step <= 0 || !data.contains("turn_ckpt_offsets") || !data.contains("turn_ckpt_required")) {
        return out;
    }

    const auto & j_off = data.at("turn_ckpt_offsets");
    const auto & j_req = data.at("turn_ckpt_required");
    if (!j_off.is_array() || !j_req.is_array() || j_off.size() != j_req.size() || j_off.size() > SERVER_TURN_MAX_OFFSETS) {
        return out;
    }

    size_t  last_off = 0;
    int32_t last_pos = 0;

    for (size_t i = 0; i < j_off.size(); i++) {
        if (!j_off.at(i).is_number_integer() || !j_req.at(i).is_number_integer()) {
            return {};
        }

        const size_t off = j_off.at(i).get<size_t>();
        const bool   req = j_req.at(i).get<int>() != 0;

        // the offsets come in order and are inside the text
        if (off <= last_off || off >= prompt.size()) {
            continue;
        }
        last_off = off;

        // the tokens of the prefix have to be the first tokens of the prompt, so the cut is not inside a token
        const llama_tokens head = common_tokenize(vocab, prompt.substr(0, off), true, true);
        if (head.empty() || head.size() >= tokens.size() || !std::equal(head.begin(), head.end(), tokens.begin())) {
            LOG_DBG("%s: the tokens of the prefix at byte %zu are not the start of the prompt\n", __func__, off);
            continue;
        }

        const int32_t pos = (int32_t) head.size();
        if (pos <= last_pos) {
            continue;
        }
        if (req ? (pos < step / 4) : (pos - last_pos < step)) {
            continue;
        }

        out.push_back({ pos, req });
        last_pos = pos;
    }

    return out;
}

const server_turn_boundary * server_turn_find(const std::vector<server_turn_boundary> & turns, int64_t pos) {
    for (const auto & turn : turns) {
        if (turn.pos == pos) {
            return &turn;
        }
    }
    return nullptr;
}
