#include "llama-expert-device.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

std::string trim(const std::string & s) {
    const size_t first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

std::vector<std::string> items(const char * value, const char * name) {
    std::vector<std::string> result;
    if (!value) { return result; }
    const std::string list = value;
    for (size_t pos = 0;;) {
        const size_t end = list.find(',', pos);
        auto item = trim(list.substr(pos, end == std::string::npos ? end : end - pos));
        if (item.empty()) { throw std::runtime_error(std::string(name) + ": empty entry"); }
        result.push_back(std::move(item));
        if (end == std::string::npos) { break; }
        pos = end + 1;
    }
    return result;
}

std::pair<std::string, std::string> pair(const std::string & item, const char * name) {
    const size_t colon = item.find(':');
    if (colon == std::string::npos || item.find(':', colon + 1) != std::string::npos) {
        throw std::runtime_error(std::string(name) + ": expected <device>:<value>");
    }
    auto device = trim(item.substr(0, colon));
    auto value = trim(item.substr(colon + 1));
    if (device.empty() || value.empty()) { throw std::runtime_error(std::string(name) + ": empty device or value"); }
    return {device, value};
}

size_t mib(const std::string & value) {
    size_t n = 0;
    const size_t max = std::numeric_limits<size_t>::max() >> 20;
    for (char c : value) {
        if (c < '0' || c > '9' || n > (max - size_t(c - '0')) / 10) {
            throw std::runtime_error("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB: expected a nonnegative MiB count within size_t");
        }
        n = n * 10 + size_t(c - '0');
    }
    return n << 20;
}

} // namespace

llama_expert_device_policy llama_expert_device_policy::from_env() {
    return parse(getenv("LLAMA_MOE_CACHE_ROLES"), getenv("LLAMA_MOE_STAGE_DEVICES"), getenv("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB"));
}

llama_expert_device_policy llama_expert_device_policy::parse(const char * roles, const char * staging, const char * margins) {
    llama_expert_device_policy result;
    for (const auto & item : items(roles, "LLAMA_MOE_CACHE_ROLES")) {
        const auto kv = pair(item, "LLAMA_MOE_CACHE_ROLES");
        llama_expert_device_role role;
        if      (kv.second == "rotate") { role = llama_expert_device_role::rotating; }
        else if (kv.second == "static") { role = llama_expert_device_role::fixed; }
        else if (kv.second == "pinned") { role = llama_expert_device_role::pinned; }
        else if (kv.second == "off")    { role = llama_expert_device_role::off; }
        else { throw std::runtime_error("LLAMA_MOE_CACHE_ROLES: expected rotate, static, pinned or off for " + kv.first); }
        if (!result.roles.emplace(kv.first, role).second) {
            throw std::runtime_error("LLAMA_MOE_CACHE_ROLES: repeated device " + kv.first);
        }
    }
    result.staging_set = staging != nullptr;
    if (staging && trim(staging) != "none") {
        for (const auto & device : items(staging, "LLAMA_MOE_STAGE_DEVICES")) {
            if (std::find(result.staging.begin(), result.staging.end(), device) != result.staging.end()) {
                throw std::runtime_error("LLAMA_MOE_STAGE_DEVICES: repeated device " + device);
            }
            if (result.role(device) == llama_expert_device_role::pinned) {
                throw std::runtime_error("LLAMA_MOE_STAGE_DEVICES: pinned device cannot stage: " + device);
            }
            result.staging.push_back(device);
        }
    }
    bool have_default = false;
    for (const auto & item : items(margins, "LLAMA_MOE_CACHE_AUTO_MARGIN_MIB")) {
        if (item.find(':') == std::string::npos) {
            if (have_default) { throw std::runtime_error("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB: repeated default"); }
            result.default_margin = mib(item);
            have_default = true;
        } else {
            const auto kv = pair(item, "LLAMA_MOE_CACHE_AUTO_MARGIN_MIB");
            if (!result.margins.emplace(kv.first, mib(kv.second)).second) {
                throw std::runtime_error("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB: repeated device " + kv.first);
            }
        }
    }
    return result;
}

void llama_expert_device_policy::validate(const std::vector<std::string> & devices) const {
    const auto check = [&](const std::string & device, const char * name) {
        if (std::find(devices.begin(), devices.end(), device) == devices.end()) {
            throw std::runtime_error(std::string(name) + ": unknown GPU device " + device);
        }
    };
    for (const auto & kv : roles)   { check(kv.first, "LLAMA_MOE_CACHE_ROLES"); }
    for (const auto & device : staging) { check(device, "LLAMA_MOE_STAGE_DEVICES"); }
    for (const auto & kv : margins) { check(kv.first, "LLAMA_MOE_CACHE_AUTO_MARGIN_MIB"); }
}

llama_expert_device_role llama_expert_device_policy::role(const std::string & device) const {
    const auto it = roles.find(device);
    return it == roles.end() ? llama_expert_device_role::automatic : it->second;
}

bool llama_expert_device_policy::stage_allowed(const std::string & device) const {
    return role(device) != llama_expert_device_role::pinned &&
        (!staging_set || std::find(staging.begin(), staging.end(), device) != staging.end());
}

size_t llama_expert_device_policy::margin(const std::string & device) const {
    const auto it = margins.find(device);
    return it == margins.end() ? default_margin : it->second;
}

size_t llama_expert_device_policy::auto_budget(const std::string & device, size_t free) const {
    return role(device) == llama_expert_device_role::off || free <= margin(device) ? 0 : free - margin(device);
}
