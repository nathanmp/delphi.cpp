#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

enum class llama_expert_device_role {
    automatic,
    rotating,
    fixed,
    pinned,
    off,
};

// Context-owned settings: draft contexts and replacement schedulers see the same policy without changing device placement.
struct llama_expert_device_policy {
    static llama_expert_device_policy from_env();
    static llama_expert_device_policy parse(const char * roles, const char * staging, const char * margins);

    void validate(const std::vector<std::string> & devices) const;
    llama_expert_device_role role(const std::string & device) const;
    bool stage_allowed(const std::string & device) const;
    size_t margin(const std::string & device) const;
    size_t auto_budget(const std::string & device, size_t free) const;

private:
    std::map<std::string, llama_expert_device_role> roles;
    std::vector<std::string> staging;
    bool staging_set = false;
    size_t default_margin = size_t(1536) << 20;
    std::map<std::string, size_t> margins;
};
