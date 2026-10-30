#include "../src/llama-expert-device.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>
#include <stdexcept>

int main() {
    constexpr size_t mib = size_t(1) << 20;
    using policy = llama_expert_device_policy;
    using role = llama_expert_device_role;
    const auto legacy = policy::parse(nullptr, nullptr, "6144");
    assert(legacy.margin("ROCm2") == 6144 * mib);
    assert(legacy.role("ROCm3") == role::automatic);
    assert(legacy.stage_allowed("ROCm3"));

    const auto p = policy::parse("ROCm0:rotate, ROCm1:static, ROCm2:off, ROCm3:pinned",
                                "ROCm0, ROCm1", "1536,ROCm2:6144,ROCm3:0");
    p.validate({"ROCm0", "ROCm1", "ROCm2", "ROCm3"});
    assert(p.role("ROCm0") == role::rotating);
    assert(p.role("ROCm1") == role::fixed);
    assert(p.role("ROCm2") == role::off);
    assert(p.role("ROCm3") == role::pinned);
    assert(p.stage_allowed("ROCm0") && p.stage_allowed("ROCm1"));
    assert(!p.stage_allowed("ROCm2") && !p.stage_allowed("ROCm3"));
    assert(p.auto_budget("ROCm0", 16384 * mib) == (16384 - 1536) * mib);
    assert(p.margin("ROCm2") == 6144 * mib);
    assert(p.auto_budget("ROCm2", 16384 * mib) == 0);
    assert(p.auto_budget("ROCm3", 16384 * mib) == 16384 * mib);
    assert(p.auto_budget("ROCm0", 1536 * mib) == 0);
    assert(p.auto_budget("ROCm0", 32 * mib) == 0);
    assert(!policy::parse("ROCm3:pinned", nullptr, nullptr).stage_allowed("ROCm3"));
    assert(!policy::parse(nullptr, "none", nullptr).stage_allowed("ROCm0"));
    assert(policy::parse(nullptr, nullptr, "ROCm2:6144").margin("ROCm1") == 1536 * mib);

    const auto invalid = [](const char * r, const char * s, const char * m) {
        try { policy::parse(r, s, m).validate({"ROCm0", "ROCm1", "ROCm2", "ROCm3"}); }
        catch (const std::runtime_error &) { return; }
        assert(false && "invalid device policy accepted");
    };
    invalid("ROCm3:pinend", nullptr, nullptr);
    invalid("ROCm3:pinned,ROCm3:rotate", nullptr, nullptr);
    invalid("ROCm8:off", nullptr, nullptr);
    invalid("ROCm3:pinned", "ROCm3", nullptr);
    invalid(nullptr, "ROCm0,ROCm0", nullptr);
    invalid(nullptr, "ROCm8", nullptr);
    invalid(nullptr, "", nullptr);
    invalid(nullptr, "ROCm0,", nullptr);
    invalid(nullptr, nullptr, "6144,1536");
    invalid(nullptr, nullptr, "ROCm2:6144,ROCm2:1536");
    invalid(nullptr, nullptr, "ROCm8:6144");
    invalid(nullptr, nullptr, "ROCm2:-1");
    invalid(nullptr, nullptr, "ROCm2:1.5");
    invalid(nullptr, nullptr, "ROCm2:6144garbage");
    invalid(nullptr, nullptr, "18446744073709551616");
    invalid(nullptr, nullptr, "ROCm2:");
    invalid(nullptr, nullptr, ",");
    const size_t largest = std::numeric_limits<size_t>::max() >> 20;
    const auto limit = policy::parse(nullptr, nullptr, std::to_string(largest).c_str());
    assert(limit.margin("ROCm0") == largest * mib);
    invalid(nullptr, nullptr, std::to_string(largest + 1).c_str());
    puts("expert device policy: passed");
}
