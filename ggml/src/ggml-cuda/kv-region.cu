#include "common.cuh"
#include "kv-region.cuh"

#include <algorithm>
#include <cstdlib>
#include <vector>

#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP)

struct ggml_cuda_kv_region {
    int         device;
    CUdeviceptr va   = 0;
    size_t      size = 0;

    struct piece {
        size_t                       off;
        size_t                       len;
        CUmemGenericAllocationHandle handle;
    };
    std::vector<piece> pieces; // sorted by off
    size_t gran = 0;           // > 0: every piece is one granule (swappable)
};

// delphi: LLAMA_KV_WINDOW_LEND=1 maps the region one granule per allocation, so that ggml_cuda_kv_region_swap can move
// VRAM between the streams of a KV window
static bool ggml_cuda_kv_region_split() {
    static const bool split = [] {
        const char * env = getenv("LLAMA_KV_WINDOW_LEND");
        return env != nullptr && atoi(env) != 0;
    }();
    return split;
}

static CUmemAllocationProp ggml_cuda_kv_region_prop(int device, bool host) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (host) {
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
        prop.location.id   = 0;
    } else {
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = ggml_cuda_info().devices[device].physical_device;
    }
    return prop;
}

size_t ggml_cuda_kv_region_granularity(int device) {
    if (!ggml_cuda_info().devices[device].vmm) {
        return 0;
    }
    size_t g = 0;
    for (bool host : { false, true }) {
        CUmemAllocationProp prop = ggml_cuda_kv_region_prop(device, host);
        size_t gi = 0;
        if (cuMemGetAllocationGranularity(&gi, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || gi == 0) {
            return 0;
        }
        g = std::max(g, gi);
    }
    return g;
}

void ggml_cuda_kv_region_free(void * region) {
    auto * r = (ggml_cuda_kv_region *) region;
    if (r == nullptr) {
        return;
    }
    for (const auto & p : r->pieces) {
        cuMemUnmap(r->va + p.off, p.len);
        cuMemRelease(p.handle);
    }
    if (r->va) {
        cuMemAddressFree(r->va, r->size);
    }
    delete r;
}

void * ggml_cuda_kv_region_create(int device, size_t size, const size_t * host_ranges, int n, void ** base) {
    const size_t g = ggml_cuda_kv_region_granularity(device);
    if (g == 0 || size == 0 || size % g != 0) {
        return nullptr;
    }

    ggml_cuda_set_device(device);

    auto * r = new ggml_cuda_kv_region;
    r->device = device;
    r->size   = size;

    if (cuMemAddressReserve(&r->va, size, 0, 0, 0) != CUDA_SUCCESS) {
        r->va = 0;
        ggml_cuda_kv_region_free(r);
        return nullptr;
    }

    const bool split = ggml_cuda_kv_region_split();
    r->gran = split ? g : 0;

    auto map = [&](size_t off, size_t len, bool host) {
        CUmemAllocationProp prop = ggml_cuda_kv_region_prop(device, host);
        for (size_t o = off; o < off + len; ) {
            const size_t l = split ? g : len;
            CUmemGenericAllocationHandle h;
            if (cuMemCreate(&h, l, &prop, 0) != CUDA_SUCCESS) {
                return false;
            }
            if (cuMemMap(r->va + o, l, 0, h, 0) != CUDA_SUCCESS) {
                cuMemRelease(h);
                return false;
            }
            r->pieces.push_back({ o, l, h });
            o += l;
        }
        return true;
    };

    bool ok = true;
    size_t cur = 0;
    for (int i = 0; i < n && ok; ++i) {
        const size_t b = host_ranges[2*i + 0];
        const size_t e = host_ranges[2*i + 1];
        if (b < cur || e < b || e > size || b % g != 0 || e % g != 0) {
            ok = false;
            break;
        }
        ok = map(cur, b - cur, false) && map(b, e - b, true);
        cur = e;
    }
    ok = ok && map(cur, size - cur, false);

    if (ok) {
        CUmemAccessDesc acc = {};
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id   = ggml_cuda_info().devices[device].physical_device;
        acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ok = cuMemSetAccess(r->va, size, &acc, 1) == CUDA_SUCCESS;
    }

    if (!ok) {
        ggml_cuda_kv_region_free(r);
        (void) cudaGetLastError();
        return nullptr;
    }

    *base = (void *) r->va;
    return r;
}

size_t ggml_cuda_kv_region_swap_granularity(void * region) {
    auto * r = (ggml_cuda_kv_region *) region;
    return r ? r->gran : 0;
}

bool ggml_cuda_kv_region_swap(void * region, size_t a, size_t b, size_t len, bool keep_a, bool keep_b) {
    auto * r = (ggml_cuda_kv_region *) region;
    if (r == nullptr || r->gran == 0 || len == 0) {
        return false;
    }
    const size_t g = r->gran;
    if (a % g != 0 || b % g != 0 || len % g != 0 || a + len > r->size || b + len > r->size ||
            (a < b + len && b < a + len)) {
        return false;
    }

    auto find = [&](size_t off) -> ggml_cuda_kv_region::piece * {
        auto it = std::lower_bound(r->pieces.begin(), r->pieces.end(), off,
                [](const ggml_cuda_kv_region::piece & p, size_t o) { return p.off < o; });
        return it != r->pieces.end() && it->off == off && it->len == g ? &*it : nullptr;
    };
    for (size_t o = 0; o < len; o += g) {
        if (find(a + o) == nullptr || find(b + o) == nullptr) {
            return false;
        }
    }

    ggml_cuda_set_device(r->device);
    CUDA_CHECK(cudaDeviceSynchronize());

    // a's data goes through pinned host memory; b's data moves into a's allocation through the old mapping, so that
    // after the swap of the allocations each address holds what it held before
    void * tmp = nullptr;
    if (keep_a) {
        CUDA_CHECK(cudaMallocHost(&tmp, len));
        CUDA_CHECK(cudaMemcpy(tmp, (void *) (r->va + a), len, cudaMemcpyDefault));
    }
    if (keep_b) {
        CUDA_CHECK(cudaMemcpy((void *) (r->va + a), (void *) (r->va + b), len, cudaMemcpyDefault));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t o = 0; o < len; o += g) {
        auto * pa = find(a + o);
        auto * pb = find(b + o);
        CU_CHECK(cuMemUnmap(r->va + pa->off, g));
        CU_CHECK(cuMemUnmap(r->va + pb->off, g));
        std::swap(pa->handle, pb->handle);
        CU_CHECK(cuMemMap(r->va + pa->off, g, 0, pa->handle, 0));
        CU_CHECK(cuMemMap(r->va + pb->off, g, 0, pb->handle, 0));
    }
    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id   = ggml_cuda_info().devices[r->device].physical_device;
    acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(r->va + a, len, &acc, 1));
    CU_CHECK(cuMemSetAccess(r->va + b, len, &acc, 1));

    if (keep_a) {
        CUDA_CHECK(cudaMemcpy((void *) (r->va + a), tmp, len, cudaMemcpyDefault));
        CUDA_CHECK(cudaFreeHost(tmp));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

#else

size_t ggml_cuda_kv_region_swap_granularity(void * region) {
    GGML_UNUSED(region);
    return 0;
}

bool ggml_cuda_kv_region_swap(void * region, size_t a, size_t b, size_t len, bool keep_a, bool keep_b) {
    GGML_UNUSED(region); GGML_UNUSED(a); GGML_UNUSED(b); GGML_UNUSED(len); GGML_UNUSED(keep_a); GGML_UNUSED(keep_b);
    return false;
}


size_t ggml_cuda_kv_region_granularity(int device) {
    GGML_UNUSED(device);
    return 0;
}

void * ggml_cuda_kv_region_create(int device, size_t size, const size_t * host_ranges, int n, void ** base) {
    GGML_UNUSED(device); GGML_UNUSED(size); GGML_UNUSED(host_ranges); GGML_UNUSED(n); GGML_UNUSED(base);
    return nullptr;
}

void ggml_cuda_kv_region_free(void * region) {
    GGML_UNUSED(region);
}

#endif
