#include "common.cuh"
#include "kv-region.cuh"

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
    std::vector<piece> pieces;
};

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

    auto map = [&](size_t off, size_t len, bool host) {
        if (len == 0) {
            return true;
        }
        CUmemAllocationProp prop = ggml_cuda_kv_region_prop(device, host);
        CUmemGenericAllocationHandle h;
        if (cuMemCreate(&h, len, &prop, 0) != CUDA_SUCCESS) {
            return false;
        }
        if (cuMemMap(r->va + off, len, 0, h, 0) != CUDA_SUCCESS) {
            cuMemRelease(h);
            return false;
        }
        r->pieces.push_back({ off, len, h });
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

#else

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
