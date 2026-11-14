// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpuset.h"
#include "ggml-exec-trace.h"
#include "ggml-impl.h"
#include "ggml-offload-cost.h"
#include "ggml-stage-copy.h"
#include "ggml-sched-input.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/file.h>
#endif


// persistent offload calibration (see ggml_backend_offload_calibration_load)

struct ggml_backend_offload_calibration_file {
    bool loaded = false;
    std::string path;
    std::unordered_map<std::string, double> values;
};

static std::mutex ggml_backend_offload_calibration_mutex;

// call with ggml_backend_offload_calibration_mutex held
static ggml_backend_offload_calibration_file & ggml_backend_offload_calibration_get_file() {
    static ggml_backend_offload_calibration_file file;
    if (!file.loaded) {
        file.loaded = true;
        const char * path = getenv("GGML_OFFLOAD_CALIBRATION");
        if (path != nullptr && path[0] != '\0') {
            file.path = path;
            if (FILE * f = fopen(path, "r")) {
                char line[512];
                while (fgets(line, sizeof(line), f)) {
                    // "<key> = <value>", the key may contain spaces
                    char * eq = strrchr(line, '=');
                    if (eq == nullptr || eq == line) {
                        continue;
                    }
                    char * end = eq;
                    while (end > line && end[-1] == ' ') {
                        end--;
                    }
                    file.values[std::string(line, end - line)] = strtod(eq + 1, nullptr);
                }
                fclose(f);
            }
            GGML_LOG_INFO("%s: %zu offload calibration values from %s\n", __func__, file.values.size(), path);
        }
    }
    return file;
}

bool ggml_backend_offload_calibration_load(const char * key, double * value) {
    std::lock_guard<std::mutex> lock(ggml_backend_offload_calibration_mutex);
    ggml_backend_offload_calibration_file & file = ggml_backend_offload_calibration_get_file();
    auto it = file.values.find(key);
    if (it == file.values.end()) {
        return false;
    }
    *value = it->second;
    return true;
}

void ggml_backend_offload_calibration_store(const char * key, double value) {
    std::lock_guard<std::mutex> lock(ggml_backend_offload_calibration_mutex);
    ggml_backend_offload_calibration_file & file = ggml_backend_offload_calibration_get_file();
    if (file.path.empty() || file.values.count(key) != 0) {
        return;
    }
    file.values[key] = value;
    // one line per value; several processes may append at once
    if (FILE * f = fopen(file.path.c_str(), "a")) {
#ifdef __linux__
        flock(fileno(f), LOCK_EX);
#endif
        fprintf(f, "%s = %.17g\n", key, value);
        fflush(f);
#ifdef __linux__
        flock(fileno(f), LOCK_UN);
#endif
        fclose(f);
    } else {
        GGML_LOG_WARN("%s: cannot append to %s\n", __func__, file.path.c_str());
    }
}

// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_input_staging {
    struct slot {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_event_t  done   = nullptr;
        size_t offset = 0;
        bool pending = false;
    };
    slot slots[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    int next = 0;

    ~ggml_backend_sched_input_staging() {
        for (auto & backend : slots) {
            for (auto & s : backend) {
                if (s.done) {
                    if (s.pending) {
                        ggml_backend_event_synchronize(s.done);
                    }
                    ggml_backend_event_free(s.done);
                }
                ggml_backend_buffer_free(s.buffer);
            }
        }
    }
};

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched_offload_decision {
    bool present;
    int candidate_backend_id;
    struct ggml_offload_cost_input input;
    struct ggml_offload_cost_result result;
    struct ggml_backend_op_cost cpu;
    struct ggml_backend_op_cost gpu;
};

// delphi: a thread that submits the splits of one GPU backend (ggml_backend_sched_set_worker). For a split it pulls the
// inputs from the other devices on its own stream, launches the split, pushes the outputs that later splits on other GPUs
// read (up to the next split on this backend) into landing buffers on their devices, then records done_event.
// - A pull waits for pull_events[b], recorded on the source backend right after the split that produced the source (or at
//   the post). Splits on b between that one and the worker's split precede it in graph order, so the allocator keeps the
//   source alive through them.
// - The main thread posts the split and moves on. Before it next queues work on a device the job read from or pushed to,
//   that device waits for done_event (later splits may reuse a source's memory; a consumer copies its input from the
//   landing buffer).
struct ggml_backend_sched_worker {
    typedef bool (*copy_on_t)(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst);
    struct copy { struct ggml_tensor * src; struct ggml_tensor * dst; };
    struct forward { int split_id; int input_id; struct ggml_tensor landing; };

    int backend_id = -1;
    int cpu = -1;
    copy_on_t copy_on = nullptr;
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<uint64_t> posted { 0 };
    std::atomic<uint64_t> done { 0 };
    bool quit = false; // under mutex

    // the job, written by the main thread before posted is raised
    struct ggml_backend_sched * sched = nullptr;
    int split_id = -1;
    int cur_copy = 0;
    std::vector<int> pull_wait;        // backends whose pull_events the job waits for before the pulls
    std::vector<int> push_wait;        // backends whose push_events the job waits for before the pushes
    std::vector<copy> pulls;
    std::vector<copy> pushes;          // into the landing tensors of `forwarded`
    std::vector<forward> forwarded;    // the (split, input) pairs that copy their input from a landing buffer
    std::vector<uint32_t> trace_ids;
    struct ggml_exec_trace_thread_context trace_ctx = {};
    enum ggml_status status = GGML_STATUS_SUCCESS;

    // per graph (main thread)
    std::vector<uint8_t> ok;                  // per split: the worker runs it
    std::vector<std::pair<int, int>> records; // (split p, backend b): after queuing split p, record pull_events[b]
    bool recorded[GGML_SCHED_MAX_BACKENDS] = {};

    ggml_backend_event_t pull_events[GGML_SCHED_MAX_BACKENDS] = {};
    ggml_backend_event_t push_events[GGML_SCHED_MAX_BACKENDS] = {};
    ggml_backend_event_t done_event = nullptr;                       // on this backend, at the end of a job
    ggml_backend_event_t pulled_event = nullptr;                     // on this backend, after the pulls of a job
    bool device_wait[GGML_SCHED_MAX_BACKENDS] = {};                  // main thread: backend b must wait for done_event
    ggml_backend_buffer_t landing[GGML_SCHED_MAX_BACKENDS] = {};
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]
    struct ggml_backend_sched_offload_decision * hv_offload_decisions; // [hash_set.size]

    // Tracks which hash slots got copies/decisions written since the last reset, so the
    // reset can clear only those instead of memsetting the full worst-case tables (which
    // are sized for ubatch-sized prefill graphs and dominate MTP decode host time at
    // large ubatches). hv_tensor_backend_ids keeps its full clear: it is small and its
    // readers rely on the -1 default for slots that were inserted but never assigned.
    // The copies/decisions arrays must be calloc'd: the reset never touches unwritten
    // slots, so they have to read as the cleared default from the start.
    ggml_bitset_t * hv_written; // [ggml_bitset_size(hash_set.size)]
    size_t        * hv_touched;  // hash ids with written copies/decisions, each listed once
    size_t          hv_touched_count;
    size_t          hv_touched_capacity;
    bool            scoped_reset; // GGML_SCHED_SCOPED_RESET=0 restores the full-table clear

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    // recorded once per graph, after its last split on the backend: the end of the previous use of copy slot c on b.
    // events[b][c] is re-recorded after every split, so within a graph it marks an earlier split of the same graph,
    // which with pipeline parallelism sits behind the previous ubatch on b.
    ggml_backend_event_t slot_events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    // without pipeline parallelism (one input copy): lets a device-to-device input copy wait on the device for the
    // destination to finish with its previous contents, instead of synchronizing the destination on the host
    ggml_backend_event_t xdev_events[GGML_SCHED_MAX_BACKENDS];
    // without pipeline parallelism: recorded on the split backend before a split's inputs are queued, so that the host
    // waits once per split, and only for the work before them, before overwriting the previous contents of the inputs
    // (GGML_SCHED_REUSE_EVENTS=0: synchronize the split backend before every such input instead)
    ggml_backend_event_t reuse_events[GGML_SCHED_MAX_BACKENDS];
    // delphi: copies between host memory and a GPU go on the GPU stream (ggml_backend_sched_host_async_kind;
    // GGML_SCHED_HOST_ASYNC=0 disables, the events are NULL then). host_read_events[b] is recorded on GPU b after the copies of a
    // split from host memory. The host waits for it before it can write that memory again: before a split on a host backend,
    // and at the end of the graph
    ggml_backend_event_t host_read_events[GGML_SCHED_MAX_BACKENDS];
    bool host_read_pending[GGML_SCHED_MAX_BACKENDS];
    bool input_async; // caller holds host graph inputs until this scheduler completes
    ggml_backend_sched_input_staging * input_staging;
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    // no backend has work in flight: set by ggml_backend_sched_synchronize, cleared when a graph is submitted
    bool backends_idle;

    // delphi: copy the source of a full-size view input instead of the view (GGML_SCHED_VIEW_ALIAS=0 disables)
    bool view_alias;

    // delphi: wait for the previous split before an input-less split only when the two backends can share or read
    // each other's memory (GGML_SCHED_PREV_SPLIT_WAIT=1 always waits)
    bool prev_split_wait_all;

    // delphi: wait on the host once per graph and backend for the previous use of the copy slot, not once per split
    // input (GGML_SCHED_COPY_REUSE_WAIT=1 waits for every input)
    bool copy_reuse_wait_all;
    bool copy_slot_prewait; // GGML_SCHED_COPY_SLOT_PREWAIT=0 disables the copy-slot proof taken before the split loop
    uint64_t n_reserves;

    // delphi: copy the inputs of a split in the graph order of their producers (GGML_SCHED_INPUT_ORDER=0 disables)
    bool input_order;

    // delphi: staging ring for host weights that splits copy whole to a device (on unless GGML_SCHED_STAGE=0), NULL when off
    struct ggml_backend_sched_stage * stage;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;

    uint64_t trace_graph_uid;
    uint32_t trace_graph_id;
    uint32_t trace_topology_graph_id;
    uint32_t * trace_tensor_ids;
    uint32_t * trace_backend_node_ids;
    uint64_t trace_ops_shapes_hash;
    uint64_t trace_backend_hash;
    uint64_t trace_split_hash;
    uint64_t trace_cross_backend_edges;
    bool trace_pipeline_parallel_recorded;

    // delphi: backend workers (ggml_backend_sched_set_worker), NULL where none
    struct ggml_backend_sched_worker * workers[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_sched_worker_ready_t worker_ready[GGML_SCHED_MAX_BACKENDS];
    void * worker_ready_data[GGML_SCHED_MAX_BACKENDS];
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// Mark a hash slot as holding copies/decisions written since the last reset. Each id is
// recorded once per epoch; the reset clears exactly the recorded slots.
static void sched_hv_mark_written(ggml_backend_sched_t sched, size_t id) {
    if (ggml_bitset_get(sched->hv_written, id)) {
        return;
    }
    ggml_bitset_set(sched->hv_written, id);
    if (sched->hv_touched_count >= sched->hv_touched_capacity) {
        const size_t new_capacity = sched->hv_touched_capacity == 0 ? 1024 : 2*sched->hv_touched_capacity;
        size_t * new_touched = (size_t *) realloc(sched->hv_touched, new_capacity*sizeof(size_t));
        if (new_touched == nullptr) {
            GGML_ABORT("%s: failed to grow touched list\n", __func__);
        }
        sched->hv_touched = new_touched;
        sched->hv_touched_capacity = new_capacity;
    }
    sched->hv_touched[sched->hv_touched_count++] = id;
}

// Read helpers that return the cleared default for slots with nothing written since the
// last reset. They use find (no insert) so read-only touches do not pollute the hash set.
static struct ggml_tensor * sched_hv_copy_id(ggml_backend_sched_t sched, size_t id, int backend_id, int copy_id) {
    if (id == GGML_HASHSET_FULL || !ggml_bitset_get(sched->hv_written, id)) {
        return nullptr;
    }
    return sched->hv_tensor_copies[id*sched->n_backends*sched->n_copies + backend_id*sched->n_copies + copy_id];
}

static struct ggml_tensor * sched_hv_copy(ggml_backend_sched_t sched, struct ggml_tensor * tensor, int backend_id, int copy_id) {
    return sched_hv_copy_id(sched, ggml_hash_find(&sched->hash_set, tensor), backend_id, copy_id);
}

static bool ggml_backend_get_op_cost(
        ggml_backend_t backend, const struct ggml_tensor * op, const struct ggml_tensor * weight,
        uint64_t operations, uint64_t active_weight_bytes, struct ggml_backend_op_cost * cost) {
    ggml_backend_get_op_cost_t get_op_cost = (ggml_backend_get_op_cost_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_get_op_cost");
    if (!get_op_cost) {
        return false;
    }
    return get_op_cost(backend, op, weight, operations, active_weight_bytes, cost);
}

static bool ggml_backend_has_op_cost(ggml_backend_t backend) {
    return ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_get_op_cost") != nullptr;
}

static int ggml_backend_sched_backend_from_buffer(
        ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op);

static bool ggml_backend_sched_offload_shape(
        const struct ggml_tensor * op, const struct ggml_tensor * weight,
        uint64_t * operations, uint64_t * active_weight_bytes) {
    if (op->op == GGML_OP_MUL_MAT && op->src[0] == weight && op->src[1]) {
        *operations = ggml_offload_cost_matmul_operations(weight->ne[1], op->src[1]->ne[1], weight->ne[0], 1);
        *active_weight_bytes = ggml_nbytes(weight);
        return *operations > 0 && *active_weight_bytes > 0;
    }
    if (op->op == GGML_OP_MUL_MAT_ID && op->src[0] == weight && op->src[1] && op->src[2]) {
        // src1->ne[2] is the token count. src1->ne[1] is a layout dimension, not the batch size.
        const int64_t n_tokens = op->src[1]->ne[2];
        const int64_t top_k = op->src[2]->ne[0];
        *operations = ggml_offload_cost_matmul_operations(weight->ne[1], n_tokens, weight->ne[0], top_k);
        *active_weight_bytes = ggml_offload_cost_moe_weight_bytes(
                ggml_nbytes(weight), weight->ne[2], n_tokens, top_k);
        return *operations > 0 && *active_weight_bytes > 0;
    }
    return false;
}

static uint64_t ggml_backend_sched_cpu_input_bytes(
        ggml_backend_sched_t sched, const struct ggml_tensor * op, const struct ggml_tensor * weight,
        int * preferred_gpu_backend_id) {
    uint64_t bytes = 0;
    uint64_t preferred_bytes = 0;
    const int cpu_backend_id = sched->n_backends - 1;
    *preferred_gpu_backend_id = -1;
    for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
        const ggml_tensor * src = op->src[src_index];
        if (!src || src == weight) {
            continue;
        }
        const size_t src_id = ggml_hash_find(&sched->hash_set, src);
        int src_backend_id = src_id == GGML_HASHSET_FULL ? -1 : sched->hv_tensor_backend_ids[src_id];
        if (src_backend_id == -1 && src->buffer) {
            src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, op);
        }
        if (src_backend_id >= 0 && src_backend_id != cpu_backend_id) {
            const uint64_t src_bytes = ggml_nbytes(src);
            bytes = ggml_offload_cost_add_sat(bytes, src_bytes);
            if (src_bytes > preferred_bytes) {
                preferred_bytes = src_bytes;
                *preferred_gpu_backend_id = src_backend_id;
            }
        }
    }
    return bytes;
}

static int ggml_backend_sched_cost_offload(
        ggml_backend_sched_t sched, struct ggml_tensor * op, const struct ggml_tensor * weight, int src_backend_id) {
    const char * cost_model = getenv("GGML_OFFLOAD_COST_MODEL");
    if (getenv("GGML_OP_OFFLOAD_MIN_BATCH") || (cost_model && atoi(cost_model) == 0)) {
        return -1;
    }

    uint64_t operations = 0;
    uint64_t active_weight_bytes = 0;
    if (!ggml_backend_sched_offload_shape(op, weight, &operations, &active_weight_bytes)) {
        return -1;
    }

    int preferred_gpu_backend_id = -1;
    const uint64_t cpu_input_bytes = ggml_backend_sched_cpu_input_bytes(
            sched, op, weight, &preferred_gpu_backend_id);

    bool has_candidate = false;
    for (int candidate = 0; candidate < src_backend_id; ++candidate) {
        if (preferred_gpu_backend_id >= 0 && candidate != preferred_gpu_backend_id) {
            continue;
        }
        // a GPU takes only the batches its offload_op accepts (32 tokens or more by default): below that, the selected-expert
        // path waits for the expert ids on every op, which the modeled fixed cost does not include (Qwen 3.8 decode with the
        // experts on the CPU: 13 instead of 26 t/s)
        has_candidate |= ggml_backend_supports_op(sched->backends[candidate], op) &&
                ggml_backend_offload_op(sched->backends[candidate], op) &&
                ggml_backend_has_op_cost(sched->backends[candidate]);
    }
    if (!has_candidate) {
        return preferred_gpu_backend_id >= 0 ? src_backend_id : -1;
    }

    struct ggml_backend_op_cost cpu = {};
    if (!ggml_backend_get_op_cost(sched->backends[src_backend_id], op, weight, operations, active_weight_bytes, &cpu)) {
        return -1;
    }

    const uint64_t cpu_output_bytes = ggml_nbytes(op);
    bool found = false;
    int best_backend_id = -1;
    struct ggml_backend_sched_offload_decision best = {};

    for (int candidate = 0; candidate < src_backend_id; ++candidate) {
        // Keep the operation with its largest resident activation. A different GPU would add an uncalibrated peer transfer.
        if (preferred_gpu_backend_id >= 0 && candidate != preferred_gpu_backend_id) {
            continue;
        }
        if (!ggml_backend_supports_op(sched->backends[candidate], op) || !ggml_backend_offload_op(sched->backends[candidate], op)) {
            continue;
        }
        struct ggml_backend_op_cost gpu = {};
        if (!ggml_backend_get_op_cost(sched->backends[candidate], op, weight, operations, active_weight_bytes, &gpu)) {
            continue;
        }

        const struct ggml_offload_cost_input input = {
            /* .operations                 = */ operations,
            /* .weight_h2d_bytes           = */ active_weight_bytes,
            /* .cpu_input_d2h_bytes        = */ cpu_input_bytes,
            /* .cpu_output_h2d_bytes       = */ cpu_output_bytes,
            /* .cpu_operations_per_second  = */ cpu.operations_per_second,
            /* .gpu_operations_per_second  = */ gpu.operations_per_second,
            /* .h2d_bytes_per_second       = */ gpu.h2d_bytes_per_second,
            /* .d2h_bytes_per_second       = */ gpu.d2h_bytes_per_second,
            /* .cpu_fixed_ns               = */ cpu.fixed_ns,
            /* .gpu_fixed_ns               = */ gpu.fixed_ns,
        };
        const struct ggml_offload_cost_result result = ggml_offload_cost(&input);
        if (!result.valid) {
            continue;
        }
        if (!found || result.gpu_total_ns < best.result.gpu_total_ns) {
            found = true;
            best_backend_id = candidate;
            best.present = true;
            best.candidate_backend_id = candidate;
            best.input = input;
            best.result = result;
            best.cpu = cpu;
            best.gpu = gpu;
        }
    }

    if (!found) {
        return -1;
    }
    {
        const size_t op_id = hash_id(op);
        sched_hv_mark_written(sched, op_id);
        sched->hv_offload_decisions[op_id] = best;
    }
    return best.result.offload ? best_backend_id : src_backend_id;
}

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

static int  ggml_backend_sched_stage_spread(ggml_backend_sched_t sched, const struct ggml_tensor * op,
        const struct ggml_tensor * weight, int backend_id);
static bool ggml_backend_sched_stage_spreading(ggml_backend_sched_t sched);
static int  ggml_backend_sched_stage_route(ggml_backend_sched_t sched, const struct ggml_tensor * op,
        const struct ggml_tensor * weight, int cpu_backend_id);

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    const int stage_backend_id = ggml_backend_sched_stage_route(sched, tensor, src, src_backend_id);
                    if (stage_backend_id != -1) {
                        SET_CAUSE(tensor, stage_backend_id == src_backend_id ? "1.stage.cpu" : "1.stage.gpu");
                        return stage_backend_id == src_backend_id ? stage_backend_id :
                            ggml_backend_sched_stage_spread(sched, tensor, src, stage_backend_id);
                    }
                    const int cost_backend_id = ggml_backend_sched_cost_offload(sched, tensor, src, src_backend_id);
                    if (cost_backend_id != -1) {
                        SET_CAUSE(tensor, cost_backend_id == src_backend_id ? "1.cost.cpu" : "1.cost.gpu");
                        return cost_backend_id == src_backend_id ? cost_backend_id :
                            ggml_backend_sched_stage_spread(sched, tensor, src, cost_backend_id);
                    }
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return ggml_backend_sched_stage_spread(sched, tensor, src, b);
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static uint32_t ggml_backend_sched_trace_device_id(ggml_backend_sched_t sched, int backend_id) {
    if (backend_id < 0 || backend_id >= sched->n_backends) {
        return GGML_EXEC_TRACE_ID_NONE;
    }

    const ggml_backend_dev_t device = ggml_backend_get_device(sched->backends[backend_id]);
    for (int candidate = 0; candidate < backend_id; ++candidate) {
        if (ggml_backend_get_device(sched->backends[candidate]) == device) {
            return candidate + 1;
        }
    }
    return backend_id + 1;
}

static int ggml_backend_sched_trace_tensor_backend_id(ggml_backend_sched_t sched, const ggml_tensor * tensor) {
    const size_t tensor_id = ggml_hash_find(&sched->hash_set, tensor);
    return tensor_id == GGML_HASHSET_FULL ? -1 : sched->hv_tensor_backend_ids[tensor_id];
}

static uint32_t ggml_backend_sched_trace_tensor_id(ggml_backend_sched_t sched, const ggml_tensor * tensor) {
    if (!sched->trace_tensor_ids) {
        return GGML_EXEC_TRACE_ID_NONE;
    }
    const size_t tensor_id = ggml_hash_find(&sched->hash_set, tensor);
    return tensor_id == GGML_HASHSET_FULL ? GGML_EXEC_TRACE_ID_NONE : sched->trace_tensor_ids[tensor_id];
}

static uint64_t ggml_backend_sched_trace_backend_ref(ggml_backend_sched_t sched, int backend_id) {
    if (backend_id < 0 || backend_id >= sched->n_backends) {
        return GGML_EXEC_TRACE_ID_NONE;
    }
    return ((uint64_t) ggml_backend_sched_trace_device_id(sched, backend_id) << 32) | (uint32_t) (backend_id + 1);
}

static uint64_t ggml_backend_sched_trace_span_begin() {
    return ggml_exec_trace_enabled() ? ggml_exec_trace_now_ns() : 0;
}

static uint64_t ggml_backend_sched_trace_detail_span_begin() {
    return ggml_exec_trace_detailed() ? ggml_exec_trace_now_ns() : 0;
}

static void ggml_backend_sched_trace_span_end(
        ggml_backend_sched_t sched,
        uint64_t start_ns,
        enum ggml_exec_trace_event_kind kind,
        uint32_t flags,
        int split_id,
        int backend_id,
        uint32_t node_id,
        uint64_t arg1 = 0,
        uint64_t arg2 = 0,
        uint64_t arg3 = 0) {
    if (start_ns == 0) {
        return;
    }

    ggml_exec_trace_event event = {};
    event.timestamp_ns = start_ns;
    event.args[0] = ggml_exec_trace_now_ns();
    event.args[1] = arg1;
    event.args[2] = arg2;
    event.args[3] = arg3;
    event.kind = kind;
    event.flags = flags;
    event.graph_id = sched->trace_graph_id;
    event.node_id = node_id;
    event.split_id = split_id >= 0 ? split_id + 1 : GGML_EXEC_TRACE_ID_NONE;
    event.backend_id = backend_id >= 0 ? backend_id + 1 : GGML_EXEC_TRACE_ID_NONE;
    event.device_id = ggml_backend_sched_trace_device_id(sched, backend_id);
    ggml_exec_trace_record(event);
}

static void ggml_backend_sched_trace_copy_end(
        ggml_backend_sched_t sched,
        uint64_t start_ns,
        enum ggml_exec_trace_event_kind kind,
        enum ggml_exec_trace_copy_path path,
        int split_id,
        int src_backend_id,
        int dst_backend_id,
        uint32_t src_node_id,
        uint32_t dst_node_id,
        uint64_t bytes) {
    if (kind == GGML_EXEC_TRACE_EVENT_COPY_HANDLING) {
        ggml_exec_trace_integrity_copy(path, split_id + 1, src_backend_id + 1, dst_backend_id + 1, bytes);
    }
    if (start_ns == 0) {
        return;
    }
    ggml_backend_sched_trace_span_end(
            sched, start_ns, kind, path, split_id, dst_backend_id, src_node_id,
            dst_node_id, bytes, ggml_backend_sched_trace_backend_ref(sched, src_backend_id));
}

static void ggml_backend_sched_trace_copy_wait_end(
        ggml_backend_sched_t sched,
        uint64_t start_ns,
        enum ggml_exec_trace_event_kind kind,
        enum ggml_exec_trace_copy_path path,
        int split_id,
        int waited_backend_id,
        int other_backend_id,
        uint32_t node_id,
        enum ggml_exec_trace_wait_api wait_api,
        uint64_t bytes = 0) {
    if (start_ns == 0) {
        return;
    }
    ggml_backend_sched_trace_span_end(
            sched, start_ns, kind, path, split_id, waited_backend_id, node_id,
            wait_api, ggml_backend_sched_trace_backend_ref(sched, other_backend_id), bytes);
}

static void ggml_backend_sched_integrity_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    const uint32_t graph_id = ggml_exec_trace_next_graph_id();
    sched->trace_graph_id = graph_id;

    if (sched->trace_graph_uid == graph->uid && sched->trace_topology_graph_id != 0) {
        ggml_exec_trace_integrity_graph(
                sched->trace_ops_shapes_hash, sched->trace_backend_hash, sched->trace_split_hash,
                graph->n_nodes, sched->n_splits, sched->trace_cross_backend_edges);
        return;
    }

    uint64_t ops_shapes_hash = ggml_exec_trace_hash_init();
    uint64_t backend_hash = ggml_exec_trace_hash_init();
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        const int backend_id = ggml_backend_sched_trace_tensor_backend_id(sched, node);
        ops_shapes_hash = ggml_exec_trace_hash_u64(ops_shapes_hash, (uint64_t) node->op);
        ops_shapes_hash = ggml_exec_trace_hash_u64(ops_shapes_hash, (uint64_t) node->type);
        backend_hash = ggml_exec_trace_hash_u64(backend_hash, (uint64_t) i);
        backend_hash = ggml_exec_trace_hash_u64(backend_hash, (uint64_t) (backend_id + 1));
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            ops_shapes_hash = ggml_exec_trace_hash_u64(ops_shapes_hash, (uint64_t) node->ne[dim]);
        }
    }

    uint64_t split_hash = ggml_exec_trace_hash_init();
    uint64_t cross_backend_edges = 0;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split & split = sched->splits[split_id];
        const ggml_backend_dev_t device = ggml_backend_get_device(sched->backends[split.backend_id]);
        const char * device_name = ggml_backend_dev_name(device);
        split_hash = ggml_exec_trace_hash_bytes(split_hash, device_name, strlen(device_name));
        split_hash = ggml_exec_trace_hash_u64(split_hash, split.backend_id + 1);
        cross_backend_edges += split.n_inputs;
    }

    sched->trace_graph_uid = graph->uid;
    sched->trace_topology_graph_id = graph_id;
    sched->trace_ops_shapes_hash = ops_shapes_hash;
    sched->trace_backend_hash = backend_hash;
    sched->trace_split_hash = split_hash;
    sched->trace_cross_backend_edges = cross_backend_edges;
    ggml_exec_trace_integrity_graph(
            ops_shapes_hash, backend_hash, split_hash, graph->n_nodes, sched->n_splits, cross_backend_edges);
}

static void ggml_backend_sched_trace_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (!sched->trace_tensor_ids) {
        sched->trace_tensor_ids = (uint32_t *) calloc(sched->hash_set.size, sizeof(sched->trace_tensor_ids[0]));
        if (!sched->trace_tensor_ids) {
            ggml_exec_trace_disable("failed to allocate scheduler trace tensor IDs");
            return;
        }
    }

    const uint64_t timestamp = ggml_exec_trace_now_ns();
    const uint32_t graph_id = ggml_exec_trace_next_graph_id();
    const bool topology_reused = sched->trace_graph_uid == graph->uid && sched->trace_topology_graph_id != 0;
    sched->trace_graph_id = graph_id;

    // Pipeline parallelism changes the schedule but is only visible in a host log that most
    // builds do not capture. Record the active scheduler state once so the analysis can report it.
    if (!sched->trace_pipeline_parallel_recorded) {
        sched->trace_pipeline_parallel_recorded = true;
        ggml_exec_trace_metadata("pipeline_parallel", sched->n_copies > 1 ? "1" : "0");
    }

    ggml_exec_trace_event graph_event = {};
    graph_event.timestamp_ns = timestamp;
    graph_event.kind = GGML_EXEC_TRACE_EVENT_GRAPH;
    graph_event.flags = topology_reused ? 1 : 0;
    graph_event.graph_id = graph_id;
    graph_event.args[0] = topology_reused ? sched->trace_topology_graph_id : graph_id;
    graph_event.args[1] = graph->n_nodes;
    graph_event.args[2] = graph->n_leafs;
    graph_event.args[3] = sched->n_splits;
    ggml_exec_trace_record(graph_event);

    if (topology_reused) {
        ggml_exec_trace_event fingerprint = {};
        fingerprint.timestamp_ns = timestamp;
        fingerprint.kind = GGML_EXEC_TRACE_EVENT_FINGERPRINT;
        fingerprint.graph_id = graph_id;
        fingerprint.args[0] = sched->trace_ops_shapes_hash;
        fingerprint.args[1] = sched->trace_backend_hash;
        fingerprint.args[2] = sched->trace_split_hash;
        fingerprint.args[3] = sched->trace_cross_backend_edges;
        ggml_exec_trace_record(fingerprint);
        ggml_exec_trace_integrity_graph(
                sched->trace_ops_shapes_hash, sched->trace_backend_hash, sched->trace_split_hash,
                graph->n_nodes, sched->n_splits, sched->trace_cross_backend_edges);
        return;
    }

    std::vector<uint32_t> node_ids(graph->n_nodes);
    std::unordered_map<const ggml_tensor *, uint32_t> tensor_ids;
    std::unordered_map<const ggml_tensor *, int> tensor_indices;
    tensor_ids.reserve((graph->n_nodes + graph->n_leafs) * 2);
    tensor_indices.reserve(graph->n_nodes * 2);
    memset(sched->trace_tensor_ids, 0, sched->hash_set.size * sizeof(sched->trace_tensor_ids[0]));

    for (int i = 0; i < graph->n_nodes; ++i) {
        node_ids[i] = ggml_exec_trace_next_node_id();
        tensor_ids.emplace(graph->nodes[i], node_ids[i]);
        tensor_indices.emplace(graph->nodes[i], i);
        const size_t tensor_id = ggml_hash_find(&sched->hash_set, graph->nodes[i]);
        if (tensor_id != GGML_HASHSET_FULL) {
            sched->trace_tensor_ids[tensor_id] = node_ids[i];
        }
    }
    for (int i = 0; i < graph->n_leafs; ++i) {
        tensor_ids.emplace(graph->leafs[i], ggml_exec_trace_next_node_id());
        const size_t tensor_id = ggml_hash_find(&sched->hash_set, graph->leafs[i]);
        if (tensor_id != GGML_HASHSET_FULL) {
            sched->trace_tensor_ids[tensor_id] = tensor_ids.at(graph->leafs[i]);
        }
    }

    // Scheduler copy tensors replace cross-backend sources in-place. Relate them to the original graph node when possible.
    auto map_tensor_copies = [&](ggml_tensor * tensor, uint32_t trace_id, int node_index) {
        const size_t tensor_id = ggml_hash_find(&sched->hash_set, tensor);
        if (tensor_id == GGML_HASHSET_FULL) {
            return;
        }
        for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
            for (int copy_id = 0; copy_id < sched->n_copies; ++copy_id) {
                ggml_tensor * copy = sched_hv_copy_id(sched, tensor_id, backend_id, copy_id);
                if (copy) {
                    tensor_ids.emplace(copy, trace_id);
                    const size_t copy_tensor_id = ggml_hash_find(&sched->hash_set, copy);
                    if (copy_tensor_id != GGML_HASHSET_FULL) {
                        sched->trace_tensor_ids[copy_tensor_id] = trace_id;
                    }
                    if (node_index >= 0) {
                        tensor_indices.emplace(copy, node_index);
                    }
                }
            }
        }
    };
    for (int i = 0; i < graph->n_nodes; ++i) {
        map_tensor_copies(graph->nodes[i], node_ids[i], i);
    }
    for (int i = 0; i < graph->n_leafs; ++i) {
        map_tensor_copies(graph->leafs[i], tensor_ids.at(graph->leafs[i]), -1);
    }

    for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
        ggml_backend_t backend = sched->backends[backend_id];
        ggml_backend_dev_t device = ggml_backend_get_device(backend);

        ggml_exec_trace_event event = {};
        event.timestamp_ns = timestamp;
        event.kind = GGML_EXEC_TRACE_EVENT_BACKEND;
        event.graph_id = graph_id;
        event.backend_id = backend_id + 1;
        event.device_id = ggml_backend_sched_trace_device_id(sched, backend_id);
        event.flags = (uint32_t) ggml_backend_dev_type(device);
        event.args[0] = ggml_exec_trace_intern(ggml_backend_name(backend));
        event.args[1] = ggml_exec_trace_intern(ggml_backend_buft_name(sched->bufts[backend_id]));
        event.args[2] = ggml_exec_trace_intern(ggml_backend_dev_name(device));
        event.args[3] = ggml_exec_trace_intern(ggml_backend_dev_description(device));
        ggml_exec_trace_record(event);
    }

    uint64_t ops_shapes_hash = ggml_exec_trace_hash_init();
    uint64_t backend_hash = ggml_exec_trace_hash_init();

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        const int backend_id = ggml_backend_sched_trace_tensor_backend_id(sched, node);

        ops_shapes_hash = ggml_exec_trace_hash_u64(ops_shapes_hash, (uint64_t) node->op);
        ops_shapes_hash = ggml_exec_trace_hash_u64(ops_shapes_hash, (uint64_t) node->type);
        backend_hash = ggml_exec_trace_hash_u64(backend_hash, (uint64_t) i);
        backend_hash = ggml_exec_trace_hash_u64(backend_hash, (uint64_t) (backend_id + 1));
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            ops_shapes_hash = ggml_exec_trace_hash_u64(ops_shapes_hash, (uint64_t) node->ne[dim]);
        }

        ggml_exec_trace_event event = {};
        event.timestamp_ns = timestamp;
        event.kind = GGML_EXEC_TRACE_EVENT_NODE;
        event.flags = (uint32_t) node->op;
        event.graph_id = graph_id;
        event.node_id = node_ids[i];
        event.backend_id = backend_id + 1;
        event.device_id = ggml_backend_sched_trace_device_id(sched, backend_id);
        event.args[0] = i;
        event.args[1] = ggml_exec_trace_intern(node->name);
        event.args[2] = ggml_exec_trace_intern(ggml_op_name(node->op));
        event.args[3] = ggml_exec_trace_intern(ggml_type_name(node->type));
        ggml_exec_trace_record(event);

        event = {};
        event.timestamp_ns = timestamp;
        event.kind = GGML_EXEC_TRACE_EVENT_NODE_DIMS;
        event.graph_id = graph_id;
        event.node_id = node_ids[i];
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            event.args[dim] = node->ne[dim];
        }
        ggml_exec_trace_record(event);

        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            const ggml_tensor * src = node->src[src_index];
            if (!src) {
                continue;
            }

            const auto src_id = tensor_ids.find(src);
            event = {};
            event.timestamp_ns = timestamp;
            event.kind = GGML_EXEC_TRACE_EVENT_NODE_SOURCE;
            event.flags = ggml_exec_trace_intern(ggml_type_name(src->type));
            event.graph_id = graph_id;
            event.node_id = node_ids[i];
            event.args[0] = src_index;
            event.args[1] = src_id == tensor_ids.end() ? 0 : src_id->second;
            event.args[2] = (uint64_t) src->type;
            event.args[3] = ggml_exec_trace_intern(src->name);
            ggml_exec_trace_record(event);

            event = {};
            event.timestamp_ns = timestamp;
            event.kind = GGML_EXEC_TRACE_EVENT_NODE_SOURCE_DIMS;
            event.graph_id = graph_id;
            event.node_id = node_ids[i];
            event.flags = (uint32_t) src_index;
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                event.args[dim] = src->ne[dim];
            }
            ggml_exec_trace_record(event);
        }

        if (node->view_src) {
            const auto view_id = tensor_ids.find(node->view_src);
            event = {};
            event.timestamp_ns = timestamp;
            event.kind = GGML_EXEC_TRACE_EVENT_NODE_VIEW;
            event.graph_id = graph_id;
            event.node_id = node_ids[i];
            event.args[0] = view_id == tensor_ids.end() ? 0 : view_id->second;
            event.args[1] = node->view_offs;
            event.args[2] = ggml_exec_trace_intern(node->view_src->name);
            ggml_exec_trace_record(event);
        }

        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0] && node->src[1]) {
            event = {};
            event.timestamp_ns = timestamp;
            event.kind = GGML_EXEC_TRACE_EVENT_NODE_MATMUL;
            event.graph_id = graph_id;
            event.node_id = node_ids[i];
            event.flags = (uint32_t) node->op;
            event.args[0] = node->src[0]->ne[1];
            event.args[1] = node->op == GGML_OP_MUL_MAT_ID ? node->src[1]->ne[2] : node->src[1]->ne[1];
            event.args[2] = node->src[0]->ne[0];
            event.args[3] = node->op == GGML_OP_MUL_MAT_ID && node->src[2] ? node->src[2]->ne[0] : 0;
            ggml_exec_trace_record(event);
        }

        const size_t decision_id = ggml_hash_find(&sched->hash_set, node);
        const ggml_backend_sched_offload_decision * decision =
            (decision_id == GGML_HASHSET_FULL || !ggml_bitset_get(sched->hv_written, decision_id)) ? nullptr :
            &sched->hv_offload_decisions[decision_id];
        if (decision && decision->present) {
            const int candidate = decision->candidate_backend_id;
            event = {};
            event.timestamp_ns = timestamp;
            event.kind = GGML_EXEC_TRACE_EVENT_OFFLOAD_DECISION;
            event.flags = decision->result.offload ? GGML_EXEC_TRACE_OFFLOAD_CANDIDATE : GGML_EXEC_TRACE_OFFLOAD_CPU;
            event.graph_id = graph_id;
            event.node_id = node_ids[i];
            event.backend_id = candidate + 1;
            event.device_id = ggml_backend_sched_trace_device_id(sched, candidate);
            event.args[0] = decision->result.cpu_compute_ns;
            event.args[1] = decision->result.cpu_transfer_ns;
            event.args[2] = decision->result.gpu_compute_ns;
            event.args[3] = decision->result.gpu_transfer_ns;
            ggml_exec_trace_record(event);

            event.kind = GGML_EXEC_TRACE_EVENT_OFFLOAD_BYTES;
            event.flags = 0;
            event.args[0] = decision->input.weight_h2d_bytes;
            event.args[1] = decision->input.cpu_input_d2h_bytes;
            event.args[2] = decision->input.cpu_output_h2d_bytes;
            event.args[3] = decision->input.operations;
            ggml_exec_trace_record(event);

            event.kind = GGML_EXEC_TRACE_EVENT_OFFLOAD_CALIBRATION;
            event.flags = decision->gpu.numa_node < 0 ? 0 : (uint32_t) decision->gpu.numa_node + 1;
            event.args[0] = (uint64_t) decision->cpu.operations_per_second;
            event.args[1] = (uint64_t) decision->gpu.operations_per_second;
            event.args[2] = (uint64_t) decision->gpu.h2d_bytes_per_second;
            event.args[3] = (uint64_t) decision->gpu.d2h_bytes_per_second;
            ggml_exec_trace_record(event);
        }
    }

    uint64_t split_hash = ggml_exec_trace_hash_init();
    uint64_t cross_backend_edges = 0;
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split & split = sched->splits[split_id];
        const ggml_backend_t backend = sched->backends[split.backend_id];
        const ggml_backend_dev_t device = ggml_backend_get_device(backend);
        const char * device_name = ggml_backend_dev_name(device);
        split_hash = ggml_exec_trace_hash_bytes(split_hash, device_name, strlen(device_name));
        split_hash = ggml_exec_trace_hash_u64(split_hash, split.backend_id + 1);
        cross_backend_edges += split.n_inputs;

        ggml_exec_trace_event event = {};
        event.timestamp_ns = timestamp;
        event.kind = GGML_EXEC_TRACE_EVENT_SCHED_SPLIT;
        event.graph_id = graph_id;
        event.split_id = split_id + 1;
        event.backend_id = split.backend_id + 1;
        event.device_id = ggml_backend_sched_trace_device_id(sched, split.backend_id);
        event.args[0] = split.i_start;
        event.args[1] = split.i_end;
        event.args[2] = split.i_end - split.i_start;
        event.args[3] = split.n_inputs;
        ggml_exec_trace_record(event);

        for (int input_id = 0; input_id < split.n_inputs; ++input_id) {
            ggml_tensor * input = split.inputs[input_id];
            const auto node_id = tensor_ids.find(input);
            const auto node_index = tensor_indices.find(input);
            uint32_t source_split_id = 0;
            if (node_index != tensor_indices.end()) {
                for (int candidate = 0; candidate < sched->n_splits; ++candidate) {
                    if (node_index->second >= sched->splits[candidate].i_start && node_index->second < sched->splits[candidate].i_end) {
                        source_split_id = candidate + 1;
                        break;
                    }
                }
            }

            event = {};
            event.timestamp_ns = timestamp;
            event.kind = GGML_EXEC_TRACE_EVENT_SCHED_SPLIT_INPUT;
            event.graph_id = graph_id;
            event.split_id = split_id + 1;
            event.backend_id = split.backend_id + 1;
            event.device_id = ggml_backend_sched_trace_device_id(sched, split.backend_id);
            event.flags = ggml_backend_sched_trace_tensor_backend_id(sched, input) + 1;
            event.args[0] = input_id;
            event.args[1] = node_id == tensor_ids.end() ? 0 : node_id->second;
            event.args[2] = source_split_id;
            event.args[3] = ggml_exec_trace_intern(input->name);
            ggml_exec_trace_record(event);
        }
    }

    ggml_exec_trace_event fingerprint = {};
    fingerprint.timestamp_ns = timestamp;
    fingerprint.kind = GGML_EXEC_TRACE_EVENT_FINGERPRINT;
    fingerprint.graph_id = graph_id;
    fingerprint.args[0] = ops_shapes_hash;
    fingerprint.args[1] = backend_hash;
    fingerprint.args[2] = split_hash;
    fingerprint.args[3] = cross_backend_edges;
    ggml_exec_trace_record(fingerprint);

    sched->trace_graph_uid = graph->uid;
    sched->trace_topology_graph_id = graph_id;
    sched->trace_ops_shapes_hash = ops_shapes_hash;
    sched->trace_backend_hash = backend_hash;
    sched->trace_split_hash = split_hash;
    sched->trace_cross_backend_edges = cross_backend_edges;
    ggml_exec_trace_integrity_graph(
            ops_shapes_hash, backend_hash, split_hash, graph->n_nodes, sched->n_splits, cross_backend_edges);
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

// delphi: the source of a view input that covers all of a computed tensor (for example a reshape), or NULL. The scheduler copies
// that source and the node reads a view of the copy. Otherwise a split that reads both the tensor and the view receives the
// same bytes twice. A view of a weight or of a graph input keeps the old copy. So does a view of a tensor that an op of the
// graph writes in place (src_written): a backend receives one copy of a tensor per graph, taken at its first use, so views
// of such a tensor read at different points of the graph need copies of their own.
static struct ggml_tensor * ggml_backend_sched_full_view_src(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id,
        const std::vector<bool> & src_written) {
    struct ggml_tensor * src = t->view_src;
    if (!sched->view_alias || src == NULL || src->view_src != NULL || !ggml_is_view_op(t->op) || (src->flags & GGML_TENSOR_FLAG_INPUT) ||
            (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) || src_written[hash_id(src)]) {
        return NULL;
    }
    if (t->view_offs != 0 || t->type != src->type || ggml_nbytes(t) != ggml_nbytes(src) || !ggml_is_contiguous(t) || !ggml_is_contiguous(src)) {
        return NULL;
    }
    if (tensor_backend_id(src) != tensor_backend_id(t) || ggml_backend_sched_buffer_supported(sched, src, backend_id)) {
        return NULL;
    }
    return src;
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// delphi: staging ring decisions that split_graph needs (defined with the ring below)
static bool ggml_backend_sched_stage_redirects(ggml_backend_sched_t sched, const struct ggml_tensor * node,
        const struct ggml_tensor * weight, int backend_id);
static int  ggml_backend_sched_stage_split_weights(ggml_backend_sched_t sched);
static ggml_backend_buffer_t ggml_backend_sched_stage_anchor(ggml_backend_sched_t sched, int backend_id);

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        // delphi: a view of a computed tensor stays on the backend of that tensor, also when the user set another backend
        // for it (llama pins "l_last" to the device of its layer). The data of the view is where the source is computed:
        // on another backend, the splits that read the view copied it from the wrong backend, synchronized only that
        // backend and could read the source before it was computed
        if (node->view_src != NULL && ggml_is_view_op(node->op) && node->view_src->op != GGML_OP_NONE &&
                tensor_backend_id(node->view_src) != -1 && *cur_backend_id != tensor_backend_id(node->view_src)) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vpin");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    // views of input copies (ggml_backend_sched_full_view_src), with their split
    std::vector<std::pair<int, struct ggml_tensor *>> view_aliases;
    std::vector<const struct ggml_tensor *> view_alias_of; // the view each alias stands for
    // tensors whose memory an op other than a view writes (in-place ops, SET_ROWS, CPY into a view): such an op is a view of
    // the tensor with a non-view op
    std::vector<bool> view_src_written;
    if (sched->view_alias) {
        view_src_written.assign(sched->hash_set.size, false);
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (node->view_src != NULL && !ggml_is_view_op(node->op)) {
                view_src_written[hash_id(node->view_src)] = true;
            }
        }
    }
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        // delphi: weights of the current split that the staging ring redirects, and whether it copies another weight
        int  split_staged   = 0;
        bool split_unstaged = false;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            // with device-ordered copies, a result of another device also ends a split that has no inputs yet: the work
            // before it runs while the other device finishes
            const bool xdev = sched->xdev_events[cur_backend_id] != NULL;
            if (node_backend_id == cur_backend_id && (split->n_inputs > 0 || (xdev && i > split->i_start))) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    // a view of a weight (a slice of the experts or the heads) has no buffer until it is allocated
                    ggml_backend_buffer_t xdev_wbuf = xdev && src->view_src != NULL ? src->view_src->buffer : NULL;
                    if (xdev_wbuf != NULL && xdev_wbuf->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && !ggml_backend_buffer_is_host(xdev_wbuf)) {
                        // a weight of another GPU is ready from the start: with device-ordered copies it is copied with the
                        // split's other inputs, since a new split here would wait for the work queued on this device so far
                        continue;
                    }
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            if (xdev && !ggml_backend_buffer_is_host(src->buffer)) {
                                continue;
                            }
                            if (split->n_inputs == 0) {
                                continue;
                            }
                            // delphi: a weight that the staging ring redirects to its slot takes no copy memory, so the split
                            // keeps up to GGML_SCHED_STAGE_SPLIT_WEIGHTS of them and the routed FFN of a layer can fuse
                            if (!split_unstaged && split_staged < ggml_backend_sched_stage_split_weights(sched) &&
                                    ggml_backend_sched_stage_redirects(sched, node, src, cur_backend_id)) {
                                continue;
                            }
                            need_new_split = true;
                            break;
                        }
                    } else if (!(src->flags & GGML_TENSOR_FLAG_INPUT)) {
                        // a result of another device: its copy waits for that device, so start a new split and let
                        // the work before this node run meanwhile (a MoE split waits for the other device's experts)
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != -1 && src_backend_id != cur_backend_id && src_backend_id != sched->n_backends - 1) {
                            // delphi: with GGML_SCHED_STAGE_SPREAD, no new split for a view of an input or when the split already
                            // waits for that device (the copies share its wait, and the routed FFN of a layer stays in one split)
                            const bool spread = ggml_backend_sched_stage_spreading(sched);
                            const struct ggml_tensor * base = src;
                            while (spread && base->view_src != NULL) {
                                base = base->view_src;
                            }
                            bool is_input = false;
                            for (int k = 0; k < split->n_inputs && !is_input; k++) {
                                is_input = split->inputs[k] == src ||
                                    (spread && (split->inputs[k] == base || tensor_backend_id(split->inputs[k]) == src_backend_id));
                            }
                            if (!is_input) {
                                need_new_split = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
                split_staged = 0;
                split_unstaged = false;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (sched->input_async && (src->flags & GGML_TENSOR_FLAG_INPUT)) {
                    // A single-copy graph must also keep upload sources out of allocator reuse.
                    ggml_set_output(src);
                }
                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (sched_hv_copy(sched, src, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        sched_hv_mark_written(sched, src_id);
                        int n_graph_inputs = sched->n_graph_inputs++;
                        if (n_graph_inputs >= sched->graph_inputs_capacity) {
                            ggml_backend_sched_graph_inputs_grow(sched);
                        }
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    struct ggml_tensor * view = src;
                    struct ggml_tensor * view_src = ggml_backend_sched_full_view_src(sched, src, cur_backend_id, view_src_written);
                    if (view_src != NULL) {
                        src    = view_src;
                        src_id = hash_id(src);
                    }
                    // create a copy of the input in the split's backend
                    if (sched_hv_copy(sched, src, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        // delphi: the staging ring points a redirected copy at its slot, so the copy needs no memory of its own.
                        // ggml-alloc skips a tensor that has a buffer and data; the anchor gives it both until the ring sets data.
                        const bool redirected = view_src == NULL && ggml_backend_sched_stage_redirects(sched, node, src, cur_backend_id);
                        if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && ggml_backend_buffer_is_host(src->buffer)) {
                            split_staged   += redirected;
                            split_unstaged |= !redirected;
                        }
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            if (redirected) {
                                tensor_copy->buffer = ggml_backend_sched_stage_anchor(sched, cur_backend_id);
                                tensor_copy->data   = ggml_backend_buffer_get_base(tensor_copy->buffer);
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        sched_hv_mark_written(sched, src_id);
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = sched_hv_copy(sched, src, cur_backend_id, sched->cur_copy);
                    if (view_src != NULL) {
                        // the view of the copy is a node of graph_copy (below), so that ggml-alloc keeps the copy while the view is used.
                        // delphi: one alias per view and split, so that the consumers of a view still share one source tensor (the
                        // fused routed FFN matches gate and up by their common input)
                        struct ggml_tensor * alias = NULL;
                        for (size_t k = 0; k < view_aliases.size() && alias == NULL; k++) {
                            if (view_aliases[k].first == i_split && view_alias_of[k] == view) {
                                alias = view_aliases[k].second;
                            }
                        }
                        if (alias == NULL) {
                            alias = ggml_view_4d(sched->ctx, node->src[j], view->ne[0], view->ne[1], view->ne[2], view->ne[3],
                                    view->nb[1], view->nb[2], view->nb[3], 0);
                            ggml_format_name(alias, "%s#%s", ggml_backend_name(sched->backends[cur_backend_id]), view->name);
                            view_aliases.push_back({ i_split, alias });
                            view_alias_of.push_back(view);
                        }
                        node->src[j] = alias;
                    }
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // Pass 5 adds the inputs of a split in the order of their first use. A device without peer access may run the
    // staged copies into one device in submission order, so a copy whose source is ready early can wait behind a
    // copy whose source is still being computed, stalling the stream of its source device (e.g. a small tensor of
    // the first pipeline stage queued behind the stream hand-off of the second stage). Sources produced earlier in
    // the graph are ready earlier across pipelined devices, so copy the inputs in the graph order of their
    // producers; leafs have no producer and come first. This changes only the submission order of copies that
    // do not depend on each other, not which copies are made or what each split waits for.
    if (sched->input_order) {
        std::vector<int> producer(sched->hash_set.size, -1);
        for (int i = 0; i < graph->n_nodes; i++) {
            producer[hash_id(graph->nodes[i])] = i;
        }
        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            std::stable_sort(split->inputs, split->inputs + split->n_inputs, [&](ggml_tensor * a, ggml_tensor * b) {
                return producer[hash_id(a)] < producer[hash_id(b)];
            });
        }
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes + (int) view_aliases.size();

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = sched_hv_copy_id(sched, input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (const auto & it : view_aliases) {
            if (it.first == i) {
                assert(graph_copy->size > graph_copy->n_nodes);
                sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                graph_copy->nodes[graph_copy->n_nodes++] = it.second;
            }
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = sched_hv_copy_id(sched, id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = sched_hv_copy_id(sched, id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            const uint64_t wait_start = ggml_backend_sched_trace_span_begin();
            ggml_backend_synchronize(sched->backends[i]);
            ggml_backend_sched_trace_span_end(
                    sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                    GGML_EXEC_TRACE_WAIT_GRAPH_REALLOCATION, -1, i, GGML_EXEC_TRACE_ID_NONE,
                    GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);
        }

        // a failed reserve leaves a buffer unallocated, and allocating the graph in it then crashes
        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve the graph\n", __func__);
            return false;
        }
        sched->n_reserves++;
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// The graph allocator reuses memory only within one buffer, shared by the backends with the same buffer type, and a
// split reads a tensor on another backend without an input copy only when its backend supports that buffer type. Two
// backends for which neither holds cannot touch each other's split memory, so an input-less split on one of them does
// not need to wait for the previous split on the other. With pipeline parallelism, that wait would block the host
// on the previous ubatch of the other device and serialize the ubatches.
static bool ggml_backend_sched_splits_share_memory(ggml_backend_sched_t sched, int backend_a, int backend_b) {
    if (sched->prev_split_wait_all || sched->bufts[backend_a] == sched->bufts[backend_b]) {
        return true;
    }
    return ggml_backend_supports_buft(sched->backends[backend_b], sched->bufts[backend_a]) ||
           ggml_backend_supports_buft(sched->backends[backend_a], sched->bufts[backend_b]);
}

// delphi: staging ring (on unless GGML_SCHED_STAGE=0).
//
// A host weight that a split copies whole to a device (an offloaded expert tensor in a wide prefill batch) is
// otherwise copied when the host reaches that split, in the order of the compute stream, so the link and the device
// take turns, and a pageable copy also blocks the host thread. The ring instead uploads these weights ahead, in split
// order, into a few device slots. A staging thread copies each weight in chunks into pinned host bounce buffers with
// several memcpy threads, and the device copies each chunk into the slot on its own copy stream. When the host
// reaches the split, the compute stream waits for the upload and copies the slot into the split input, which the
// allocator placed in the compute buffer as usual.
//
// Invariants:
// - A slot is uploaded again only after the device-to-device copy that read its previous content has completed: the
//   staging thread waits for its record on the compute stream. A wait on an event that has no record yet would not
//   order anything, so the job that waits is queued only after the record (busy is cleared together with it).
// - The compute stream waits for the done event of a slot only after the staging thread has recorded it for this
//   upload (issued), for the same reason.
// - A bounce buffer is refilled only after the host has synchronized with the copy that read it.
// - The split input is written only by the device-to-device copy on the compute stream of its split, at the point
//   where the old copy path wrote it, so allocator reuse of the compute buffer keeps the same ordering.
// - The uploads read the weights, which nothing writes while graphs run.
#define GGML_SCHED_STAGE_MAX_SLOTS  16
#define GGML_SCHED_STAGE_MAX_BOUNCE 16
#define GGML_SCHED_STAGE_MIN_BYTES  (1u << 20)

// the CPUs next to a PCI device, from sysfs; empty if unknown
static std::vector<int> ggml_backend_sched_stage_local_cpus(const char * pci_id) {
    std::vector<int> cpus;
#ifdef __linux__
    if (pci_id == nullptr) {
        return cpus;
    }
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", pci_id);
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return cpus;
    }
    int first, last;
    char sep;
    while (fscanf(f, "%d", &first) == 1) {
        last = first;
        sep = (char) fgetc(f);
        if (sep == '-') {
            if (fscanf(f, "%d", &last) != 1) {
                break;
            }
            sep = (char) fgetc(f);
        }
        for (int c = first; c <= last && c < CPU_SETSIZE; c++) {
            cpus.push_back(c);
        }
        if (sep != ',') {
            break;
        }
    }
    fclose(f);
#else
    GGML_UNUSED(pci_id);
#endif
    return cpus;
}

// the thread stays inside the allowed CPUs (ggml-cpuset.h): the CPUs of the list that are allowed, else all allowed CPUs
static void ggml_backend_sched_stage_pin(const std::vector<int> & cpus) {
#ifdef __linux__
    std::vector<int> allowed(cpus);
    allowed.resize(ggml_cpuset_filter(allowed.data(), (int) allowed.size()));
    if (allowed.empty()) {
        allowed.resize(CPU_SETSIZE);
        allowed.resize(ggml_cpuset_list(allowed.data(), (int) allowed.size()));
    }
    if (allowed.empty()) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : allowed) {
        CPU_SET(c, &set);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    GGML_UNUSED(cpus);
#endif
}

// splits a memcpy over the calling thread and n - 1 helper threads
// Optional: a backend that converts staged weights into another layout can do it during the upload instead of in prepare, off
// the compute stream. pack_unit returns the bytes of one expert of the host weight if it wants that (uploads then come in
// whole experts), else 0; upload_packed uploads size bytes of the weight at src to slot + offset in that layout, records copied
// (optional) once src is read and done (optional) once they are ready. Found as "ggml_backend_stage_pack_unit" and
// "ggml_backend_stage_upload_packed".
typedef size_t (*ggml_backend_stage_pack_unit_t)(ggml_backend_t backend, const struct ggml_tensor * weight);
// prepacked: src holds the data in that layout already, made by pack_host ("ggml_backend_stage_pack_host", optional), which
// converts part ith of nth of size bytes (whole experts) of the weight from src to dst in host memory.
typedef void (*ggml_backend_stage_upload_packed_t)(ggml_backend_t backend, const struct ggml_tensor * weight, void * slot, size_t offset,
        const void * src, size_t size, bool prepacked, ggml_backend_event_t copied, ggml_backend_event_t done);


// page-locks the host weight buffers that the ring stages from (GGML_SCHED_STAGE_REGISTER), so that uploads go straight
// from the weights to the device, without the memcpy into the bounce buffers. hipHostRegister runs at ~1.2 GB/s (DeepSeek
// V4.1 Flash: 166 GiB in 142 s), and in the background it stalls the devices' queues for its whole duration, so each
// buffer is locked when the ring first stages from it, before any upload. Each buffer is one registration: HIP treats a
// registration as its own allocation and rejects any copy, also outside the ring, that crosses from one into the next.
struct ggml_backend_sched_stage_registrar {
    struct region {
        ggml_backend_buffer_t buffer;
        char * base; // page-aligned
        size_t size;
        bool locked;
    };

    ggml_backend_stage_register_t   lock   = nullptr;
    ggml_backend_stage_unregister_t unlock = nullptr;
    std::mutex mutex; // regions is read by the staging threads
    std::vector<region> regions;

    void add(ggml_backend_buffer_t buffer) {
        std::lock_guard<std::mutex> guard(mutex);
        for (const auto & r : regions) {
            if (r.buffer == buffer) {
                return;
            }
        }
        const uintptr_t page = 4096;
        const uintptr_t begin = ((uintptr_t) ggml_backend_buffer_get_base(buffer) + page - 1) & ~(page - 1);
        const uintptr_t end = ((uintptr_t) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer)) & ~(page - 1);
        region r = { buffer, (char *) begin, end > begin ? end - begin : 0, false };
        if (r.size > 0) {
            const int64_t t0 = ggml_time_us();
            r.locked = lock(r.base, r.size);
            if (r.locked) {
                GGML_LOG_INFO("%s: page-locked %.1f GiB of staged weights in %.1f s\n", "ggml_backend_sched_stage",
                        r.size/1073741824.0, (ggml_time_us() - t0)/1e6);
            }
        }
        regions.push_back(r);
    }

    // true if [src, src + size) lies in one page-locked region
    bool locked(const void * src, size_t size) {
        const char * s = (const char *) src;
        std::lock_guard<std::mutex> guard(mutex);
        for (const auto & r : regions) {
            if (r.locked && s >= r.base && s + size <= r.base + r.size) {
                return true;
            }
        }
        return false;
    }

    ~ggml_backend_sched_stage_registrar() {
        for (const auto & r : regions) {
            if (r.locked) {
                unlock(r.base);
            }
        }
    }
};

struct ggml_backend_sched_stage_item {
    struct ggml_tensor * input;     // host weight
    struct ggml_tensor * input_cpy; // its split input on the split backend
    int split_id;
    int backend_id;
    int slot;                       // -1 until queued
    uint64_t seq;                   // job number of the upload
    bool reused = false;
};

// GGML_SCHED_STAGE_TRACE=<file>: one line per upload job, "backend slot bytes queued start free issued" in microseconds on the clock
// of the execution trace (GGML_EXEC_TRACE), so the jobs can be placed against the device spans of the splits
static int64_t ggml_backend_sched_stage_trace_us(void) {
#if defined(CLOCK_MONOTONIC_RAW)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (int64_t) ts.tv_sec*1000000 + ts.tv_nsec/1000;
#else
    return ggml_time_us();
#endif
}

struct ggml_backend_sched_stage_job {
    int backend_id;
    int slot;
    bool wait_free;
    const void * src;
    size_t size;
    int64_t t_queued; // GGML_SCHED_STAGE_TRACE
    const struct ggml_tensor * weight; // the host weight, for upload_packed
    size_t unit;                       // upload_packed in whole units of this many bytes; 0: plain uploads
};


struct ggml_backend_sched_stage_ring {
    ggml_backend_stage_upload_t  upload  = nullptr;
    ggml_backend_stage_pack_unit_t     pack_unit     = nullptr;
    ggml_backend_stage_upload_packed_t upload_packed = nullptr;
    ggml_backend_stage_pack_host_t     pack_host     = nullptr;
    ggml_backend_stage_d2d_t     d2d     = nullptr;
    ggml_backend_stage_prepare_t prepare = nullptr;
    ggml_backend_stage_wait_free_t wait_free = nullptr;
    ggml_backend_buffer_t        anchor  = nullptr; // buffer and placeholder data of redirected copies (see split_graph)
    ggml_backend_buffer_t buffers[GGML_SCHED_STAGE_MAX_SLOTS] = {}; // null when the slots are lent
    void *                slots[GGML_SCHED_STAGE_MAX_SLOTS]   = {}; // device memory of each slot
    bool                  lent = false;                             // slots belong to the stage lender
    ggml_backend_event_t  done[GGML_SCHED_STAGE_MAX_SLOTS]    = {}; // upload finished, recorded on the copy stream
    ggml_backend_event_t  free[GGML_SCHED_STAGE_MAX_SLOTS]    = {}; // slot read, recorded on the compute stream
    bool busy[GGML_SCHED_STAGE_MAX_SLOTS]          = {};            // queued, and the read is not recorded yet
    bool free_recorded[GGML_SCHED_STAGE_MAX_SLOTS] = {};
    size_t slot_size = 0;
    int n_slots = 0; // slots in use: GGML_SCHED_STAGE_SLOTS, or fewer when the memory for all of them is not available
    int next = 0;
    // the slots that the last allocation wanted when it got fewer (0 if it got all): later graphs do not try again for
    // as many, which would drain the ring and lend the expert cache again for every graph, until the slots are dropped
    int short_of = 0;

    // ggml_backend_sched_stage_keep: the weight in each slot, which later graphs read without another upload
    const ggml_tensor * resident[GGML_SCHED_STAGE_MAX_SLOTS] = {};
    uint64_t resident_seq[GGML_SCHED_STAGE_MAX_SLOTS] = {};
    uint32_t readers[GGML_SCHED_STAGE_MAX_SLOTS] = {};
    bool prepared[GGML_SCHED_STAGE_MAX_SLOTS] = {}; // ring.prepare ran on the slot after its last upload

    // pinned host bounce buffers, used by the staging thread only; bounce == nullptr uploads from the weight directly
    ggml_backend_buffer_t bounce = nullptr;
    ggml_backend_event_t  chunk_ev[GGML_SCHED_STAGE_MAX_BOUNCE]  = {};
    ggml_backend_event_t  bounce_ev[GGML_SCHED_STAGE_MAX_BOUNCE] = {}; // last copy that reads the bounce buffer
    int bounce_next = 0;
};

// one staging thread per device, so that one device's uploads never wait behind another's
struct ggml_backend_sched_stage_worker {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv_job;
    std::condition_variable cv_issued;
    std::deque<ggml_backend_sched_stage_job> jobs;
    uint64_t n_pushed = 0; // jobs queued since the start
    uint64_t n_issued = 0; // jobs whose copies and done record are queued on the device
    bool stop = false;
    ggml_backend_sched_stage_copier copier;
    std::vector<int> cpus; // the thread and its copier run here: the CPUs next to the device

    // GGML_SCHED_STAGE_STATS=1: where the thread spends its time, printed when the stage is freed
    std::string name; // the backends can be gone when the stage is freed
    int64_t us_idle = 0, us_free = 0, us_bounce = 0, us_copy = 0, us_upload = 0;
    uint64_t bytes_copy = 0, bytes_direct = 0;
};

struct ggml_backend_sched_stage {
    bool enabled = true;
    bool keep = false;
    int  n_slots = 2;  // each slot is one layer's expert tensor in VRAM, which the expert cache does not get
    int  n_bounce = 4;
    size_t chunk_size = 64u << 20;
    bool use_bounce = true;
    std::vector<ggml_backend_sched_stage_item> items; // staged inputs of the current graph, in split order
    size_t n_queued   = 0;
    size_t n_consumed = 0;
    ggml_backend_sched_stage_ring rings[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS] = {};
    std::unique_ptr<ggml_backend_sched_stage_worker> workers[GGML_SCHED_MAX_BACKENDS]; // null for backends without a ring
    std::unique_ptr<ggml_backend_sched_stage_registrar> registrar; // GGML_SCHED_STAGE_REGISTER
    // GGML_SCHED_STAGE_REDIRECT: a split reads its staged inputs in their slots, which it frees once its work is queued
    bool redirect = true;
    struct held { ggml_tensor * input_cpy; void * data; int backend_id; int slot; };
    std::vector<held> holding;
    ggml_backend_sched_stage_lend_t lend = nullptr;
    void * lend_user = nullptr;
    int split_weights = 1; // redirected weights per split (GGML_SCHED_STAGE_SPLIT_WEIGHTS)
    bool spread = true;    // GGML_SCHED_STAGE_SPREAD
    // GGML_SCHED_STAGE_KEEP_SPREAD (default 0): spread in keep mode too. A kept bank serves several graphs, so one link is enough,
    // and on one GPU the activations do not cross to another for each layer.
    bool keep_spread = false;
    int min_tokens = 512;  // GGML_SCHED_STAGE_MIN_TOKENS
    int64_t cpu_us = 0;    // time of the CPU splits (ggml_backend_sched_stage_stats)
    bool blocked[GGML_SCHED_MAX_BACKENDS] = {}; // ggml_backend_sched_stage_allow
};

static void ggml_backend_sched_stage_run_job(
        ggml_backend_sched_stage * stage, ggml_backend_sched_stage_worker * w, const ggml_backend_sched_stage_job & job) {
    ggml_backend_sched_stage_ring & ring = stage->rings[job.backend_id];
    ggml_backend_t backend = stage->backends[job.backend_id];
    int64_t t = ggml_time_us();
    const int64_t t_start = ggml_backend_sched_stage_trace_us();
    auto lap = [&](int64_t & acc) { const int64_t now = ggml_time_us(); acc += now - t; t = now; };
    if (job.wait_free) {
        // a host wait: HIP rejects a copy stream wait on an event of the compute stream while that stream captures.
        // Older HIP also rejects the host wait then, which ring.wait_free avoids.
        if (ring.wait_free != nullptr) {
            ring.wait_free(ring.free[job.slot]);
        } else {
            ggml_backend_event_synchronize(ring.free[job.slot]);
        }
    }
    lap(w->us_free);
    const int64_t t_free = ggml_backend_sched_stage_trace_us();
    static FILE * trace = getenv("GGML_SCHED_STAGE_TRACE") ? fopen(getenv("GGML_SCHED_STAGE_TRACE"), "w") : nullptr;
    struct trace_end {
        FILE * f; const ggml_backend_sched_stage_job & job; int64_t t_start, t_free;
        ~trace_end() { if (f) { fprintf(f, "%d %d %zu %lld %lld %lld %lld\n", job.backend_id, job.slot, job.size, (long long) job.t_queued,
                (long long) t_start, (long long) t_free, (long long) ggml_backend_sched_stage_trace_us()); fflush(f); } }
    } trace_guard { trace, job, t_start, t_free };
    char * dst = (char *) ring.slots[job.slot];
    if (job.unit > 0) {
        // whole units per chunk, straight from page-locked weights or through the bounce buffers
        const size_t chunk = std::max(job.unit, stage->chunk_size / job.unit * job.unit);
        const bool direct = ring.bounce == nullptr || (stage->registrar && stage->registrar->locked(job.src, job.size));
        GGML_ASSERT(job.size % job.unit == 0 && (direct || chunk <= stage->chunk_size));
        char * bounce = direct ? nullptr : (char *) ggml_backend_buffer_get_base(ring.bounce);
        for (size_t off = 0; off < job.size; off += chunk) {
            const size_t n = std::min(chunk, job.size - off);
            const bool last = off + n >= job.size;
            if (direct) {
                ring.upload_packed(backend, job.weight, dst, off, (const char *) job.src + off, n, false, nullptr, last ? ring.done[job.slot] : nullptr);
                w->bytes_direct += n;
                lap(w->us_upload);
                continue;
            }
            const int c = ring.bounce_next;
            ring.bounce_next = (c + 1) % stage->n_bounce;
            if (ring.bounce_ev[c] != nullptr) {
                ggml_backend_event_synchronize(ring.bounce_ev[c]);
            }
            lap(w->us_bounce);
            if (ring.pack_host != nullptr) {
                w->copier.pack(bounce + c*stage->chunk_size, (const char *) job.src + off, n, ring.pack_host, job.weight);
            } else {
                w->copier.copy(bounce + c*stage->chunk_size, (const char *) job.src + off, n);
            }
            lap(w->us_copy);
            // the bounce buffer is free once the device read it, before the chunk is packed
            ring.upload_packed(backend, job.weight, dst, off, bounce + c*stage->chunk_size, n, ring.pack_host != nullptr,
                    ring.chunk_ev[c], last ? ring.done[job.slot] : nullptr);
            ring.bounce_ev[c] = ring.chunk_ev[c];
            lap(w->us_upload);
            w->bytes_copy += n;
        }
        return;
    }
    if (stage->registrar && stage->registrar->locked(job.src, job.size)) {
        // straight from the page-locked weights
        ring.upload(backend, dst, job.src, job.size, ring.done[job.slot]);
        w->bytes_direct += job.size;
        lap(w->us_upload);
        return;
    }
    if (ring.bounce == nullptr) {
        ring.upload(backend, dst, job.src, job.size, ring.done[job.slot]);
        return;
    }
    char * bounce = (char *) ggml_backend_buffer_get_base(ring.bounce);
    for (size_t off = 0; off < job.size; off += stage->chunk_size) {
        const size_t n = std::min(stage->chunk_size, job.size - off);
        const int c = ring.bounce_next;
        ring.bounce_next = (c + 1) % stage->n_bounce;
        if (ring.bounce_ev[c] != nullptr) {
            ggml_backend_event_synchronize(ring.bounce_ev[c]);
        }
        lap(w->us_bounce);
        w->copier.copy(bounce + c*stage->chunk_size, (const char *) job.src + off, n);
        lap(w->us_copy);
        ggml_backend_event_t ev = off + n >= job.size ? ring.done[job.slot] : ring.chunk_ev[c];
        ring.upload(backend, dst + off, bounce + c*stage->chunk_size, n, ev);
        ring.bounce_ev[c] = ev;
        lap(w->us_upload);
        w->bytes_copy += n;
    }
}

static void ggml_backend_sched_stage_work(ggml_backend_sched_stage * stage, ggml_backend_sched_stage_worker * w) {
    ggml_backend_sched_stage_pin(w->cpus);
    for (;;) {
        ggml_backend_sched_stage_job job;
        {
            const int64_t t_idle = ggml_time_us();
            std::unique_lock<std::mutex> lock(w->mutex);
            w->cv_job.wait(lock, [&] { return w->stop || !w->jobs.empty(); });
            w->us_idle += ggml_time_us() - t_idle;
            if (w->jobs.empty()) {
                return;
            }
            job = w->jobs.front();
            w->jobs.pop_front();
        }
        ggml_backend_sched_stage_run_job(stage, w, job);
        {
            std::lock_guard<std::mutex> lock(w->mutex);
            w->n_issued++;
        }
        w->cv_issued.notify_all();
    }
}

// wait until the staging thread of the device has queued the copies of its job seq on the device
static void ggml_backend_sched_stage_wait_issued(ggml_backend_sched_stage_worker * w, uint64_t seq) {
    std::unique_lock<std::mutex> lock(w->mutex);
    w->cv_issued.wait(lock, [&] { return w->n_issued > seq; });
}

static ggml_backend_sched_stage * ggml_backend_sched_stage_new(ggml_backend_sched_t sched) {
    // delphi: on by default (GGML_SCHED_STAGE=0 turns it off). It does nothing without routed experts in host memory
    const char * env = getenv("GGML_SCHED_STAGE");
    if (env != nullptr && atoi(env) == 0) {
        return nullptr;
    }
    auto * stage = new ggml_backend_sched_stage;
    if (const char * e = getenv("GGML_SCHED_STAGE_SLOTS")) {
        stage->n_slots = std::max(1, std::min(GGML_SCHED_STAGE_MAX_SLOTS, atoi(e)));
    }
    if (const char * e = getenv("GGML_SCHED_STAGE_REDIRECT")) {
        stage->redirect = atoi(e) != 0;
    }
    if (const char * e = getenv("GGML_SCHED_STAGE_BOUNCE")) {
        stage->n_bounce = std::min(GGML_SCHED_STAGE_MAX_BOUNCE, atoi(e));
        stage->use_bounce = stage->n_bounce > 0;
        stage->n_bounce = std::max(2, stage->n_bounce);
    }
    if (const char * e = getenv("GGML_SCHED_STAGE_CHUNK_MB")) {
        stage->chunk_size = (size_t) std::max(1, atoi(e)) << 20;
    }
    int n_threads = 0; // per device; 0: 8 split over the staging devices
    if (const char * e = getenv("GGML_SCHED_STAGE_THREADS")) {
        n_threads = std::max(1, atoi(e));
    }
    // run the staging threads next to the device: left to the scheduler, they land on both sockets of a NUMA host,
    // and filling the bounce buffers ran at 26 GB/s instead of 35 GB/s, below the PCIe 5.0 x16 link rate
    bool pin_ok = true;
    if (const char * e = getenv("GGML_SCHED_STAGE_PIN")) {
        pin_ok = atoi(e) != 0;
    }
    int n_dev = 0;
    for (int b = 0; b < sched->n_backends; b++) {
        stage->backends[b] = sched->backends[b];
        ggml_backend_dev_t dev = ggml_backend_get_device(sched->backends[b]);
        if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        ggml_backend_sched_stage_ring & ring = stage->rings[b];
        ring.upload = (ggml_backend_stage_upload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_upload");
        ring.d2d    = (ggml_backend_stage_d2d_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_d2d");
        ring.prepare = (ggml_backend_stage_prepare_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_prepare");
        ring.wait_free = (ggml_backend_stage_wait_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_wait_free");
        if (const char * e = getenv("GGML_SCHED_STAGE_WAIT_FREE"); e != nullptr && atoi(e) == 0) {
            ring.wait_free = nullptr;
        }
        ring.pack_unit     = (ggml_backend_stage_pack_unit_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_pack_unit");
        ring.upload_packed = (ggml_backend_stage_upload_packed_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_upload_packed");
        ring.pack_host     = (ggml_backend_stage_pack_host_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_pack_host");
        if (ring.pack_unit == nullptr || ring.upload_packed == nullptr) {
            ring.pack_unit = nullptr;
            ring.upload_packed = nullptr;
            ring.pack_host = nullptr;
        }
        if (const char * e = getenv("GGML_SCHED_STAGE_PACK_HOST"); e != nullptr && atoi(e) == 0) {
            ring.pack_host = nullptr;
        }
        if (ring.upload == nullptr || ring.d2d == nullptr) {
            ring.upload = nullptr;
            continue;
        }
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);
        stage->workers[b].reset(new ggml_backend_sched_stage_worker);
        stage->workers[b]->name = ggml_backend_name(sched->backends[b]);
        if (pin_ok) {
            stage->workers[b]->cpus = ggml_backend_sched_stage_local_cpus(props.device_id);
        }
        n_dev++;
    }
    if (n_dev == 0) {
        if (env != nullptr) {
            GGML_LOG_WARN("%s: GGML_SCHED_STAGE=1, but no backend supports staging\n", __func__);
        }
        stage->enabled = false;
    }
    // one layer's routed gate/up/down per split, while another split's weights can still upload
    stage->split_weights = stage->n_slots >= 6 ? 3 : 1;
    if (const char * e = getenv("GGML_SCHED_STAGE_SPREAD")) {
        stage->spread = atoi(e) != 0;
    }
    if (const char * e = getenv("GGML_SCHED_STAGE_KEEP_SPREAD")) {
        stage->keep_spread = atoi(e) != 0;
    }
    if (const char * e = getenv("GGML_SCHED_STAGE_MIN_TOKENS")) {
        stage->min_tokens = std::max(0, atoi(e));
    }
    if (const char * e = getenv("GGML_SCHED_STAGE_SPLIT_WEIGHTS")) {
        stage->split_weights = std::max(1, std::min(stage->n_slots, atoi(e)));
    }
    if (n_threads == 0) {
        // packing on the host is ~2x the work of the plain copy (8 threads: 16 vs 33 GB/s, 16 threads: 27 vs 42)
        bool pack_host = false;
        for (const auto & ring : stage->rings) {
            pack_host = pack_host || ring.pack_host != nullptr;
        }
        n_threads = std::max(2, (pack_host ? 16 : 8)/std::max(1, n_dev));
    }
    if (const char * e = getenv("GGML_SCHED_STAGE_REGISTER"); e != nullptr && atoi(e) != 0 && n_dev > 0) {
        for (int b = 0; b < sched->n_backends && stage->registrar == nullptr; b++) {
            if (stage->workers[b] == nullptr) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(sched->backends[b]));
            auto lock   = (ggml_backend_stage_register_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_register");
            auto unlock = (ggml_backend_stage_unregister_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_stage_unregister");
            if (lock != nullptr && unlock != nullptr) {
                stage->registrar.reset(new ggml_backend_sched_stage_registrar);
                stage->registrar->lock   = lock;
                stage->registrar->unlock = unlock;
            }
        }
    }
    for (auto & w : stage->workers) {
        if (w == nullptr) {
            continue;
        }
        if (stage->use_bounce) {
            w->copier.start(n_threads, w->cpus, ggml_backend_sched_stage_pin);
        }
        w->thread = std::thread(ggml_backend_sched_stage_work, stage, w.get());
    }
    return stage;
}

// wait until the staging threads have queued every job on their devices
static void ggml_backend_sched_stage_drain(ggml_backend_sched_stage * stage) {
    for (auto & w : stage->workers) {
        if (w != nullptr) {
            std::unique_lock<std::mutex> lock(w->mutex);
            w->cv_issued.wait(lock, [&] { return w->n_issued == w->n_pushed; });
        }
    }
}

// wait until no queued upload writes and no queued work reads a slot, then forget the slots (lent slots are not freed)
static void ggml_backend_sched_stage_drop_slots(ggml_backend_sched_stage_ring & ring) {
    for (int s = 0; s < GGML_SCHED_STAGE_MAX_SLOTS; s++) {
        if (ring.done[s] != nullptr) {
            ggml_backend_event_synchronize(ring.done[s]);
        }
        if (ring.free[s] != nullptr && ring.free_recorded[s]) {
            ggml_backend_event_synchronize(ring.free[s]);
        }
        ggml_backend_buffer_free(ring.buffers[s]);
        ring.buffers[s] = nullptr;
        ring.slots[s] = nullptr;
        ring.busy[s] = false;
        ring.free_recorded[s] = false;
        ring.resident[s] = nullptr;
        ring.readers[s] = 0;
        ring.prepared[s] = false;
    }
    ring.lent = false;
    ring.slot_size = 0;
    ring.n_slots = 0;
    ring.next = 0;
    ring.short_of = 0;
}

static void ggml_backend_sched_stage_release_ring(ggml_backend_sched_stage_ring & ring) {
    ggml_backend_sched_stage_drop_slots(ring);
    for (int c = 0; c < GGML_SCHED_STAGE_MAX_BOUNCE; c++) {
        if (ring.bounce_ev[c] != nullptr) {
            ggml_backend_event_synchronize(ring.bounce_ev[c]);
        }
        ring.bounce_ev[c] = nullptr;
    }
    ggml_backend_buffer_free(ring.bounce);
    ring.bounce = nullptr;
    ring.bounce_next = 0;
}

static void ggml_backend_sched_stage_free(ggml_backend_sched_stage * stage) {
    if (stage == nullptr) {
        return;
    }
    const char * stats = getenv("GGML_SCHED_STAGE_STATS");
    for (int b = 0; b < GGML_SCHED_MAX_BACKENDS; b++) {
        auto & w = stage->workers[b];
        if (w == nullptr) {
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(w->mutex);
            w->stop = true;
        }
        w->cv_job.notify_all();
        w->thread.join();
        if (stats != nullptr && atoi(stats) != 0) {
            GGML_LOG_WARN("%s: %s: memcpy %.1f GB in %.2f s (%.1f GB/s), direct %.1f GB; wait free %.2f s, bounce %.2f s, "
                    "upload calls %.2f s, idle %.2f s\n", __func__, w->name.c_str(),
                    w->bytes_copy/1e9, w->us_copy/1e6, w->us_copy > 0 ? w->bytes_copy/1e3/w->us_copy : 0.0,
                    w->bytes_direct/1e9, w->us_free/1e6, w->us_bounce/1e6, w->us_upload/1e6, w->us_idle/1e6);
        }
    }
    for (auto & ring : stage->rings) {
        ggml_backend_sched_stage_release_ring(ring);
        for (int s = 0; s < GGML_SCHED_STAGE_MAX_SLOTS; s++) {
            ggml_backend_event_free(ring.done[s]);
            ggml_backend_event_free(ring.free[s]);
        }
        for (int c = 0; c < GGML_SCHED_STAGE_MAX_BOUNCE; c++) {
            ggml_backend_event_free(ring.chunk_ev[c]);
        }
        ggml_backend_buffer_free(ring.anchor);
    }
    delete stage;
}

// the host weight `weight` of `node` that the copy loop would copy whole to the split backend
static bool ggml_backend_sched_stage_whole(
        ggml_backend_sched_t sched, const struct ggml_tensor * node, const struct ggml_tensor * weight, int backend_id) {
    if ((weight->flags & GGML_TENSOR_FLAG_INPUT) || weight->buffer == nullptr ||
            ggml_backend_buffer_get_usage(weight->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(weight->buffer) || ggml_nbytes(weight) < GGML_SCHED_STAGE_MIN_BYTES ||
            sched->stage->rings[backend_id].upload == nullptr || sched->stage->blocked[backend_id]) {
        return false;
    }
    if (node != nullptr && node->op == GGML_OP_MUL_MAT_ID && (node->src[0] == weight || sched_hv_copy(sched, (ggml_tensor *) weight, backend_id, sched->cur_copy) == node->src[0])) {
        // the copy loop copies only the used experts when it expects some experts to stay unused
        const struct ggml_tensor * ids = node->src[2];
        return ggml_offload_cost_expected_active_experts(weight->ne[2], ids->ne[1], ids->ne[0]) >= weight->ne[2] - 0.5;
    }
    return true;
}

static bool ggml_backend_sched_stage_eligible(
        ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split,
        const struct ggml_tensor * input, const struct ggml_tensor * input_cpy) {
    // a redirected copy has no memory of its own: it must be staged
    if (input_cpy->buffer != nullptr && input_cpy->buffer == sched->stage->rings[split->backend_id].anchor) {
        return true;
    }
    const struct ggml_tensor * node = split->graph.n_nodes > 0 ? split->graph.nodes[0] : nullptr;
    if (node != nullptr && !(node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy)) {
        node = nullptr;
    }
    return ggml_backend_sched_stage_whole(sched, node, input, split->backend_id);
}

static bool ggml_backend_sched_stage_redirects(ggml_backend_sched_t sched, const struct ggml_tensor * node,
        const struct ggml_tensor * weight, int backend_id) {
    ggml_backend_sched_stage * stage = sched->stage;
    return stage != nullptr && stage->enabled && stage->redirect && sched->callback_eval == nullptr &&
        ggml_backend_sched_stage_whole(sched, node, weight, backend_id);
}

// GGML_SCHED_STAGE_SPREAD=1: a redirected (staged) MUL_MAT_ID whose weight is layer N's ("blk.N.") runs on staging GPU N % n, in backend
// order, so each GPU's link streams every n-th layer while the others compute; gate/up/down of a layer stay together. Anything else,
// and weights without a layer number, keep the backend chosen for them.
static int ggml_backend_sched_stage_spread(ggml_backend_sched_t sched, const struct ggml_tensor * op,
        const struct ggml_tensor * weight, int backend_id) {
    ggml_backend_sched_stage * stage = sched->stage;
    int layer = -1;
    if (stage == nullptr || !stage->spread || (stage->keep && !stage->keep_spread) || op->op != GGML_OP_MUL_MAT_ID || op->src[0] != weight ||
            sscanf(weight->name, "blk.%d.", &layer) != 1 || layer < 0 ||
            !ggml_backend_sched_stage_redirects(sched, op, weight, backend_id)) {
        return backend_id;
    }
    int gpus[GGML_SCHED_MAX_BACKENDS];
    int n = 0;
    for (int b = 0; b < sched->n_backends - 1; b++) {
        if (stage->rings[b].upload != nullptr && !stage->blocked[b] && ggml_backend_supports_op(sched->backends[b], op)) {
            gpus[n++] = b;
        }
    }
    return n > 1 ? gpus[layer % n] : backend_id;
}

static bool ggml_backend_sched_stage_spreading(ggml_backend_sched_t sched) {
    return sched->stage != nullptr && sched->stage->spread;
}

// GGML_SCHED_STAGE_MIN_TOKENS (default 512; llama measures it at warmup): with staging on, a routed-expert multiply (MUL_MAT_ID)
// on host weights runs on the CPU for a batch of fewer tokens, and on a GPU, which streams the experts, for a wider one.
// Streaming costs the same for any batch, as all experts cross the link, so a short batch is faster on the CPU. The GPU is the
// one holding the batch's activations if it can stage, else the first that can. 0 leaves the choice to the offload cost model.
// -1: not decided here.
static int ggml_backend_sched_stage_route(ggml_backend_sched_t sched, const struct ggml_tensor * op,
        const struct ggml_tensor * weight, int cpu_backend_id) {
    const ggml_backend_sched_stage * stage = sched->stage;
    if (stage == nullptr || !stage->enabled || stage->min_tokens <= 0 || op->op != GGML_OP_MUL_MAT_ID || op->src[0] != weight) {
        return -1;
    }
    // keep mode (a layer-bank prefill window): a short chunk of the window reads the bank that its first chunk made resident
    if (!stage->keep && op->src[2]->ne[1] < stage->min_tokens) {
        return cpu_backend_id;
    }
    int preferred = -1;
    ggml_backend_sched_cpu_input_bytes(sched, op, weight, &preferred);
    if (preferred >= 0 && preferred < cpu_backend_id && stage->rings[preferred].upload != nullptr && !stage->blocked[preferred] &&
            ggml_backend_supports_op(sched->backends[preferred], op)) {
        return preferred;
    }
    bool any_blocked = false;
    for (int b = 0; b < cpu_backend_id; b++) {
        any_blocked |= stage->rings[b].upload != nullptr && stage->blocked[b];
        if (stage->rings[b].upload != nullptr && !stage->blocked[b] && ggml_backend_supports_op(sched->backends[b], op)) {
            return b;
        }
    }
    return any_blocked ? cpu_backend_id : -1;
}

static int ggml_backend_sched_stage_split_weights(ggml_backend_sched_t sched) {
    return sched->stage != nullptr ? sched->stage->split_weights : 1;
}

static ggml_backend_buffer_t ggml_backend_sched_stage_anchor(ggml_backend_sched_t sched, int backend_id) {
    ggml_backend_sched_stage_ring & ring = sched->stage->rings[backend_id];
    if (ring.anchor == nullptr) {
        ring.anchor = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(sched->backends[backend_id]), 256);
        GGML_ASSERT(ring.anchor != nullptr);
        // the copies hold weights: backends may treat them as such (e.g. the fused HIP G128 FFN)
        ggml_backend_buffer_set_usage(ring.anchor, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    return ring.anchor;
}

// keep mode: a slot for a new upload, a free one first, then one whose weight no item of the current graph from
// item `from` on and no weight in `skip` reads; -1 if there is none. With `force`, when every free slot holds a weight
// that a later item reads, the one read last: a ring with fewer slots than the graph's weights (the memory for more
// was not available) uploads such a weight again instead of stalling.
static int ggml_backend_sched_stage_evictable(const ggml_backend_sched_stage * stage, const ggml_backend_sched_stage_ring & ring,
        struct ggml_tensor * const * skip = nullptr, int n_skip = 0, size_t from = 0, bool force = false) {
    int stale = -1;
    int last  = -1;
    size_t last_use = 0;
    for (int s = 0; s < ring.n_slots; ++s) {
        if (ring.busy[s]) {
            continue;
        }
        if (ring.resident[s] == nullptr) {
            return s;
        }
        size_t first_use = SIZE_MAX;
        for (size_t i = from; i < stage->items.size() && first_use == SIZE_MAX; ++i) {
            if (stage->items[i].input == ring.resident[s]) {
                first_use = i;
            }
        }
        bool used = first_use != SIZE_MAX;
        for (int i = 0; i < n_skip; ++i) {
            used |= skip[i] == ring.resident[s];
        }
        if (!used && stale < 0) {
            stale = s;
        }
        if (force && (last < 0 || first_use > last_use)) {
            last = s;
            last_use = first_use;
        }
    }
    return stale >= 0 ? stale : last;
}

// queue the upload of a weight into a slot of its ring
static uint64_t ggml_backend_sched_stage_push(ggml_backend_sched_t sched, int backend_id, int slot, const struct ggml_tensor * input) {
    ggml_backend_sched_stage * stage = sched->stage;
    ggml_backend_sched_stage_ring & ring = stage->rings[backend_id];
    ggml_backend_sched_stage_worker * w = stage->workers[backend_id].get();
    uint64_t seq;
    {
        std::lock_guard<std::mutex> lock(w->mutex);
        // packed uploads only where the split reads the slot itself: a copied slot would be prepared (packed) again
        size_t unit = ring.pack_unit != nullptr && stage->redirect && sched->callback_eval == nullptr ?
                ring.pack_unit(stage->backends[backend_id], input) : 0;
        if (unit > stage->chunk_size || ggml_nbytes(input) % std::max<size_t>(unit, 1) != 0) {
            unit = 0; // a unit must fit a bounce buffer
        }
        w->jobs.push_back({ backend_id, slot, ring.free_recorded[slot], input->data, ggml_nbytes(input),
                ggml_backend_sched_stage_trace_us(), input, unit });
        seq = w->n_pushed++;
    }
    ring.resident[slot] = stage->keep ? input : nullptr;
    ring.resident_seq[slot] = seq;
    ring.prepared[slot] = false;
    w->cv_job.notify_one();
    return seq;
}

// queue uploads for the next items while their slots are free
static void ggml_backend_sched_stage_pump(ggml_backend_sched_t sched) {
    ggml_backend_sched_stage * stage = sched->stage;
    while (stage->n_queued < stage->items.size()) {
        ggml_backend_sched_stage_item & item = stage->items[stage->n_queued];
        ggml_backend_sched_stage_ring & ring = stage->rings[item.backend_id];
        int slot = ring.next;
        if (stage->keep) {
            int found = -1;
            for (int s = 0; s < ring.n_slots && found < 0; ++s) {
                found = ring.resident[s] == item.input ? s : -1;
            }
            if (found >= 0) {
                item.slot = found;
                item.seq = ring.resident_seq[found];
                item.reused = true;
                ring.readers[found]++;
                ring.busy[found] = true;
                stage->n_queued++;
                continue;
            }
            // the items before this one have their slots, so only the later ones keep a weight
            slot = ggml_backend_sched_stage_evictable(stage, ring, nullptr, 0, stage->n_queued + 1, true);
            if (slot < 0) {
                break;
            }
        }
        if (ring.busy[slot]) {
            break;
        }
        item.seq = ggml_backend_sched_stage_push(sched, item.backend_id, slot, item.input);
        ring.readers[slot] = 1;
        ring.busy[slot] = true;
        ring.next = (slot + 1) % ring.n_slots;
        item.slot = slot;
        stage->n_queued++;
    }
}

// collect the staged inputs of the graph, size the rings and queue the first uploads
static void ggml_backend_sched_stage_begin(ggml_backend_sched_t sched) {
    ggml_backend_sched_stage * stage = sched->stage;
    // an interrupted graph can leave jobs behind; let them reach the device before the slots are reused. In keep mode the slots
    // record their weights, so that the uploads of a prefetch continue while the next graph starts, and only a ring that
    // changes its size drains.
    if (!stage->keep) {
        ggml_backend_sched_stage_drain(stage);
    }
    stage->items.clear();
    stage->n_queued = 0;
    stage->n_consumed = 0;
    if (!stage->enabled) {
        return;
    }

    size_t need[GGML_SCHED_MAX_BACKENDS] = { 0 };
    int split_slots[GGML_SCHED_MAX_BACKENDS] = { 0 }; // most staged inputs of one split
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        const struct ggml_backend_sched_split * split = &sched->splits[split_id];
        for (int i = 0; i < split->n_inputs; i++) {
            struct ggml_tensor * input = split->inputs[i];
            struct ggml_tensor * input_cpy = sched_hv_copy(sched, input, split->backend_id, sched->cur_copy);
            if (!ggml_backend_sched_stage_eligible(sched, split, input, input_cpy)) {
                continue;
            }
            stage->items.push_back({ input, input_cpy, split_id, split->backend_id, -1, 0 });
            // with GGML_SCHED_STAGE_REDIRECT a split holds a slot for each of its staged inputs until its work is queued
            {
                int n_split = 0;
                for (const auto & it : stage->items) {
                    n_split += it.split_id == split_id && it.backend_id == split->backend_id;
                }
                split_slots[split->backend_id] = std::max(split_slots[split->backend_id], n_split);
            }
            if (stage->registrar) {
                stage->registrar->add(input->buffer);
            }
            // with GGML_SCHED_STAGE_REDIRECT a kernel reads the slot as its input tensor, padding included
            size_t size = ggml_nbytes(input);
            if (stage->redirect) {
                size = std::max(size, ggml_backend_buft_get_alloc_size(
                        ggml_backend_get_default_buffer_type(sched->backends[split->backend_id]), input_cpy));
            }
            need[split->backend_id] = std::max(need[split->backend_id], size);
        }
    }

    // keep mode: a layer's gate/up/down stay, and the next layer's bank uploads beside them unless spread puts it on another ring
    const bool keep_spread = stage->spread && stage->keep_spread;
    const int wanted_slots = stage->keep ? std::max(keep_spread ? 3 : 6, stage->n_slots) : stage->n_slots;
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_sched_stage_ring & ring = stage->rings[b];
        if (keep_spread && stage->keep && need[b] == 0 && ring.n_slots > 0 && ring.n_slots < wanted_slots) {
            // a ring that this graph does not use grows too, for the prefetch of a later layer
            need[b] = ring.slot_size;
        }
        // the uploads of interrupted items still complete in copy stream order before any later one
        for (int s = 0; s < GGML_SCHED_STAGE_MAX_SLOTS; s++) {
            ring.busy[s] = false;
            ring.readers[s] = 0;
        }
        // a split that holds more slots than the ring has would wait for itself
        const int n_hold = stage->redirect ? std::min(split_slots[b], GGML_SCHED_STAGE_MAX_SLOTS) : 1;
        const int n_want = std::max(wanted_slots, n_hold);
        if (need[b] == 0 || (need[b] <= ring.slot_size && ring.n_slots >= n_hold &&
                (!stage->keep || ring.n_slots >= wanted_slots || ring.short_of >= n_want))) {
            continue;
        }
        if (stage->keep) {
            ggml_backend_sched_stage_drain(stage);
        }
        ggml_backend_sched_stage_release_ring(ring);
        ggml_backend_t backend = sched->backends[b];
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        // Fewer slots when the memory for all of them is not there (an expert cache tier too small to lend them, next to a
        // draft model): a split holds at most split_weights slots at a time, so that many still run, with less overlap. The
        // graph is already split, and a redirected input has no memory but its slot, so giving up is not an option.
        const int n_min = std::max(n_hold, 1);
        bool ok = false;
        int n = n_want;
        for (; n >= n_min && !ok; n--) {
            // lent memory first; the lender cannot lend part of the slots
            ring.lent = stage->lend != nullptr && stage->lend(stage->lend_user, backend, n, need[b], ring.slots);
            if (!ring.lent) {
                std::fill(std::begin(ring.slots), std::end(ring.slots), nullptr);
            }
            ok = true;
            for (int s = 0; s < n && ok; s++) {
                if (!ring.lent) {
                    ring.buffers[s] = ggml_backend_buft_alloc_buffer(buft, need[b]);
                    ring.slots[s] = ring.buffers[s] != nullptr ? ggml_backend_buffer_get_base(ring.buffers[s]) : nullptr;
                }
                ok = ring.slots[s] != nullptr;
                if (ok && ring.done[s] == nullptr) {
                    ring.done[s] = ggml_backend_event_new(dev);
                    ring.free[s] = ggml_backend_event_new(dev);
                    ok = ring.done[s] != nullptr && ring.free[s] != nullptr;
                }
            }
            if (!ok) {
                ggml_backend_sched_stage_drop_slots(ring);
            }
        }
        if (!ok) {
            ggml_backend_sched_stage_release_ring(ring);
            bool redirected = false;
            for (const auto & item : stage->items) {
                redirected |= item.backend_id == b && item.input_cpy->buffer == ring.anchor;
            }
            if (redirected) {
                GGML_ABORT("staging: cannot allocate %d slot(s) of %.1f MiB on %s for the redirected inputs of this graph; "
                        "free memory on the device (LLAMA_MOE_CACHE_AUTO_MARGIN_MIB, a smaller LLAMA_MOE_CACHE_MIB) or set "
                        "GGML_SCHED_STAGE_REDIRECT=0", n_min, need[b]/1048576.0, ggml_backend_name(backend));
            }
            GGML_LOG_WARN("%s: staging disabled: cannot allocate %d slots of %.1f MiB on %s\n", __func__,
                    n_min, need[b]/1048576.0, ggml_backend_name(backend));
            stage->enabled = false;
            stage->items.clear();
            return;
        }
        ring.n_slots = n + 1;
        ring.slot_size = need[b];
        ring.short_of = ring.n_slots < n_want ? n_want : 0;

        ggml_backend_buffer_type_t host_buft = stage->use_bounce ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (host_buft != nullptr) {
            ring.bounce = ggml_backend_buft_alloc_buffer(host_buft, stage->n_bounce*stage->chunk_size);
            for (int c = 0; ring.bounce != nullptr && c < stage->n_bounce; c++) {
                if (ring.chunk_ev[c] == nullptr) {
                    ring.chunk_ev[c] = ggml_backend_event_new(dev);
                }
                if (ring.chunk_ev[c] == nullptr) {
                    ggml_backend_buffer_free(ring.bounce);
                    ring.bounce = nullptr;
                }
            }
        }
        if (ring.n_slots < n_want) {
            GGML_LOG_WARN("%s: staging ring on %s: %d of %d slots of %.1f MiB, the memory for more is not available\n", __func__,
                    ggml_backend_name(backend), ring.n_slots, n_want, need[b]/1048576.0);
        }
        GGML_LOG_INFO("%s: staging ring on %s: %d %s slots of %.1f MiB, %s\n", __func__, ggml_backend_name(backend),
                ring.n_slots, ring.lent ? "lent" : "own", need[b]/1048576.0, ring.bounce != nullptr ? "pinned bounce buffers" : "direct uploads");
    }

    ggml_backend_sched_stage_pump(sched);
}

// copy a staged input into its split input on the compute stream; false if the input is not staged
static bool ggml_backend_sched_stage_consume(
        ggml_backend_sched_t sched, int split_id, ggml_backend_t split_backend, struct ggml_tensor * input_cpy) {
    ggml_backend_sched_stage * stage = sched->stage;
    if (stage == nullptr) {
        return false;
    }
    size_t k = stage->n_consumed;
    while (k < stage->items.size() && stage->items[k].split_id == split_id && stage->items[k].input_cpy != input_cpy) {
        k++;
    }
    if (k >= stage->items.size() || stage->items[k].split_id != split_id) {
        return false;
    }
    // the inputs of a split are copied in order, so earlier items of this split are consumed already
    GGML_ASSERT(k == stage->n_consumed && "staging: split inputs consumed out of order");
    if (stage->items[k].slot < 0) {
        // the item waits for a slot that an earlier input of this split read, which is free now
        GGML_ASSERT(stage->n_queued == k && "staging: ring stalled");
        ggml_backend_sched_stage_pump(sched);
        GGML_ASSERT(stage->items[k].slot >= 0 && "staging: ring stalled (with GGML_SCHED_STAGE_REDIRECT, a split needs a slot for each staged input)");
    }
    const ggml_backend_sched_stage_item & item = stage->items[k];
    ggml_backend_sched_stage_ring & ring = stage->rings[item.backend_id];
    ggml_backend_sched_stage_wait_issued(stage->workers[item.backend_id].get(), item.seq);
    ggml_backend_event_wait(split_backend, ring.done[item.slot]);
    if (stage->redirect && sched->callback_eval == nullptr) {
        // the split reads the slot in place; ggml_backend_sched_stage_release frees it after the split's work
        stage->holding.push_back({ input_cpy, input_cpy->data, item.backend_id, item.slot });
        input_cpy->data = ring.slots[item.slot];
        if (ring.prepare != nullptr && !ring.prepared[item.slot]) {
            ring.prepare(split_backend, input_cpy);
            ring.prepared[item.slot] = true;
        }
        stage->n_consumed++;
        return true;
    }
    ring.d2d(split_backend, input_cpy->data, ring.slots[item.slot], ggml_nbytes(item.input));
    if (ring.prepare != nullptr) {
        ring.prepare(split_backend, input_cpy);
    }
    ggml_backend_event_record(ring.free[item.slot], split_backend);
    ring.free_recorded[item.slot] = true;
    GGML_ASSERT(ring.readers[item.slot] > 0);
    ring.busy[item.slot] = --ring.readers[item.slot] != 0;
    stage->n_consumed++;
    ggml_backend_sched_stage_pump(sched);
    return true;
}

// after a split's work is queued: free the slots it reads in place and give its inputs their own memory back
static void ggml_backend_sched_stage_release(ggml_backend_sched_t sched, ggml_backend_t split_backend) {
    ggml_backend_sched_stage * stage = sched->stage;
    if (stage == nullptr || stage->holding.empty()) {
        return;
    }
    for (const auto & h : stage->holding) {
        ggml_backend_sched_stage_ring & ring = stage->rings[h.backend_id];
        ggml_backend_event_record(ring.free[h.slot], split_backend);
        ring.free_recorded[h.slot] = true;
        GGML_ASSERT(ring.readers[h.slot] > 0);
        ring.busy[h.slot] = --ring.readers[h.slot] != 0;
        h.input_cpy->data = h.data;
    }
    stage->holding.clear();
    ggml_backend_sched_stage_pump(sched);
}

void ggml_backend_sched_stage_keep(ggml_backend_sched_t sched, bool keep) {
    auto * stage = sched->stage;
    if (!stage || (!keep && !stage->keep)) {
        return;
    }
    ggml_backend_sched_stage_drain(stage);
    ggml_backend_sched_synchronize(sched);
    stage->keep = keep && stage->enabled && stage->redirect && sched->callback_eval == nullptr;
    for (auto & ring : stage->rings) {
        std::fill(std::begin(ring.resident), std::end(ring.resident), nullptr);
        if (!stage->keep && ring.n_slots > stage->n_slots) {
            for (int s = stage->n_slots; s < ring.n_slots; ++s) {
                ggml_backend_buffer_free(ring.buffers[s]);
                ring.buffers[s] = nullptr;
            }
            ring.n_slots = stage->n_slots;
            ring.next %= ring.n_slots;
        }
    }
}

void ggml_backend_sched_stage_prefetch(ggml_backend_sched_t sched, ggml_backend_t backend, struct ggml_tensor * const * weights, int n_weights) {
    ggml_backend_sched_stage * stage = sched->stage;
    if (stage == nullptr || !stage->keep) {
        return;
    }
    // the GPUs of ggml_backend_sched_stage_spread
    const bool spread = stage->spread && stage->keep_spread;
    int gpus[GGML_SCHED_MAX_BACKENDS];
    int n = 0;
    for (int b = 0; b < sched->n_backends - 1; b++) {
        if (stage->rings[b].upload != nullptr && !stage->blocked[b]) {
            gpus[n++] = b;
        }
    }
    const int backend_id = backend != nullptr ? ggml_backend_sched_backend_id(sched, backend) : -1;
    if (spread ? n < 2 : backend_id < 0) {
        return;
    }
    for (int i = 0; i < n_weights; ++i) {
        const ggml_tensor * weight = weights[i];
        int layer = -1;
        if (weight == nullptr || sscanf(weight->name, "blk.%d.", &layer) != 1 || layer < 0) {
            continue;
        }
        const int b = spread ? gpus[layer % n] : backend_id;
        ggml_backend_sched_stage_ring & ring = stage->rings[b];
        if (ring.n_slots == 0 || !ggml_backend_sched_stage_whole(sched, nullptr, weight, b) ||
                ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(sched->backends[b]), weight) > ring.slot_size) {
            continue;
        }
        bool resident = false;
        for (int s = 0; s < ring.n_slots; ++s) {
            resident |= ring.resident[s] == weight;
        }
        const int slot = resident ? -1 : ggml_backend_sched_stage_evictable(stage, ring, weights, n_weights);
        if (slot >= 0) {
            ggml_backend_sched_stage_push(sched, b, slot, weight);
            ring.readers[slot] = 0;
        }
    }
}

void ggml_backend_sched_set_stage_lender(ggml_backend_sched_t sched, ggml_backend_sched_stage_lend_t lend, void * user_data) {
    if (sched->stage != nullptr) {
        sched->stage->lend = lend;
        sched->stage->lend_user = user_data;
    }
}

bool ggml_backend_sched_stage_reclaim(ggml_backend_sched_t sched) {
    ggml_backend_sched_stage * stage = sched->stage;
    if (stage == nullptr) {
        return false;
    }
    // queued jobs still upload into the slots; their done events are recorded only once they are issued
    ggml_backend_sched_stage_drain(stage);
    bool any = false;
    for (auto & ring : stage->rings) {
        if (ring.lent) {
            ggml_backend_sched_stage_drop_slots(ring);
            any = true;
        }
    }
    return any;
}

size_t ggml_backend_sched_stage_release(ggml_backend_sched_t sched) {
    ggml_backend_sched_stage * stage = sched->stage;
    if (stage == nullptr) {
        return 0;
    }
    ggml_backend_sched_stage_drain(stage);
    size_t freed = 0;
    for (auto & ring : stage->rings) {
        for (int s = 0; s < GGML_SCHED_STAGE_MAX_SLOTS; s++) {
            freed += ring.buffers[s] != nullptr ? ggml_backend_buffer_get_size(ring.buffers[s]) : 0;
        }
        ggml_backend_sched_stage_release_ring(ring);
    }
    return freed;
}

int ggml_backend_sched_stage_get_min_tokens(ggml_backend_sched_t sched) {
    return sched->stage != nullptr && sched->stage->enabled ? sched->stage->min_tokens : -1;
}

void ggml_backend_sched_stage_set_min_tokens(ggml_backend_sched_t sched, int min_tokens) {
    if (sched->stage != nullptr) {
        sched->stage->min_tokens = std::max(0, min_tokens);
    }
}

void ggml_backend_sched_stage_stats(ggml_backend_sched_t sched, int64_t * cpu_us, uint64_t * n_jobs) {
    *cpu_us = 0;
    *n_jobs = 0;
    if (sched->stage == nullptr) {
        return;
    }
    *cpu_us = sched->stage->cpu_us;
    for (const auto & w : sched->stage->workers) {
        if (w != nullptr) {
            std::lock_guard<std::mutex> lock(w->mutex);
            *n_jobs += w->n_pushed;
        }
    }
}

void ggml_backend_sched_stage_allow(ggml_backend_sched_t sched, ggml_backend_t backend, bool allow) {
    if (sched->stage == nullptr) {
        return;
    }
    const int b = ggml_backend_sched_backend_id(sched, backend);
    if (b >= 0) {
        sched->stage->blocked[b] = !allow;
    }
}

// delphi: backend workers (see ggml_backend_sched_worker)

static void ggml_backend_sched_worker_ready(ggml_backend_sched_t sched, int split_id) {
    const auto & split = sched->splits[split_id];
    const int b = split.backend_id;
    if (sched->worker_ready[b] && sched->worker_ready[b](&split.graph, true, sched->worker_ready_data[b])) {
        const uint64_t start = ggml_backend_sched_trace_span_begin();
        sched->worker_ready[b](&split.graph, false, sched->worker_ready_data[b]);
        ggml_backend_sched_trace_span_end(sched, start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                GGML_EXEC_TRACE_WAIT_WORKER_READY, split_id, b, GGML_EXEC_TRACE_ID_NONE, 0);
    }
}

// The GPU work before a CPU split is queued: the CPU backend can read the experts it will likely use while it waits for
// the inputs of the split (GGML_CPU_EXPERT_PREFETCH)
static void ggml_backend_sched_cpu_prefetch(ggml_backend_t backend, struct ggml_cgraph * graph) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }
    typedef void (*prefetch_fn_t)(ggml_backend_t, struct ggml_cgraph *);
    static prefetch_fn_t prefetch_fn = (prefetch_fn_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cpu_prefetch_experts");
    if (prefetch_fn != nullptr) {
        prefetch_fn(backend, graph);
    }
}

// A CPU split waits for the GPU split before it. The experts of the next CPU split are read in that time: the CPU split
// of the experts itself follows a small CPU split (the routing), and when it starts, the GPU work before it is done.
static void ggml_backend_sched_cpu_prefetch_next(ggml_backend_sched_t sched, int split_id) {
    ggml_backend_t backend = sched->backends[sched->splits[split_id].backend_id];
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU || ggml_backend_is_meta(backend)) {
        return;
    }
    for (int j = split_id + 1; j < sched->n_splits; j++) {
        if (sched->backends[sched->splits[j].backend_id] == backend) {
            ggml_backend_sched_cpu_prefetch(backend, &sched->splits[j].graph);
            return;
        }
    }
}

static void ggml_backend_sched_worker_run(ggml_backend_sched_worker * w) {
    ggml_backend_sched_t sched = w->sched;
    ggml_exec_trace_set_thread_context(w->trace_ctx);
    ggml_backend_t backend = sched->backends[w->backend_id];
    struct ggml_backend_sched_split * split = &sched->splits[w->split_id];

    ggml_backend_sched_cpu_prefetch_next(sched, w->split_id);

    for (int b : w->pull_wait) {
        ggml_backend_event_wait(backend, w->pull_events[b]);
    }
    for (const auto & c : w->pulls) {
        if (!w->copy_on(backend, c.src, c.dst)) {
            GGML_ABORT("backend worker: cannot copy %s to %s", c.src->name, ggml_backend_name(backend));
        }
    }
    ggml_backend_event_record(w->pulled_event, backend);

    ggml_backend_sched_worker_ready(sched, w->split_id);

    const uint64_t compute_start = ggml_backend_sched_trace_span_begin();
    if (ggml_exec_trace_observing()) {
        ggml_exec_trace_backend_begin(
                sched->trace_graph_id, w->split_id + 1, w->backend_id + 1,
                ggml_backend_sched_trace_device_id(sched, w->backend_id),
                w->trace_ids.data(), split->graph.n_nodes);
    }
    w->status = ggml_backend_graph_compute_async(backend, &split->graph);
    ggml_exec_trace_backend_end();
    ggml_backend_sched_trace_span_end(
            sched, compute_start, GGML_EXEC_TRACE_EVENT_BACKEND_SUBMIT, 0, w->split_id, w->backend_id,
            GGML_EXEC_TRACE_ID_NONE, (uint32_t) w->status, split->i_start, split->i_end);
    if (sched->events[w->backend_id][w->cur_copy] != NULL) {
        ggml_backend_event_record(sched->events[w->backend_id][w->cur_copy], backend);
    }

    for (int b : w->push_wait) {
        ggml_backend_event_wait(backend, w->push_events[b]);
    }
    for (const auto & c : w->pushes) {
        if (!w->copy_on(backend, c.src, c.dst)) {
            GGML_ABORT("backend worker: cannot copy %s from %s", c.src->name, ggml_backend_name(backend));
        }
    }
    ggml_backend_event_record(w->done_event, backend);
}

static void ggml_backend_sched_worker_main(ggml_backend_sched_worker * w) {
#ifdef __linux__
    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (ggml_cpuset_allowed(w->cpu)) {
            CPU_SET(w->cpu, &set);
        } else {
            // a CPU outside the allowed set: any allowed CPU
            for (int c = 0; c < CPU_SETSIZE; c++) {
                if (ggml_cpuset_allowed(c)) {
                    CPU_SET(c, &set);
                }
            }
        }
        if (CPU_COUNT(&set) > 0 && pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
            GGML_LOG_WARN("%s: cannot pin the worker of %s to CPU %d\n", __func__,
                    ggml_backend_name(w->sched->backends[w->backend_id]), w->cpu);
        }
    }
#endif
    // between jobs, spin for a while (decode posts one job per layer), then sleep
    static const int64_t spin_us = [] {
        const char * env = getenv("GGML_SCHED_WORKER_SPIN_US");
        return env != nullptr ? (int64_t) atoll(env) : (int64_t) 2000;
    }();
    uint64_t seen = 0;
    for (;;) {
        int64_t t0 = -1;
        for (int i = 0; w->posted.load(std::memory_order_acquire) == seen; ++i) {
            if ((i & 255) != 255) {
                continue;
            }
            const int64_t t = ggml_time_us();
            if (t0 < 0) {
                t0 = t;
            } else if (t - t0 > spin_us) {
                std::unique_lock<std::mutex> lock(w->mutex);
                w->cv.wait(lock, [&] { return w->quit || w->posted.load(std::memory_order_acquire) != seen; });
                if (w->quit) {
                    return;
                }
            }
        }
        seen++;
        ggml_backend_sched_worker_run(w);
        w->done.store(seen, std::memory_order_release);
    }
}

// wait until the worker has queued all posted jobs (not until the device has run them)
static void ggml_backend_sched_worker_join(ggml_backend_sched_worker * w) {
    if (w == nullptr) {
        return;
    }
    const uint64_t target = w->posted.load(std::memory_order_relaxed);
    while (w->done.load(std::memory_order_acquire) != target) {
    }
}

static void ggml_backend_sched_worker_stop(ggml_backend_sched_worker * w) {
    if (w == nullptr) {
        return;
    }
    ggml_backend_sched_worker_join(w);
    {
        std::lock_guard<std::mutex> lock(w->mutex);
        w->quit = true;
    }
    w->cv.notify_one();
    if (w->thread.joinable()) {
        w->thread.join();
    }
    for (int b = 0; b < GGML_SCHED_MAX_BACKENDS; b++) {
        if (w->pull_events[b] != NULL) {
            ggml_backend_event_free(w->pull_events[b]);
        }
        if (w->push_events[b] != NULL) {
            ggml_backend_event_free(w->push_events[b]);
        }
    }
    ggml_backend_event_free(w->done_event);
    ggml_backend_event_free(w->pulled_event);
    for (int b = 0; b < GGML_SCHED_MAX_BACKENDS; b++) {
        if (w->landing[b] != NULL) {
            // no backend synchronize: at teardown the backends may be gone already (freeing device memory waits anyway)
            ggml_backend_buffer_free(w->landing[b]);
        }
    }
    delete w;
}

// before the main thread copies inputs to backend b: the pulls of the jobs that read from b are done. Input copies may write
// from the host at once, into memory that held a pulled source; waiting for the pulls only (they finish soon after the post)
// keeps a host-synchronous copy from also waiting for the whole job
static void ggml_backend_sched_workers_fence_pulls(ggml_backend_sched_t sched, int b) {
    for (int wb = 0; wb < sched->n_backends; wb++) {
        ggml_backend_sched_worker * w = sched->workers[wb];
        if (w != nullptr && w->device_wait[b]) {
            ggml_backend_sched_worker_join(w);
            ggml_backend_event_synchronize(w->pulled_event);
        }
    }
}

// before the main thread queues compute on backend b: b waits for the jobs that read from or wrote to it
static void ggml_backend_sched_workers_fence(ggml_backend_sched_t sched, int b) {
    for (int wb = 0; wb < sched->n_backends; wb++) {
        ggml_backend_sched_worker * w = sched->workers[wb];
        if (w != nullptr && w->device_wait[b]) {
            ggml_backend_sched_worker_join(w);
            ggml_backend_event_wait(sched->backends[b], w->done_event);
            w->device_wait[b] = false;
        }
    }
}

static bool ggml_backend_sched_worker_device_ok(ggml_backend_sched_t sched, ggml_backend_sched_worker * w, int b) {
    if (b < 0 || b == w->backend_id) {
        return false;
    }
    ggml_backend_t backend = sched->backends[b];
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    return dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
           ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(ggml_backend_get_device(sched->backends[w->backend_id])) &&
           backend->iface.event_wait != NULL && backend->iface.event_record != NULL;
}

// a split the worker can run: device inputs only (no user inputs, host memory or weights)
static bool ggml_backend_sched_worker_split_ok(ggml_backend_sched_t sched, ggml_backend_sched_worker * w,
        const struct ggml_backend_sched_split * split) {
    if (sched->worker_ready[w->backend_id]) {
        // The caller still submits ordinary local work. A ready split uses only local storage, so no pull or push is needed.
        // Select it at submission, after an earlier CPU split can publish the actual routing decision.
        return split->n_inputs == 0;
    }
    for (int i = 0; i < split->n_inputs; i++) {
        struct ggml_tensor * input = split->inputs[i];
        ggml_backend_buffer_t buf = input->view_src ? input->view_src->buffer : input->buffer;
        if ((input->flags & GGML_TENSOR_FLAG_INPUT) || buf == NULL || ggml_backend_buffer_is_host(buf) ||
                ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return false;
        }
        const int b = tensor_backend_id(input);
        if (!ggml_backend_sched_worker_device_ok(sched, w, b) || sched->workers[b] != nullptr) {
            return false;
        }
    }
    return true;
}

// post split_id to its worker: the pulls of its inputs, and the pushes of its outputs that later splits on other GPUs read
// before the next split on this backend
static void ggml_backend_sched_worker_post(ggml_backend_sched_t sched, ggml_backend_sched_worker * w, int split_id) {
    ggml_backend_sched_worker_join(w);
    const struct ggml_backend_sched_split * split = &sched->splits[split_id];
    w->sched = sched;
    w->split_id = split_id;
    w->cur_copy = sched->cur_copy;
    w->pull_wait.clear();
    w->push_wait.clear();
    w->pulls.clear();
    w->pushes.clear();
    w->forwarded.clear();
    auto add = [](std::vector<int> & v, int b) {
        if (std::find(v.begin(), v.end(), b) == v.end()) {
            v.push_back(b);
        }
    };
    auto event = [&](ggml_backend_event_t * events, int b) {
        if (events[b] == NULL) {
            events[b] = ggml_backend_event_new(ggml_backend_get_device(sched->backends[b]));
        }
        return events[b];
    };
    for (int i = 0; i < split->n_inputs; i++) {
        struct ggml_tensor * input = split->inputs[i];
        add(w->pull_wait, tensor_backend_id(input));
        w->pulls.push_back({ input, sched_hv_copy(sched, input, w->backend_id, sched->cur_copy) });
    }
    size_t landing_size[GGML_SCHED_MAX_BACKENDS] = { 0 };
    for (int s = split_id + 1; s < sched->n_splits && sched->splits[s].backend_id != w->backend_id; s++) {
        const struct ggml_backend_sched_split * next = &sched->splits[s];
        const int b = next->backend_id;
        if (!ggml_backend_sched_worker_device_ok(sched, w, b)) {
            continue;
        }
        const size_t align = ggml_backend_buft_get_alignment(sched->bufts[b]);
        for (int i = 0; i < next->n_inputs; i++) {
            struct ggml_tensor * input = next->inputs[i];
            if (tensor_backend_id(input) != w->backend_id || (input->flags & GGML_TENSOR_FLAG_INPUT)) {
                continue;
            }
            ggml_backend_sched_worker::forward f = { s, i, *input };
            f.landing.view_src = NULL;
            f.landing.view_offs = 0;
            f.landing.data = (void *) landing_size[b]; // offset, made a pointer below
            f.landing.buffer = NULL;
            landing_size[b] += GGML_PAD(ggml_nbytes(input), align);
            w->forwarded.push_back(f);
            add(w->push_wait, b);
        }
    }
    for (int b : w->push_wait) {
        if (w->landing[b] == NULL || ggml_backend_buffer_get_size(w->landing[b]) < landing_size[b]) {
            if (w->landing[b] != NULL) {
                ggml_backend_synchronize(sched->backends[b]);
                ggml_backend_buffer_free(w->landing[b]);
            }
            w->landing[b] = ggml_backend_buft_alloc_buffer(sched->bufts[b], std::max<size_t>(landing_size[b], 1 << 20));
            GGML_ASSERT(w->landing[b] != NULL);
        }
    }
    for (auto & f : w->forwarded) {
        const int b = sched->splits[f.split_id].backend_id;
        f.landing.buffer = w->landing[b];
        f.landing.data = (char *) ggml_backend_buffer_get_base(w->landing[b]) + (size_t) f.landing.data;
        w->pushes.push_back({ sched->splits[f.split_id].inputs[f.input_id], &f.landing });
    }
    // a pull source not recorded after the split that produced it: record now. A push target: its last landing copy and
    // everything else queued on it before this split
    for (int b : w->pull_wait) {
        if (!w->recorded[b]) {
            ggml_backend_event_record(event(w->pull_events, b), sched->backends[b]);
        }
        w->device_wait[b] = true;
    }
    for (int b : w->push_wait) {
        ggml_backend_event_record(event(w->push_events, b), sched->backends[b]);
        w->device_wait[b] = true;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        w->recorded[b] = false;
    }
    if (ggml_exec_trace_observing()) {
        w->trace_ids.resize(split->graph.n_nodes);
        for (int j = 0; j < split->graph.n_nodes; ++j) {
            w->trace_ids[j] = ggml_exec_trace_enabled() ?
                    ggml_backend_sched_trace_tensor_id(sched, split->graph.nodes[j]) : GGML_EXEC_TRACE_ID_NONE;
        }
    }
    w->trace_ctx = ggml_exec_trace_get_thread_context();
    w->posted.fetch_add(1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(w->mutex);
    }
    w->cv.notify_one();
}

// per graph: which splits the worker runs, and after which split each of their pull sources is recorded: the last split on
// the source backend that computes one of the pulled tensors (none found: at the post)
static void ggml_backend_sched_worker_plan(ggml_backend_sched_t sched, ggml_backend_sched_worker * w) {
    w->ok.assign(sched->n_splits, 0);
    w->records.clear();
    for (int s = 0; s < sched->n_splits; s++) {
        const struct ggml_backend_sched_split * split = &sched->splits[s];
        if (split->backend_id != w->backend_id || !ggml_backend_sched_worker_split_ok(sched, w, split)) {
            continue;
        }
        w->ok[s] = 1;
        int last[GGML_SCHED_MAX_BACKENDS];
        std::fill(last, last + GGML_SCHED_MAX_BACKENDS, -2); // -2: no input from b, -1: at the post
        for (int i = 0; i < split->n_inputs; i++) {
            const struct ggml_tensor * input = split->inputs[i];
            const int b = tensor_backend_id(split->inputs[i]);
            int producer = -1;
            for (int p = s - 1; p >= 0 && producer < 0 && sched->splits[p].backend_id != w->backend_id; p--) {
                const struct ggml_backend_sched_split * prev = &sched->splits[p];
                for (int j = 0; prev->backend_id == b && j < prev->graph.n_nodes; j++) {
                    if (prev->graph.nodes[j] == input || prev->graph.nodes[j] == input->view_src) {
                        producer = p;
                        break;
                    }
                }
            }
            last[b] = last[b] == -1 || producer < 0 ? -1 : std::max(last[b], producer);
        }
        for (int b = 0; b < sched->n_backends; b++) {
            if (last[b] >= 0) {
                w->records.push_back({ last[b], b });
            }
        }
    }
}

static const ggml_backend_sched_worker::forward * ggml_backend_sched_worker_forwarded(
        const ggml_backend_sched_worker * w, int split_id, int input_id) {
    for (const auto & f : w->forwarded) {
        if (f.split_id == split_id && f.input_id == input_id) {
            return &f;
        }
    }
    return nullptr;
}

void ggml_backend_sched_set_worker(ggml_backend_sched_t sched, ggml_backend_t backend, bool enable, int cpu) {
    GGML_ASSERT(sched);
    const int b = ggml_backend_sched_backend_id(sched, backend);
    if (b < 0) {
        return;
    }
    ggml_backend_sched_worker * w = sched->workers[b];
    if (w != nullptr && enable && w->cpu == cpu) {
        return;
    }
    if (w != nullptr) {
        ggml_backend_sched_worker_stop(w);
        sched->workers[b] = nullptr;
    }
    if (!enable) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto copy_on = reg ? (ggml_backend_sched_worker::copy_on_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_on") : nullptr;
    if (copy_on == nullptr || backend->iface.event_record == NULL || backend->iface.event_wait == NULL) {
        GGML_LOG_WARN("%s: %s cannot have a worker\n", __func__, ggml_backend_name(backend));
        return;
    }
    w = new ggml_backend_sched_worker();
    w->backend_id = b;
    w->cpu = cpu;
    w->copy_on = copy_on;
    w->sched = sched;
    w->done_event = ggml_backend_event_new(dev);
    w->pulled_event = ggml_backend_event_new(dev);
    w->thread = std::thread(ggml_backend_sched_worker_main, w);
    sched->workers[b] = w;
    GGML_LOG_INFO("%s: splits on %s are submitted by a worker thread%s\n", __func__, ggml_backend_name(backend),
            cpu >= 0 ? (" on CPU " + std::to_string(cpu)).c_str() : "");
}

void ggml_backend_sched_set_worker_ready(ggml_backend_sched_t sched, ggml_backend_t backend,
        ggml_backend_sched_worker_ready_t callback, void * user_data) {
    const int b = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(b >= 0);
    ggml_backend_sched_worker_join(sched->workers[b]);
    sched->worker_ready[b] = callback;
    sched->worker_ready_data[b] = user_data;
}

// delphi: before the host writes memory that a queued copy to a GPU may read, wait for those copies
static void ggml_backend_sched_host_read_fence(ggml_backend_sched_t sched) {
    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->host_read_pending[b]) {
            ggml_backend_event_synchronize(sched->host_read_events[b]);
            sched->host_read_pending[b] = false;
        }
    }
}

// delphi: a split input that is copied between host memory and a GPU on the GPU stream, instead of with both backends
// synchronized around a synchronous copy (GGML_SCHED_HOST_ASYNC=0 disables). Not for weights.
// - GPU to host (a CPU split reads a GPU output, HOST_ASYNC_D2H): the copy is queued after the producer, and the host
//   synchronizes the GPU once, after all inputs of the split. Before: once before the copy and once in it.
// - host to GPU (a GPU split reads a CPU output, HOST_ASYNC_H2D): the stream orders the copy after the earlier reads of the
//   destination, so the host does not wait for the GPU. The caller records host_read_events after the inputs of the split.
//   Not with pipeline parallelism (n_copies > 1): the upload is queued behind the previous graph on that GPU, and the fences
//   that protect the host memory it reads would then wait for that graph, so the host could not run ahead.
enum ggml_backend_sched_host_async { HOST_ASYNC_NONE, HOST_ASYNC_D2H, HOST_ASYNC_H2D };

static ggml_backend_sched_host_async ggml_backend_sched_host_async_kind(ggml_backend_sched_t sched, ggml_backend_t input_backend,
        ggml_backend_t split_backend, int split_backend_id, const struct ggml_tensor * input, const struct ggml_tensor * input_cpy) {
    ggml_backend_dev_t input_dev = ggml_backend_get_device(input_backend);
    ggml_backend_dev_t split_dev = ggml_backend_get_device(split_backend);
    if (ggml_backend_is_meta(input_backend) || ggml_backend_is_meta(split_backend) || input_dev == nullptr || split_dev == nullptr ||
            input->buffer == NULL || input_cpy->buffer == NULL || ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return HOST_ASYNC_NONE;
    }
    const bool input_host = ggml_backend_buffer_is_host(input->buffer);
    const bool cpy_host = ggml_backend_buffer_is_host(input_cpy->buffer);
    const int input_backend_id = ggml_backend_sched_backend_id(sched, input_backend);
    if (ggml_backend_dev_type(split_dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cpy_host && !input_host &&
            ggml_backend_dev_type(input_dev) != GGML_BACKEND_DEVICE_TYPE_CPU && input_backend_id >= 0 &&
            sched->host_read_events[input_backend_id] != NULL && input_backend->iface.get_tensor_async != NULL) {
        return HOST_ASYNC_D2H;
    }
    if (sched->n_copies == 1 && ggml_backend_dev_type(split_dev) != GGML_BACKEND_DEVICE_TYPE_CPU && input_host && !cpy_host &&
            ggml_backend_dev_type(input_dev) == GGML_BACKEND_DEVICE_TYPE_CPU &&
            sched->host_read_events[split_backend_id] != NULL && split_backend->iface.set_tensor_async != NULL) {
        return HOST_ASYNC_H2D;
    }
    return HOST_ASYNC_NONE;
}

static bool ggml_backend_sched_input_can_stage(ggml_backend_t backend, const ggml_tensor * input, const ggml_tensor * copy) {
    return (input->flags & GGML_TENSOR_FLAG_INPUT) && input->buffer && copy->buffer &&
            ggml_backend_buffer_is_host(input->buffer) && !ggml_backend_buffer_is_host(copy->buffer) &&
            backend->iface.set_tensor_async && backend->iface.event_record;
}

// GGML_SCHED_INPUT_STAGING_MIN_KIB: initial capacity of each staging slot (default 4096)
static size_t ggml_backend_sched_input_staging_min_bytes() {
    static const size_t bytes = [] {
        const char * env = getenv("GGML_SCHED_INPUT_STAGING_MIN_KIB");
        return (size_t) (env ? std::max(0, atoi(env)) : 4096) * 1024;
    }();
    return bytes;
}

static int ggml_backend_sched_input_stage_begin(ggml_backend_sched_t sched) {
    auto * stage = sched->input_staging;
    if (!stage) {
        return 0;
    }
    // This ring follows submissions, including repeated graphs, rather than the allocation copy index.
    const int c = stage->next;
    stage->next = (c + 1) % sched->n_copies;
    size_t sizes[GGML_SCHED_MAX_BACKENDS] = {};
    for (int i = 0; i < sched->n_splits; ++i) {
        const auto & split = sched->splits[i];
        const int b = split.backend_id;
        for (int j = 0; j < split.n_inputs; ++j) {
            auto * input = split.inputs[j];
            auto * copy = sched_hv_copy(sched, input, b, sched->cur_copy);
            if (ggml_backend_sched_input_can_stage(sched->backends[b], input, copy)) {
                sizes[b] += ggml_nbytes(input);
            }
        }
    }
    for (int b = 0; b < sched->n_backends; ++b) {
        auto & s = stage->slots[b][c];
        s.offset = 0;
        if (!sizes[b]) {
            continue;
        }
        auto * dev = ggml_backend_get_device(sched->backends[b]);
        auto * buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (!buft) {
            continue;
        }
        if (s.pending) {
            // Only the uploads use this host buffer; later computation need not complete before reuse.
            ggml_backend_event_synchronize(s.done);
            s.pending = false;
        }
        if (!s.buffer || ggml_backend_buffer_get_size(s.buffer) < sizes[b]) {
            // Freeing pinned memory waits for all work of the device and breaks the prompt pipeline. Prompt inputs grow
            // with the KV view (about 0.4 MiB at 60K for GLM-5.3), so start with room for long prompts.
            const size_t capacity = std::max({ sizes[b], ggml_backend_sched_input_staging_min_bytes(),
                    s.buffer ? 2*ggml_backend_buffer_get_size(s.buffer) : 0 });
            ggml_backend_buffer_free(s.buffer);
            s.buffer = ggml_backend_buft_alloc_buffer(buft, capacity);
        }
        if (s.buffer && !s.done) {
            s.done = ggml_backend_event_new(dev);
        }
    }
    return c;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    bool backends_idle = sched->backends_idle;
    sched->backends_idle = false;
    struct ggml_backend_sched_split * splits = sched->splits;
    const int input_stage_copy = ggml_backend_sched_input_stage_begin(sched);

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    int prev_backend_id = -1;

    // The input copies of this graph use copy slot cur_copy, which the last graph with the same slot may still read.
    // Before its first event record in this graph, events[b][cur_copy] marks the end of that graph on backend b;
    // afterwards it marks a split of this graph, which comes later in the stream of b. A host synchronization on it at
    // any point of this graph therefore proves that the previous use of the slot on b has ended, and no later input
    // of this graph needs to wait for it again. Waiting again would wait for the earlier splits of this graph on b,
    // which with pipeline parallelism sit behind the previous ubatch, and would serialize the ubatches.
    bool copy_slot_free[GGML_SCHED_MAX_BACKENDS] = { false };

    // Take that proof before the first event record of this graph. A backend whose first host-side input belongs to a
    // later split (CUDA1 running a split before the one that reads a user input, for example) would otherwise wait for
    // its own earlier split of this graph, i.e. for the previous device's part of this ubatch, with the host blocked.
    // Here the event still marks the slot's previous graph, which with n_copies > 1 has normally ended long ago.
    if (!sched->copy_reuse_wait_all && sched->copy_slot_prewait) {
        for (int split_id = 0; split_id < sched->n_splits; split_id++) {
            const struct ggml_backend_sched_split * split = &splits[split_id];
            const int b = split->backend_id;
            ggml_backend_event_t ev = sched->events[b][sched->cur_copy];
            if (copy_slot_free[b] || ev == NULL) {
                continue;
            }
            for (int input_id = 0; input_id < split->n_inputs; input_id++) {
                const struct ggml_tensor * input = split->inputs[input_id];
                if ((input->flags & GGML_TENSOR_FLAG_INPUT) || (input->buffer && ggml_backend_buffer_is_host(input->buffer))) {
                    const uint64_t wait_start = ggml_backend_sched_trace_detail_span_begin();
                    ggml_backend_event_synchronize(ev);
                    ggml_backend_sched_trace_span_end(
                            sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                            GGML_EXEC_TRACE_WAIT_COPY_DST_REUSE, split_id, b, GGML_EXEC_TRACE_ID_NONE,
                            GGML_EXEC_TRACE_WAIT_API_EVENT_SYNCHRONIZE);
                    copy_slot_free[b] = true;
                    break;
                }
            }
        }
    }

    if (sched->stage != nullptr) {
        ggml_backend_sched_stage_begin(sched);
    }

    // delphi: backend workers take graphs that stage nothing (decode), without an eval callback
    bool workers_on = false;
    for (int b = 0; b < sched->n_backends; b++) {
        workers_on |= sched->workers[b] != nullptr;
    }
    workers_on = workers_on && !sched->callback_eval && (sched->stage == nullptr || sched->stage->items.empty());
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_sched_worker * w = sched->workers[b];
        if (w != nullptr) {
            w->forwarded.clear(); // they name splits of the previous graph
            std::fill(w->recorded, w->recorded + GGML_SCHED_MAX_BACKENDS, false);
            if (workers_on) {
                ggml_backend_sched_worker_plan(sched, w);
            }
        }
    }

    struct landed_input { ggml_backend_sched_worker * w; const ggml_backend_sched_worker::forward * f; struct ggml_tensor * input; };
    std::vector<landed_input> landed;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // before any wait for the previous splits (a split of a backend worker prefetches on the worker)
        if (!workers_on || sched->workers[split_backend_id] == nullptr) {
            ggml_backend_sched_cpu_prefetch_next(sched, split_id);
        }

        if (workers_on) {
            ggml_backend_sched_worker * w = sched->workers[split_backend_id];
            const auto ready = sched->worker_ready[split_backend_id];
            if (w != nullptr && w->ok[split_id] &&
                    (!ready || ready(&split->graph, true, sched->worker_ready_data[split_backend_id]))) {
                ggml_backend_sched_worker_post(sched, w, split_id);
                prev_backend_id = split_backend_id;
                continue;
            }
            // this thread queues work on the split backend below
            const uint64_t join_start = w != nullptr ? ggml_backend_sched_trace_span_begin() : 0;
            ggml_backend_sched_worker_join(w);
            ggml_backend_sched_trace_span_end(sched, join_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                    GGML_EXEC_TRACE_WAIT_PREVIOUS_SPLIT, split_id, split_backend_id, GGML_EXEC_TRACE_ID_NONE, 0);
            ggml_backend_sched_workers_fence_pulls(sched, split_backend_id);
        }

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id &&
                ggml_backend_sched_splits_share_memory(sched, prev_backend_id, split_backend_id)) {
            ggml_backend_sched_worker_join(sched->workers[prev_backend_id]);
            const uint64_t wait_start = ggml_backend_sched_trace_span_begin();
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
                ggml_backend_sched_trace_span_end(
                        sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                        GGML_EXEC_TRACE_WAIT_PREVIOUS_SPLIT, split_id, prev_backend_id, GGML_EXEC_TRACE_ID_NONE,
                        GGML_EXEC_TRACE_WAIT_API_EVENT_SYNCHRONIZE);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
                ggml_backend_sched_trace_span_end(
                        sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                        GGML_EXEC_TRACE_WAIT_PREVIOUS_SPLIT, split_id, prev_backend_id, GGML_EXEC_TRACE_ID_NONE,
                        GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);
            }
        }

        // copy the input tensors to the split backend
        const bool split_is_meta = ggml_backend_is_meta(split_backend);
        bool input_sync_pending = false;
        bool d2h_sync[GGML_SCHED_MAX_BACKENDS] = { false };
        bool h2d[GGML_SCHED_MAX_BACKENDS] = { false };
        if (ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            ggml_backend_sched_host_read_fence(sched);
        }

        // the point after which the split backend no longer reads the previous contents of this split's inputs: it is
        // recorded before the staged inputs are queued, so a device-to-device copy does not also wait for their uploads
        ggml_backend_event_t xdev_ev = NULL;
        for (int input_id = 0; input_id < split->n_inputs && sched->xdev_events[split_backend_id] != NULL && xdev_ev == NULL; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            ggml_backend_dev_t input_dev = input_backend ? ggml_backend_get_device(input_backend) : nullptr;
            if (input_backend != split_backend && input_dev != nullptr && ggml_backend_dev_type(input_dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                    !(split->inputs[input_id]->flags & GGML_TENSOR_FLAG_INPUT)) {
                xdev_ev = sched->xdev_events[split_backend_id];
                ggml_backend_event_record(xdev_ev, split_backend);
            }
        }
        // the reuse wait below is needed once per split: after it, the split backend has only the copies and staged
        // uploads of this split's inputs queued, which do not read their previous contents. Waiting on the backend again
        // for each input also waited for those: on two GPUs, about 25 ms per MiMo layer between the activation copies
        ggml_backend_event_t reuse_ev = sched->reuse_events[split_backend_id];
        bool reuse_waited = false;
        if (reuse_ev != NULL && split->n_inputs > 0) {
            ggml_backend_event_record(reuse_ev, split_backend);
        }
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            if (workers_on) {
                ggml_backend_sched_worker * w = sched->workers[tensor_backend_id(input)];
                const ggml_backend_sched_worker::forward * f = w != nullptr ? ggml_backend_sched_worker_forwarded(w, split_id, input_id) : nullptr;
                if (f != nullptr) {
                    landed.push_back({ w, f, input });
                    continue; // copied from the landing buffer after the other inputs
                }
                ggml_backend_sched_worker_join(w);
            }
            struct ggml_tensor * input_cpy = sched_hv_copy(sched, input, split_backend_id, sched->cur_copy);
            const uint64_t copy_start = ggml_backend_sched_trace_span_begin();
            const bool observe_copy = copy_start || ggml_exec_trace_integrity_enabled();
            const int input_backend_id = observe_copy ? ggml_backend_sched_backend_id(sched, input_backend) : -1;
            const uint32_t input_node_id = copy_start ? ggml_backend_sched_trace_tensor_id(sched, input) : GGML_EXEC_TRACE_ID_NONE;
            // Scheduler copies keep the source tensor ID; the backend fields identify the storage endpoints.
            const uint32_t input_cpy_node_id = input_node_id;
            enum ggml_exec_trace_copy_path copy_path = GGML_EXEC_TRACE_COPY_BACKEND_ASYNC;
            uint64_t copy_bytes = observe_copy ? ggml_nbytes(input) : 0;

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                auto * staged = sched->input_staging ? &sched->input_staging->slots[split_backend_id][input_stage_copy] : nullptr;
                const bool input_staged = staged && staged->buffer && staged->done &&
                        ggml_backend_sched_input_can_stage(split_backend, input, input_cpy);
                const bool input_async = sched->input_async && input->buffer != NULL && input_cpy->buffer != NULL &&
                        ggml_backend_buffer_is_host(input->buffer) && !ggml_backend_buffer_is_host(input_cpy->buffer) &&
                        split_backend->iface.set_tensor_async != NULL;
                copy_path = input_async || input_staged ? GGML_EXEC_TRACE_COPY_BACKEND_ASYNC : GGML_EXEC_TRACE_COPY_USER_INPUT_SYNC;
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                const uint64_t wait_start = ggml_backend_sched_trace_detail_span_begin();
                if (copy_slot_free[split_backend_id] && !sched->copy_reuse_wait_all) {
                    // the previous use of the slot has ended (see copy_slot_free)
                } else if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                    copy_slot_free[split_backend_id] = true;
                    ggml_backend_sched_trace_copy_wait_end(
                            sched, wait_start, GGML_EXEC_TRACE_EVENT_COPY_WAIT_DST, copy_path,
                            split_id, split_backend_id, input_backend_id, input_node_id,
                            GGML_EXEC_TRACE_WAIT_API_EVENT_SYNCHRONIZE, copy_bytes);
                } else {
                    ggml_backend_synchronize(split_backend);
                    ggml_backend_sched_trace_copy_wait_end(
                            sched, wait_start, GGML_EXEC_TRACE_EVENT_COPY_WAIT_DST, copy_path,
                            split_id, split_backend_id, input_backend_id, input_node_id,
                            GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE, copy_bytes);
                }
                const uint64_t body_start = ggml_backend_sched_trace_detail_span_begin();
                if (input_staged) {
                    const size_t size = ggml_nbytes(input);
                    GGML_ASSERT(staged->offset + size <= ggml_backend_buffer_get_size(staged->buffer));
                    auto * data = (char *) ggml_backend_buffer_get_base(staged->buffer) + staged->offset;
                    memcpy(data, input->data, size);
                    staged->offset += size;
                    ggml_backend_tensor_set_async(split_backend, input_cpy, data, 0, size);
                    ggml_backend_event_record(staged->done, split_backend);
                    staged->pending = true;
                } else if (input_async) {
                    // The caller holds the source until this graph completes. Avoid a synchronous buffer copy,
                    // which can wait for unrelated streams while background prefill runs on the same device.
                    ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                } else if (split_is_meta && backends_idle && ggml_backend_buffer_is_host(input->buffer)) {
                    // one synchronous copy per input and simple device costs about 0.1 ms per input on a Meta
                    // backend; queue them all and synchronize once after the last input instead. Only while
                    // nothing else is queued: otherwise that synchronization would also wait for the previous
                    // ubatch and serialize prefill
                    ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                    input_sync_pending = true;
                } else {
                    ggml_backend_tensor_copy(input, input_cpy);
                }
                ggml_backend_sched_trace_copy_end(
                        sched, body_start, GGML_EXEC_TRACE_EVENT_COPY_BODY, copy_path,
                        split_id, input_backend_id, split_backend_id,
                        input_node_id, input_cpy_node_id, copy_bytes);
            } else if (ggml_backend_sched_stage_consume(sched, split_id, split_backend, input_cpy)) {
                // uploaded ahead by the staging ring; the device-to-device copy is ordered on the compute stream
            } else {
                GGML_ASSERT((sched->stage == nullptr || input_cpy->buffer != sched->stage->rings[split_backend_id].anchor) &&
                        "staging: a redirected input has no memory, but the ring did not stage it");
                // wait for the split backend to finish using the input before overwriting it. The slot event states
                // exactly that; a backend may queue the copy outside its compute stream after this wait (CUDA without
                // peer access uploads on a copy stream), so a later event would delay the copy for nothing
                ggml_backend_dev_t input_dev = ggml_backend_get_device(input_backend);
                const ggml_backend_sched_host_async host_async =
                        ggml_backend_sched_host_async_kind(sched, input_backend, split_backend, split_backend_id, input, input_cpy);
                if (host_async == HOST_ASYNC_H2D) {
                    // ordered on the split backend's stream after its earlier reads of the input
                } else if (sched->slot_events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->slot_events[split_backend_id][sched->cur_copy]);
                } else if (xdev_ev != NULL && input_backend != split_backend &&
                        input_dev != nullptr && ggml_backend_dev_type(input_dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                        input_backend->iface.event_wait != NULL && split_backend->iface.cpy_tensor_async != NULL) {
                    // the async copy runs on the input backend's stream: make that stream wait for the split backend
                    ggml_backend_event_wait(input_backend, xdev_ev);
                } else if (reuse_ev != NULL) {
                    if (!reuse_waited) {
                        const uint64_t wait_start = ggml_backend_sched_trace_detail_span_begin();
                        ggml_backend_event_synchronize(reuse_ev);
                        reuse_waited = true;
                        ggml_backend_sched_trace_span_end(
                                sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                                GGML_EXEC_TRACE_WAIT_COPY_DST_REUSE, split_id, split_backend_id,
                                input_node_id, GGML_EXEC_TRACE_WAIT_API_EVENT_SYNCHRONIZE);
                    }
                } else {
                    const uint64_t wait_start = ggml_backend_sched_trace_detail_span_begin();
                    ggml_backend_synchronize(split_backend);
                    ggml_backend_sched_trace_span_end(
                            sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                            GGML_EXEC_TRACE_WAIT_COPY_DST_REUSE, split_id, split_backend_id,
                            input_node_id, GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {
                    copy_path = GGML_EXEC_TRACE_COPY_MOE_EXPERT_ASYNC;
                    copy_bytes = 0;

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_tensor * ids_tensor = node->src[2];
                    const double expected_active = ggml_offload_cost_expected_active_experts(
                            n_expert, ids_tensor->ne[1], ids_tensor->ne[0]);
                    if (expected_active >= n_expert - 0.5) {
                        // Reading IDs forces a GPU synchronization. Use one full copy when fewer than
                        // half an expert is expected to be inactive.
                        ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                        copy_bytes = observe_copy ? ggml_nbytes(input) : 0;
                    } else {
                        const uint64_t wait_start = ggml_backend_sched_trace_detail_span_begin();
                        ggml_backend_synchronize(input_backend);
                        ggml_backend_sched_trace_copy_wait_end(
                                sched, wait_start, GGML_EXEC_TRACE_EVENT_COPY_WAIT_SRC, copy_path,
                                split_id, input_backend_id, split_backend_id, input_node_id,
                                GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);

                        // get the ids
                        ggml_backend_t ids_backend = split_backend;

                        if (ggml_nelements(ids_tensor) == 0) {
                            continue;
                        }

                        // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                        // in that case, we use the original ids tensor
                        for (int i = input_id + 1; i < split->n_inputs; i++) {
                            if (ids_tensor == sched_hv_copy(sched, split->inputs[i], split_backend_id, sched->cur_copy)) {
                                ids_tensor = split->inputs[i];
                                ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                                break;
                            }
                        }

                        if (ids_tensor != prev_ids_tensor) {
                            ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                            ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                            const uint64_t ids_wait_start = ggml_backend_sched_trace_detail_span_begin();
                            const int ids_backend_id = ids_wait_start ? ggml_backend_sched_backend_id(sched, ids_backend) : -1;
                            const uint32_t ids_node_id = ids_wait_start ? ggml_backend_sched_trace_tensor_id(sched, ids_tensor) : GGML_EXEC_TRACE_ID_NONE;
                            ggml_backend_synchronize(ids_backend);
                            ggml_backend_sched_trace_span_end(
                                    sched, ids_wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                                    GGML_EXEC_TRACE_WAIT_MOE_IDS, split_id, ids_backend_id,
                                    ids_node_id,
                                    GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);

                            // find the used experts
                            used_ids.clear();
                            used_ids.resize(ggml_bitset_size(n_expert));
                            for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                                for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                    int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                    GGML_ASSERT(id >= 0 && id < n_expert);
                                    ggml_bitset_set(used_ids.data(), id);
                                }
                            }

                            prev_ids_tensor = ids_tensor;
                        }

                        // group consecutive experts and copy them together
                        auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                            const size_t expert_offset = first_id * expert_size;
                            const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                            const size_t padding = std::min<size_t>(expert_size, 512);
                            const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                            ggml_backend_tensor_set_async(split_backend,
                                input_cpy,
                                (const uint8_t *)input->data + expert_offset, expert_offset,
                                // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                                // this is necessary for MMQ in the CUDA backend
                                expert_size_copy + padding_end);
                            if (observe_copy) {
                                copy_bytes += expert_size_copy + padding_end;
                            }
                        };

                        int id = 0;
                        while (!ggml_bitset_get(used_ids.data(), id)) {
                            id++;
                        }
                        int32_t first_id = id;
                        int32_t last_id = first_id;

                        for (++id; id < n_expert; ++id) {
                            if (!ggml_bitset_get(used_ids.data(), id)) {
                                continue;
                            }

                            if (id == last_id + 1) {
                                last_id = id;
                                continue;
                            }

                            copy_experts(first_id, last_id);

                            first_id = id;
                            last_id = id;
                        }
                        copy_experts(first_id, last_id);
                    }
                } else if (host_async == HOST_ASYNC_D2H) {
                    ggml_backend_tensor_get_async(input_backend, input, input_cpy->data, 0, ggml_nbytes(input));
                    d2h_sync[ggml_backend_sched_backend_id(sched, input_backend)] = true;
                } else if (host_async == HOST_ASYNC_H2D) {
                    ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                    h2d[split_backend_id] = true;
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        copy_path = GGML_EXEC_TRACE_COPY_FALLBACK_SYNC;
                        const uint64_t src_wait_start = ggml_backend_sched_trace_detail_span_begin();
                        ggml_backend_synchronize(input_backend);
                        ggml_backend_sched_trace_copy_wait_end(
                                sched, src_wait_start, GGML_EXEC_TRACE_EVENT_COPY_WAIT_SRC, copy_path,
                                split_id, input_backend_id, split_backend_id, input_node_id,
                                GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE, copy_bytes);
                        const uint64_t dst_wait_start = ggml_backend_sched_trace_detail_span_begin();
                        if (copy_slot_free[split_backend_id] && !sched->copy_reuse_wait_all) {
                            // the previous use of the slot has ended (see copy_slot_free)
                        } else if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                            copy_slot_free[split_backend_id] = true;
                            ggml_backend_sched_trace_copy_wait_end(
                                    sched, dst_wait_start, GGML_EXEC_TRACE_EVENT_COPY_WAIT_DST, copy_path,
                                    split_id, split_backend_id, input_backend_id, input_node_id,
                                    GGML_EXEC_TRACE_WAIT_API_EVENT_SYNCHRONIZE, copy_bytes);
                        } else {
                            ggml_backend_synchronize(split_backend);
                            ggml_backend_sched_trace_copy_wait_end(
                                    sched, dst_wait_start, GGML_EXEC_TRACE_EVENT_COPY_WAIT_DST, copy_path,
                                    split_id, split_backend_id, input_backend_id, input_node_id,
                                    GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE, copy_bytes);
                        }
                        const uint64_t body_start = ggml_backend_sched_trace_detail_span_begin();
                        ggml_backend_tensor_copy(input, input_cpy);
                        ggml_backend_sched_trace_copy_end(
                                sched, body_start, GGML_EXEC_TRACE_EVENT_COPY_BODY, copy_path,
                                split_id, input_backend_id, split_backend_id,
                                input_node_id, input_cpy_node_id, copy_bytes);
                    }
                }
            }

            ggml_backend_sched_trace_copy_end(
                    sched, copy_start, GGML_EXEC_TRACE_EVENT_COPY_HANDLING, copy_path,
                    split_id, input_backend_id, split_backend_id,
                    input_node_id, input_cpy_node_id, copy_bytes);
        }

        if (workers_on) {
            // the split backend waits for the jobs it depends on only now, so that the copies above do not wait for them
            ggml_backend_sched_workers_fence(sched, split_backend_id);
            for (const auto & l : landed) {
                if (!l.w->copy_on(split_backend, &l.f->landing, sched_hv_copy(sched, l.input, split_backend_id, sched->cur_copy))) {
                    GGML_ABORT("backend worker: cannot copy %s on %s", l.input->name, ggml_backend_name(split_backend));
                }
            }
            landed.clear();
        }

        if (input_sync_pending) {
            ggml_backend_synchronize(split_backend);
        }
        for (int b = 0; b < sched->n_backends; b++) {
            if (d2h_sync[b]) {
                const uint64_t wait_start = ggml_backend_sched_trace_detail_span_begin();
                ggml_backend_synchronize(sched->backends[b]);
                ggml_backend_sched_trace_span_end(
                        sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                        GGML_EXEC_TRACE_WAIT_PREVIOUS_SPLIT, split_id, b, GGML_EXEC_TRACE_ID_NONE,
                        GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);
            }
            if (h2d[b]) {
                ggml_backend_event_record(sched->host_read_events[b], sched->backends[b]);
                sched->host_read_pending[b] = true;
            }
        }
        if (split_is_meta) {
            backends_idle = false;
        }

        // Also keep readiness when workers are unavailable, staging is active, or an eval callback requires serial execution.
        ggml_backend_sched_worker_ready(sched, split_id);

        if (!sched->callback_eval) {
            const uint64_t compute_start = ggml_backend_sched_trace_span_begin();
            const enum ggml_exec_trace_event_kind compute_kind = compute_start &&
                    ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU ?
                    GGML_EXEC_TRACE_EVENT_CPU_COMPUTE : GGML_EXEC_TRACE_EVENT_BACKEND_SUBMIT;
            if (ggml_exec_trace_observing()) {
                for (int j = 0; j < split->graph.n_nodes; ++j) {
                    sched->trace_backend_node_ids[j] = compute_start ?
                            ggml_backend_sched_trace_tensor_id(sched, split->graph.nodes[j]) : GGML_EXEC_TRACE_ID_NONE;
                }
                ggml_exec_trace_backend_begin(
                        sched->trace_graph_id, split_id + 1, split_backend_id + 1,
                        ggml_backend_sched_trace_device_id(sched, split_backend_id),
                        sched->trace_backend_node_ids, split->graph.n_nodes);
            }
            if (sched->stage != nullptr && !sched->stage->holding.empty()) {
                // redirected inputs read a slot that changes from use to use; a graph id would let a backend replay
                // a captured graph without comparing the data pointers (ggml-cuda), so drop it
                split->graph.uid = 0;
            }
            // a CPU split computes on this thread: its time is the CPU expert time that the staging threshold weighs
            const int64_t t_cpu = sched->stage != nullptr &&
                    ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU ? ggml_time_us() : 0;
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (t_cpu) {
                sched->stage->cpu_us += ggml_time_us() - t_cpu;
            }
            ggml_backend_sched_stage_release(sched, split_backend);
            ggml_exec_trace_backend_end();
            ggml_backend_sched_trace_span_end(
                    sched, compute_start, compute_kind, 0, split_id, split_backend_id,
                    GGML_EXEC_TRACE_ID_NONE, (uint32_t) ec, split->i_start, split->i_end);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            ggml_exec_trace_integrity_eval_callback();
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                const uint64_t compute_start = ggml_backend_sched_trace_span_begin();
                const enum ggml_exec_trace_event_kind compute_kind = compute_start &&
                        ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU ?
                        GGML_EXEC_TRACE_EVENT_CPU_COMPUTE : GGML_EXEC_TRACE_EVENT_BACKEND_SUBMIT;
                if (ggml_exec_trace_observing()) {
                    for (int j = 0; j < gv.n_nodes; ++j) {
                        sched->trace_backend_node_ids[j] = compute_start ?
                                ggml_backend_sched_trace_tensor_id(sched, gv.nodes[j]) : GGML_EXEC_TRACE_ID_NONE;
                    }
                    ggml_exec_trace_backend_begin(
                            sched->trace_graph_id, split_id + 1, split_backend_id + 1,
                            ggml_backend_sched_trace_device_id(sched, split_backend_id),
                            sched->trace_backend_node_ids, gv.n_nodes);
                }
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                ggml_exec_trace_backend_end();
                ggml_backend_sched_trace_span_end(
                        sched, compute_start, compute_kind, 0, split_id, split_backend_id,
                        GGML_EXEC_TRACE_ID_NONE, (uint32_t) ec,
                        split->i_start + j0, split->i_start + j1 + 1);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                const uint64_t wait_start = ggml_backend_sched_trace_span_begin();
                ggml_backend_synchronize(split_backend);
                ggml_backend_sched_trace_span_end(
                        sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                        GGML_EXEC_TRACE_WAIT_EVAL_CALLBACK, split_id, split_backend_id,
                        GGML_EXEC_TRACE_ID_NONE, GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // a pull source of a worker split that follows
        if (workers_on) {
            for (int b = 0; b < sched->n_backends; b++) {
                ggml_backend_sched_worker * w = sched->workers[b];
                for (size_t r = 0; w != nullptr && r < w->records.size(); r++) {
                    if (w->records[r].first != split_id) {
                        continue;
                    }
                    const int src = w->records[r].second;
                    ggml_backend_sched_worker_join(w); // its previous job has waited for the previous record
                    if (w->pull_events[src] == NULL) {
                        w->pull_events[src] = ggml_backend_event_new(ggml_backend_get_device(sched->backends[src]));
                    }
                    ggml_backend_event_record(w->pull_events[src], sched->backends[src]);
                    w->recorded[src] = true;
                }
            }
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    // the workers have queued their jobs; the backends they wrote to wait for them before anything that follows
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_sched_worker * w = sched->workers[b];
        if (w == nullptr) {
            continue;
        }
        ggml_backend_sched_worker_join(w);
        for (int d = 0; d < sched->n_backends; d++) {
            ggml_backend_sched_workers_fence(sched, d);
        }
        if (w->status != GGML_STATUS_SUCCESS) {
            const enum ggml_status status = w->status;
            w->status = GGML_STATUS_SUCCESS;
            return status;
        }
    }

    ggml_backend_sched_host_read_fence(sched);

    // mark the end of this graph's use of copy slot cur_copy on every backend that ran a split of it
    bool slot_recorded[GGML_SCHED_MAX_BACKENDS] = { false };
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        const int b = splits[split_id].backend_id;
        if (!slot_recorded[b] && sched->slot_events[b][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->slot_events[b][sched->cur_copy], sched->backends[b]);
            slot_recorded[b] = true;
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    const char * GGML_SCHED_VIEW_ALIAS = getenv("GGML_SCHED_VIEW_ALIAS");
    sched->view_alias = GGML_SCHED_VIEW_ALIAS ? atoi(GGML_SCHED_VIEW_ALIAS) != 0 : true;
    const char * GGML_SCHED_PREV_SPLIT_WAIT = getenv("GGML_SCHED_PREV_SPLIT_WAIT");
    sched->prev_split_wait_all = GGML_SCHED_PREV_SPLIT_WAIT ? atoi(GGML_SCHED_PREV_SPLIT_WAIT) != 0 : false;
    const char * GGML_SCHED_COPY_REUSE_WAIT = getenv("GGML_SCHED_COPY_REUSE_WAIT");
    sched->copy_reuse_wait_all = GGML_SCHED_COPY_REUSE_WAIT ? atoi(GGML_SCHED_COPY_REUSE_WAIT) != 0 : false;
    const char * GGML_SCHED_COPY_SLOT_PREWAIT = getenv("GGML_SCHED_COPY_SLOT_PREWAIT");
    sched->copy_slot_prewait = GGML_SCHED_COPY_SLOT_PREWAIT ? atoi(GGML_SCHED_COPY_SLOT_PREWAIT) != 0 : true;
    const char * GGML_SCHED_INPUT_ORDER = getenv("GGML_SCHED_INPUT_ORDER");
    sched->input_order = GGML_SCHED_INPUT_ORDER ? atoi(GGML_SCHED_INPUT_ORDER) != 0 : true;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    // note: calloc, not malloc - the scoped reset below only clears slots written since the
    // last reset, so slots that were never written must already hold the cleared default
    // (NULL). this matters most for a fresh scheduler whose first reset clears nothing.
    sched->hv_tensor_copies      = (ggml_tensor **) calloc(sched->hash_set.size * sched->n_backends * sched->n_copies, sizeof(struct ggml_tensor *));
    sched->hv_offload_decisions  = (ggml_backend_sched_offload_decision *) calloc(sched->hash_set.size, sizeof(sched->hv_offload_decisions[0]));
    sched->hv_written            = (ggml_bitset_t *) calloc(ggml_bitset_size(sched->hash_set.size), sizeof(ggml_bitset_t));
    sched->hv_touched            = nullptr;
    sched->hv_touched_count      = 0;
    sched->hv_touched_capacity   = 0;

    {
        const char * GGML_SCHED_SCOPED_RESET = getenv("GGML_SCHED_SCOPED_RESET");
        sched->scoped_reset = GGML_SCHED_SCOPED_RESET ? atoi(GGML_SCHED_SCOPED_RESET) != 0 : true;
    }

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));
    sched->trace_backend_node_ids = (uint32_t *) calloc(nodes_size, sizeof(sched->trace_backend_node_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
                sched->slot_events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    const char * xdev_env = getenv("GGML_SCHED_XDEV_EVENTS");
    if (sched->n_copies == 1 && xdev_env != nullptr && atoi(xdev_env) != 0) {
        for (int b = 0; b < n_backends; b++) {
            ggml_backend_dev_t dev = backends[b]->device;
            if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                sched->xdev_events[b] = ggml_backend_event_new(dev);
            }
        }
    }

    const char * host_async_env = getenv("GGML_SCHED_HOST_ASYNC");
    if (host_async_env == nullptr || atoi(host_async_env) != 0) {
        for (int b = 0; b < n_backends; b++) {
            ggml_backend_dev_t dev = backends[b]->device;
            if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                    backends[b]->iface.event_record != NULL) {
                sched->host_read_events[b] = ggml_backend_event_new(dev);
            }
        }
    }

    const char * reuse_env = getenv("GGML_SCHED_REUSE_EVENTS");
    if (sched->n_copies == 1 && (reuse_env == nullptr || atoi(reuse_env) != 0)) {
        for (int b = 0; b < n_backends; b++) {
            ggml_backend_dev_t dev = backends[b]->device;
            if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                sched->reuse_events[b] = ggml_backend_event_new(dev);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;
    sched->stage = ggml_backend_sched_stage_new(sched);
    sched->backends_idle = true;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_sched_worker_stop(sched->workers[b]);
        sched->workers[b] = nullptr;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
            ggml_backend_event_free(sched->slot_events[b][c]);
        }
        if (sched->xdev_events[b] != NULL) {
            ggml_backend_event_free(sched->xdev_events[b]);
        }
        if (sched->reuse_events[b] != NULL) {
            ggml_backend_event_free(sched->reuse_events[b]);
        }
        if (sched->host_read_events[b] != NULL) {
            ggml_backend_event_free(sched->host_read_events[b]);
        }
    }
    delete sched->input_staging;
    ggml_backend_sched_stage_free(sched->stage);
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->hv_offload_decisions);
    free(sched->hv_written);
    free(sched->hv_touched);
    free(sched->trace_tensor_ids);
    free(sched->trace_backend_node_ids);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    sched->trace_graph_id = GGML_EXEC_TRACE_ID_NONE;
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        if (sched->scoped_reset) {
            // clear only the slots written since the last reset; every reader of these
            // tables treats a slot without the written bit as the cleared default
            const size_t n_copy_cells = (size_t) sched->n_backends*sched->n_copies;
            for (size_t t = 0; t < sched->hv_touched_count; ++t) {
                const size_t id = sched->hv_touched[t];
                GGML_ASSERT(id < sched->hash_set.size);
                memset(&sched->hv_tensor_copies[id*n_copy_cells], 0, n_copy_cells*sizeof(struct ggml_tensor *));
                memset(&sched->hv_offload_decisions[id], 0, sizeof(sched->hv_offload_decisions[0]));
                ggml_bitset_clear(sched->hv_written, id);
            }
            sched->hv_touched_count = 0;
        } else {
            memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
            memset(sched->hv_offload_decisions,   0, sched->hash_set.size * sizeof(sched->hv_offload_decisions[0]));
            memset(sched->hv_written, 0, ggml_bitset_size(sched->hash_set.size)*sizeof(ggml_bitset_t));
        }
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }
    sched->n_reserves++;

    ggml_backend_sched_reset(sched);

    return true;
}

uint64_t ggml_backend_sched_get_n_reserves(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_reserves;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    if (ggml_exec_trace_enabled()) {
        try {
            ggml_backend_sched_trace_graph(sched, graph);
        } catch (...) {
            ggml_exec_trace_disable("failed to record scheduler topology");
        }
    } else if (ggml_exec_trace_integrity_enabled()) {
        ggml_backend_sched_integrity_graph(sched, graph);
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_wait_graph(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (sched->n_copies <= 1) {
        ggml_backend_sched_synchronize(sched);
        return;
    }
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_sched_worker_join(sched->workers[i]);
    }
    // the slot event of cur_copy is recorded after the last split of the last graph on each backend
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->slot_events[i][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->slot_events[i][sched->cur_copy]);
        } else {
            ggml_backend_synchronize(sched->backends[i]);
        }
    }
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_sched_worker_join(sched->workers[i]);
    }
    for (int i = 0; i < sched->n_backends; i++) {
        const uint64_t wait_start = ggml_backend_sched_trace_span_begin();
        ggml_backend_synchronize(sched->backends[i]);
        ggml_backend_sched_trace_span_end(
                sched, wait_start, GGML_EXEC_TRACE_EVENT_BACKEND_WAIT,
                GGML_EXEC_TRACE_WAIT_SCHEDULER_SYNCHRONIZE, -1, i,
                GGML_EXEC_TRACE_ID_NONE, GGML_EXEC_TRACE_WAIT_API_BACKEND_SYNCHRONIZE);
    }
    std::fill(sched->host_read_pending, sched->host_read_pending + GGML_SCHED_MAX_BACKENDS, false);
    sched->backends_idle = true;
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_input_async(ggml_backend_sched_t sched, bool enable) {
    GGML_ASSERT(sched);
    sched->input_async = enable;
}

void ggml_backend_sched_set_input_staging(ggml_backend_sched_t sched, bool enable) {
    GGML_ASSERT(sched);
    if (enable && !sched->input_staging) {
        sched->input_staging = new ggml_backend_sched_input_staging();
    } else if (!enable) {
        delete sched->input_staging;
        sched->input_staging = nullptr;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
