#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include <mooncake_ep_exception.cuh>
#include <c10/core/Event.h>

namespace mooncake {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        C10_CUDA_CHECK(cudaStreamWaitEvent(
            at::cuda::getCurrentCUDAStream().stream(),
            event->cuda_event(),
            0));
    }
};

inline torch::Event create_event(const at::cuda::CUDAStream& s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

inline void stream_wait(const at::cuda::CUDAStream& s_0,
                        const at::cuda::CUDAStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    C10_CUDA_CHECK(cudaStreamWaitEvent(
        s_0.stream(),
        create_event(s_1).cuda_event(),
        0
    ));
}

inline void stream_wait(const at::cuda::CUDAStream& s,
                        const EventHandle& event) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(
        s.stream(),
        event.event->cuda_event(),
        0
    ));
}

}  // namespace mooncake
