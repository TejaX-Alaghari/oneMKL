/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#ifndef _ROCBLAS_SCOPED_HANDLE_HPP_
#define _ROCBLAS_SCOPED_HANDLE_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#include <sycl/detail/backend_traits_hip.hpp>
#include <sycl/detail/common.hpp>
#else
#include <CL/sycl.hpp>
#include <CL/sycl/detail/backend_traits_hip.hpp>
#include <CL/sycl/detail/common.hpp>
#endif
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include "rocblas_helper.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {

template <typename T>
struct rocblas_handle_container {
    using handle_container_t = std::unordered_map<T, std::atomic<rocblas_handle> *>;
    handle_container_t rocblas_handle_container_mapper_{};
    ~rocblas_handle_container() noexcept(false);
};

class RocblasScopedContextHandler {
    HIPcontext original_;
    sycl::context *placedContext_;
    bool needToRecover_;
    sycl::interop_handler &ih;
    static thread_local rocblas_handle_container<pi_context> handle_helper;
    sycl::context get_context(const sycl::queue &queue);
    hipStream_t get_stream(const sycl::queue &queue);

public:
    RocblasScopedContextHandler(sycl::queue queue, sycl::interop_handler &ih);

    ~RocblasScopedContextHandler() noexcept(false);

    /**
     * @brief get_handle: creates the handle by implicitly impose the advice
     * given by rocm for creating a rocblas_handle. (e.g. one hipStream per device
     * per thread).
     * @param queue sycl queue.
     * @return rocblas_handle a handle to construct rocblas routines
     */
    rocblas_handle get_handle(const sycl::queue &queue);
    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T get_mem(U acc) {
        return reinterpret_cast<T>(ih.get_mem<sycl::backend::hip>(acc));
    }
};

} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif //_ROCBLAS_SCOPED_HANDLE_HPP_
