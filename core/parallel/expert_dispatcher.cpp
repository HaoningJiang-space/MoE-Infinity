// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "model/model_topology.h"
#include "model/moe.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <future>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace {
void RecordAtomicMax(std::atomic<std::uint64_t>& target,
                     std::uint64_t value) {
  auto current = target.load();
  while (value > current && !target.compare_exchange_weak(current, value)) {
  }
}

std::chrono::milliseconds PendingStallTimeout() {
  const char* timeout_ms = std::getenv("MOE_INFINITY_PENDING_STALL_TIMEOUT_MS");
  if (timeout_ms == nullptr || timeout_ms[0] == '\0') {
    return std::chrono::seconds(300);
  }

  char* end = nullptr;
  long long parsed = std::strtoll(timeout_ms, &end, 10);
  if (end == timeout_ms || parsed <= 0) {
    return std::chrono::seconds(300);
  }
  return std::chrono::milliseconds(parsed);
}
}  // namespace

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type, int num_threads)
    : pending_(0),
      num_enqueued_(0),
      start_(false),
      expert_type_(expert_type),
      dtype_(dtype),
      num_experts_(num_experts),
      // input_mutex_(kNumDevices),
      // input_cv_(kNumDevices),
      // exec_mutex_(kNumDevices),
      // exec_cv_(kNumDevices),
      cache_mutex_(kNumDevices()),
      cache_cv_(kNumDevices()),
      input_queue_(kNumDevices()),
      gpu_overload_(kNumDevices()),
      exec_queue_(kNumDevices()),
      cached_experts_(kNumDevices()),
      modules_(kNumDevices(), nullptr) {
  main_thread_stop_flag_.store(false);
  for (auto& overload : gpu_overload_) {
    overload.store(false, std::memory_order_relaxed);
  }

  // module_ = new MoEMLP(dtype, expert_type);

  // Futex<bool> initial_value(false);
  // gpu_overload_ = std::move(std::vector<Futex<bool>>(kNumDevices,
  // initial_value));

  for (int i = 0; i < kNumDevices(); ++i) {
    auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
    std::string thread_name = "GPUFetchFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);

    modules_[i] = new MoEMLP(dtype, expert_type);
    // gpu_overload_.emplace_back(false);
  }

  for (int i = 0; i < kNumDevices() * num_threads; ++i) {
    cudaSetDevice(i % kNumDevices());
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.emplace_back(exec_stream);
    // cudaDeviceSynchronize();

    auto thread_func =
        std::bind(&ExpertDispatcher::GPUExecFunc, this, i % kNumDevices());
    std::string thread_name =
        "GPUExecFunc" + std::to_string(i % kNumDevices());
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());
  }

  at::InferenceMode infer_guard(0);

  for (int i = 0; i < num_experts; ++i) {
    experts_.emplace_back();
    for (int j = 0; j < num_layers; ++j) {
      experts_[i].emplace_back();
      experts_[i][j] = std::make_shared<ExpertNode>();
      experts_[i][j]->expert_type = expert_type;
      int expert_type = expert_type_;
      switch (expert_type) {
        case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
          experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
          break;
        case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
          experts_[i][j]->module =
              new SwitchTransformersDenseGatedActDense(dtype);
          break;
        case NLLB_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new NllbMoeDenseActDense(dtype);
          break;
        case FSGPT_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new FSGPTMoEDenseActDense(dtype);
          break;
        case MIXTRAL_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new MixtralMoEDenseActDense(dtype);
          break;
        case DEEPSEEK_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new DeepSeekMoEDenseActDense(dtype);
          break;
        default:
          DLOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                     expert_type);
      }
      experts_[i][j]->module->eval();
      experts_[i][j]->layer_idx = j;
      experts_[i][j]->expert_idx = i;
    }
  }
}

ExpertDispatcher::~ExpertDispatcher() {
  main_thread_stop_flag_.store(true);
  ShutdownQueues();
  for (auto& thread : threads_) {
    thread->join();
  }

  for (auto& stream : exec_streams_) {
    cudaStreamDestroy(stream);
  }
}

void ExpertDispatcher::ShutdownQueues() {
  for (int gpu_id = 0; gpu_id < kNumDevices(); ++gpu_id) {
    CallArgs input_sentinel;
    input_sentinel.layer_idx = -1;
    input_sentinel.expert_idx = -1;
    input_sentinel.gpu_id = gpu_id;
    input_queue_[gpu_id].Push(input_sentinel);
  }

  int exec_threads_per_gpu =
      std::max<int>(1, static_cast<int>(exec_streams_.size()) / kNumDevices());
  for (int gpu_id = 0; gpu_id < kNumDevices(); ++gpu_id) {
    for (int i = 0; i < exec_threads_per_gpu; ++i) {
      ExecArgs exec_sentinel;
      exec_queue_[gpu_id].Push(exec_sentinel);
    }
  }
}

void ExpertDispatcher::EnqueueExpert(int layer_idx, int expert_idx, int gpu_id,
                                     bool remote) {
  ExpertDispatcher::CallArgs args;
  args.layer_idx = layer_idx;
  args.expert_idx = expert_idx;
  args.gpu_id = gpu_id;
  args.remote = remote;
  Enqueue(args);
}

void ExpertDispatcher::Enqueue(CallArgs& args) {
  // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
  int layer_idx = args.layer_idx;
  int expert_idx = args.expert_idx;
  auto expert_node = experts_[expert_idx][layer_idx];

  if (!expert_node->node->mutex.try_lock()) {
    auto wait_start = std::chrono::steady_clock::now();
    const auto kBusyNodeTimeout = PendingStallTimeout();
    constexpr auto kBusyNodePollInterval = std::chrono::milliseconds(1);
    DLOG_WARN("ExpertDispatcher::Enqueue: waiting on busy expert node "
              "(expert_idx ",
              expert_idx, " layer_idx ", layer_idx, "node ",
              expert_node->node->str(), ")");
    while (!expert_node->node->mutex.try_lock()) {
      auto now = std::chrono::steady_clock::now();
      if (now - wait_start >= kBusyNodeTimeout) {
        auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           now - wait_start)
                           .count();
        busy_wait_count_.fetch_add(1);
        busy_wait_total_wait_us_.fetch_add(wait_us);
        RecordAtomicMax(busy_wait_max_wait_us_,
                        static_cast<std::uint64_t>(wait_us));
        std::ostringstream oss;
        oss << "ExpertDispatcher::Enqueue busy expert node timeout: expert_idx="
            << expert_idx << " layer_idx=" << layer_idx
            << " wait_us=" << wait_us
            << " pending=" << pending_.load()
            << " enqueue=" << enqueue_count_.load()
            << " fetch_dequeue=" << fetch_dequeue_count_.load()
            << " exec_dequeue=" << exec_dequeue_count_.load()
            << " output=" << output_count_.load() << " node="
            << expert_node->node->str();
        throw std::runtime_error(oss.str());
      }
      std::this_thread::sleep_for(kBusyNodePollInterval);
    }
    auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - wait_start)
                       .count();
    busy_wait_count_.fetch_add(1);
    busy_wait_total_wait_us_.fetch_add(wait_us);
    RecordAtomicMax(busy_wait_max_wait_us_,
                    static_cast<std::uint64_t>(wait_us));
  }
  expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

  if (expert_node->node->device.is_cuda()) {
    cache_hit_fetch_count_.fetch_add(1);
    if (kTaskPool != nullptr && kTaskPool->IsCacheCandidate(expert_node->node)) {
      candidate_resident_hit_count_.fetch_add(1);
    }
    if ((expert_node->node->io_state & NODE_STATE_PREFETCHED) != 0) {
      prefetch_resident_hit_count_.fetch_add(1);
    }
    args.gpu_id = expert_node->node->device.index();

    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();

    ExecArgs exec_args;
    // exec_args.hidden_states = std::move(input);
    exec_args.expert_node = expert_node;
    expert_node->SetTensorsFromBlob(expert_node->node->device);
    exec_args.out_gpu_id = original_device.index();
    exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
    exec_args.evict = false;
    exec_args.hit = true;

    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    // std::unique_lock<std::mutex> lock(exec_mutex_[args.gpu_id]);
    // exec_queue_[args.gpu_id].push_back(std::move(exec_args));
    exec_queue_[args.gpu_id].Push(exec_args);
  } else {
    // std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
    // input_queue_[args.gpu_id].push_back(std::move(args));
    input_queue_[args.gpu_id].Push(args);
  }
  // input_cv_[args.gpu_id].notify_all();
  // exec_cv_[args.gpu_id].notify_all();
  // input_queue_.push_back(std::move(args));
  num_enqueued_.fetch_add(1);
  enqueue_count_.fetch_add(1);

  // auto& a = input_queue_.back();
  // if (expert_node->node->device.is_cuda()) {
  //   a.gpu_id = expert_node->node->device.index();
  // }
  // DLOG_TRACE("ExpertDispatcher::Enqueue: num_enqueued_ ",
  // num_enqueued_.load(),
  //            "input_queue_ ", input_queue_.size(), "gpu_id ", a.gpu_id,
  //            "layer_idx ", a.layer_idx, "expert_idx ", a.expert_idx, "remote
  //            ", a.remote);
  // lock.unlock();
  // cvs_[MUTEX_TYPE::INPUT_MUTEX].notify_all();
}

std::vector<std::uint64_t> ExpertDispatcher::GetRuntimeStats() const {
  return {
      enqueue_count_.load(),
      busy_wait_count_.load(),
      busy_wait_total_wait_us_.load(),
      busy_wait_max_wait_us_.load(),
      cache_hit_fetch_count_.load(),
      cache_miss_fetch_count_.load(),
      eviction_count_.load(),
      all_locked_event_count_.load(),
      no_victim_wait_count_.load(),
      no_victim_wait_total_us_.load(),
      no_victim_wait_max_us_.load(),
      fetch_dequeue_count_.load(),
      exec_dequeue_count_.load(),
      output_count_.load(),
      pending_wait_count_.load(),
      pending_wait_total_us_.load(),
      pending_wait_max_us_.load(),
      pending_stall_count_.load(),
      prefetch_resident_hit_count_.load(),
      late_prefetch_demand_miss_count_.load(),
      demand_candidate_protect_skip_count_.load(),
      demand_candidate_protect_fallback_count_.load(),
      candidate_resident_hit_count_.load(),
      candidate_demand_miss_count_.load(),
  };
}

void ExpertDispatcher::ResetRuntimeStats() {
  enqueue_count_.store(0);
  busy_wait_count_.store(0);
  busy_wait_total_wait_us_.store(0);
  busy_wait_max_wait_us_.store(0);
  cache_hit_fetch_count_.store(0);
  cache_miss_fetch_count_.store(0);
  eviction_count_.store(0);
  all_locked_event_count_.store(0);
  no_victim_wait_count_.store(0);
  no_victim_wait_total_us_.store(0);
  no_victim_wait_max_us_.store(0);
  fetch_dequeue_count_.store(0);
  exec_dequeue_count_.store(0);
  output_count_.store(0);
  pending_wait_count_.store(0);
  pending_wait_total_us_.store(0);
  pending_wait_max_us_.store(0);
  pending_stall_count_.store(0);
  prefetch_resident_hit_count_.store(0);
  late_prefetch_demand_miss_count_.store(0);
  demand_candidate_protect_skip_count_.store(0);
  demand_candidate_protect_fallback_count_.store(0);
  candidate_resident_hit_count_.store(0);
  candidate_demand_miss_count_.store(0);
}

void ExpertDispatcher::RegisterExpert(
    int layer_idx, int expert_idx, const std::vector<std::uint32_t>& tensor_ids,
    std::string jit_path) {
  NodePtr cached_node = nullptr;
  for (auto tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (cached_node == nullptr) {
      cached_node = node;
      experts_[expert_idx][layer_idx]->node = node;
      // experts_[expert_idx][layer_idx]->jit_module =
      //     new torch::jit::script::Module(torch::jit::load(jit_path));
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
  }
}

void ExpertDispatcher::NotifyFetchStart() {
  for (int i = 0; i < kNumDevices(); ++i) {
    // std::unique_lock<std::mutex> lock(input_mutex_[i]);
    input_queue_[i].NotifyAll();
  }
}

void ExpertDispatcher::ClearExpertCacheCounts() {
  for (auto& expert : experts_) {
    for (auto& expert_node : expert) {
      if (expert_node->node == nullptr) {
        continue;
      }
      expert_node->node->incache_visit_count = 0;
    }
  }
}

void ExpertDispatcher::ResetExpertCacheState() {
  WaitForPendingZero("ResetExpertCacheState");
  if (kTaskPool != nullptr) {
    kTaskPool->ReplaceCacheCandidates({});
  }

  for (auto& queue : input_queue_) {
    queue.Clear();
  }
  for (auto& queue : exec_queue_) {
    queue.Clear();
  }
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue_.clear();
  }

  std::unordered_set<Node*> visited;
  for (auto& expert_by_layer : experts_) {
    for (auto& expert_node : expert_by_layer) {
      if (expert_node == nullptr || expert_node->node == nullptr) {
        continue;
      }
      auto node = expert_node->node;
      if (visited.find(node.get()) != visited.end()) {
        continue;
      }
      visited.insert(node.get());
      std::unique_lock<std::mutex> node_lock(node->mutex);
      if (node->device.is_cuda()) {
        node->SetDevice(node->default_host);
      }
      node->incache_visit_count = 0;
      node->unused_count = 0;
      node->is_overflow = false;
      node->io_state = NODE_STATE_NONE;
      node->state = 0;
      node->cv.notify_all();
    }
  }

  for (int gpu_id = 0; gpu_id < kNumDevices(); ++gpu_id) {
    {
      std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
      cached_experts_[gpu_id].clear();
      cache_sizes_[gpu_id] =
          kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, gpu_id));
      gpu_overload_[gpu_id].store(false, std::memory_order_release);
    }
    cache_cv_[gpu_id].notify_all();
  }
  pending_.store(0);
}

// void ExpertDispatcher::GPUThreadFunc(int gpu_id) {
//   while (!main_thread_stop_flag_.load()) {
//   }
// }

ExpertNodePtr ExpertDispatcher::FindExpertEvict(int gpu_id) {
  uint64_t min_visit_count = INT_MAX;
  ExpertNodePtr evict_expert_node = nullptr;
  bool skipped_candidate = false;
  const bool protect_candidates =
      kTaskPool != nullptr &&
      kTaskPool->CandidateDemandEvictionProtectionEnabled();

  auto scan = [&](bool allow_candidates) {
    for (auto& key : cached_experts_[gpu_id]) {
      auto layer_idx = key >> 32;
      auto expert_idx = key & 0xFFFFFFFF;
      auto node = experts_[expert_idx][layer_idx]->node;
      if (node == nullptr) continue;
      if (!allow_candidates && protect_candidates &&
          kTaskPool->IsCacheCandidate(node)) {
        skipped_candidate = true;
        demand_candidate_protect_skip_count_.fetch_add(1);
        continue;
      }
      if (node->device.is_cuda() &&
          node->incache_visit_count < min_visit_count && node->mutex.try_lock()) {
        if (evict_expert_node != nullptr) {
          evict_expert_node->node->mutex.unlock();
        }
        evict_expert_node = experts_[expert_idx][layer_idx];
        min_visit_count = node->incache_visit_count;
      }
    }
  };

  scan(/*allow_candidates=*/false);
  if (evict_expert_node == nullptr && skipped_candidate) {
    demand_candidate_protect_fallback_count_.fetch_add(1);
    scan(/*allow_candidates=*/true);
  }
  return evict_expert_node;
}

void ExpertDispatcher::GPUFetchFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  while (!main_thread_stop_flag_.load()) {
    // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
    // if (cache_ == nullptr) {
    //   auto cache_limit =
    //   kDeviceMemoryPool->GetSparseCacheLimit(torch::Device(torch::kCUDA,
    //   gpu_id));
    //   // get any one expert size
    //   auto num_layers = experts_[0].size();
    //   auto num_experts = experts_.size();
    //   auto expert_node = experts_[num_layers-1][num_experts-1];

    //   int cache_capacity = cache_limit / expert_node->node->byte_size;
    //   cache_capacity_ = cache_capacity;
    // }
    // std::unique_lock<std::mutex> lock(input_mutex_[gpu_id]);
    // input_cv_[gpu_id].wait(lock, [&] { return !input_queue_[gpu_id].empty();
    // });

    // CallArgs args = std::move(input_queue_[gpu_id].front());
    // input_queue_[gpu_id].pop_front();

    // lock.unlock();
    CallArgs args;
    input_queue_[gpu_id].Pop(args);
    fetch_dequeue_count_.fetch_add(1);
    if (main_thread_stop_flag_.load() && args.layer_idx < 0) {
      break;
    }

    auto device = CUDA_DEVICE(gpu_id);
    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();
    int64_t layer_idx = args.layer_idx;
    int64_t expert_idx = args.expert_idx;
    int64_t batch_size = hidden_states_.size(0);

    auto expert_node = experts_[expert_idx][layer_idx];
    bool cache_hit = expert_node->node->device.is_cuda();
    bool is_candidate = kTaskPool != nullptr &&
                        kTaskPool->IsCacheCandidate(expert_node->node);
    if (cache_hit) {
      cache_hit_fetch_count_.fetch_add(1);
      if (is_candidate) {
        candidate_resident_hit_count_.fetch_add(1);
      }
      if ((expert_node->node->io_state & NODE_STATE_PREFETCHED) != 0) {
        prefetch_resident_hit_count_.fetch_add(1);
      }
    } else {
      cache_miss_fetch_count_.fetch_add(1);
      if (is_candidate) {
        candidate_demand_miss_count_.fetch_add(1);
      }
      if (kTaskPool != nullptr &&
          kTaskPool->HasPendingPrefetch(expert_node->node)) {
        late_prefetch_demand_miss_count_.fetch_add(1);
      }
    }

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: gpu_id " << gpu_id
    //           << " layer_idx " << layer_idx << " expert_idx " << expert_idx
    //           << " cache_hit " << cache_hit << " node "
    //           << expert_node->node->device.str() << std::endl;
    DLOG_DEBUG("ExpertDispatcher::GPUFetchFunc: gpu_id ", gpu_id, " layer_idx ",
               layer_idx, " expert_idx ", expert_idx, "cache_hit ", cache_hit,
               "cache_size ", cache_sizes_[gpu_id], " incache count ",
               cached_experts_[gpu_id].size());

    bool overload_fetch = false;
    if (!cache_hit && cache_sizes_[gpu_id] < expert_node->node->byte_size) {
      if (batch_size > 1) {
        // force fetch to GPU regardless of cache size, only for prefill
        // only one extra cache slot for prefill
        DLOG_DEBUG("overloading expert cache: gpu_id ", gpu_id, " cache size ",
                   cache_sizes_[gpu_id], " incache count ",
                   cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                   " expert_idx ", expert_idx);
        {
          std::unique_lock<std::mutex> lock(cache_mutex_[gpu_id]);
          cache_cv_[gpu_id].wait(lock, [&] {
            return main_thread_stop_flag_.load() ||
                   !gpu_overload_[gpu_id].load(std::memory_order_acquire);
          });
          if (main_thread_stop_flag_.load()) {
            continue;
          }
          gpu_overload_[gpu_id].store(true, std::memory_order_release);
        }
        overload_fetch = true;
      } else {
        // find the expert in gpu and min incache_visit_count
        ExpertNodePtr evict_expert_node = FindExpertEvict(gpu_id);
        while (evict_expert_node == nullptr && !main_thread_stop_flag_.load()) {
          all_locked_event_count_.fetch_add(1);
          no_victim_wait_count_.fetch_add(1);
          auto no_victim_wait_start = std::chrono::steady_clock::now();
          // wait for notification that cache is available
          DLOG_WARN(
              "All cached expert locked, waiting for cache to be available. "
              "gpu_id ",
              gpu_id, " cache size ", cache_sizes_[gpu_id], " incache count ",
              cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
              " expert_idx ", expert_idx);
          {
            std::unique_lock<std::mutex> lock(cache_mutex_[gpu_id]);
            cache_cv_[gpu_id].wait_for(lock, std::chrono::milliseconds(100));
          }
          auto no_victim_wait_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - no_victim_wait_start)
                  .count();
          auto no_victim_wait_us_u64 =
              static_cast<std::uint64_t>(no_victim_wait_us);
          no_victim_wait_total_us_.fetch_add(no_victim_wait_us_u64);
          RecordAtomicMax(no_victim_wait_max_us_, no_victim_wait_us_u64);
          evict_expert_node = FindExpertEvict(gpu_id);
        }
        // auto num_layers = experts_[0].size();
        // auto num_experts = experts_.size();

        // for (size_t i = 0; i < num_experts; ++i) {
        //   for (size_t j = 0; j < num_layers; ++j) {
        // auto node = experts_[i][j]->node;
        // if (node == nullptr) {
        //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: node is nullptr"
        //   //           << " layer_idx " << j << " expert_idx " << i <<
        //   //           std::endl;
        //   continue;
        // }
        // if (node->device.is_cuda() &&
        //     node->incache_visit_count < min_visit_count &&
        //     node->mutex.try_lock()) {
        //   evict_node = node;
        //   min_visit_count = node->incache_visit_count;
        //   node->mutex.unlock();
        //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: evict node "
        //   //           << evict_node->device.str() << " incache_visit_count "
        //   //           << min_visit_count << std::endl;
        // }
        //   }
        // }
        if (evict_expert_node == nullptr) {
          DLOG_WARN(
              "ExpertDispatcher::GPUFetchFunc: no victim available during "
              "shutdown, gpu_id ",
              gpu_id, " cache size ", cache_sizes_[gpu_id],
              " in cache count ", cached_experts_[gpu_id].size());
          continue;
        }

        DLOG_DEBUG("evicting expert: gpu_id ", gpu_id, " cache size ",
                   cache_sizes_[gpu_id], " incache count ",
                   cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                   " expert_idx ", expert_idx);
        eviction_count_.fetch_add(1);

        auto evict_node = evict_expert_node->node;
        std::unique_lock<std::mutex> evict_lock(evict_node->mutex,
                                                std::adopt_lock);
        evict_node->SetDevice(evict_node->default_host);
        cache_sizes_[gpu_id] += evict_node->byte_size;
        int64_t evict_layer_idx = evict_expert_node->layer_idx;
        int64_t evict_expert_idx = evict_expert_node->expert_idx;

        // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
        uint64_t evict_key = (evict_layer_idx << 32) + evict_expert_idx;
        auto it = cached_experts_[gpu_id].find(evict_key);
        if (it != cached_experts_[gpu_id].end()) {
          cached_experts_[gpu_id].erase(it);
        } else {
          DLOG_FATAL(
              "ExpertDispatcher::GPUFetchFunc: evict_key not found. layer_idx ",
              evict_layer_idx, " expert_idx ", evict_expert_idx);
        }
      }
    }

    if (!overload_fetch) {
      cache_sizes_[gpu_id] -= expert_node->node->byte_size;
      uint64_t key = (layer_idx << 32) + expert_idx;
      cached_experts_[gpu_id].insert(key);
    }

    expert_node->node->SetDevice(device, true, stream);
    expert_node->node->incache_visit_count += 1;
    expert_node->SetTensorsFromBlob(device);
    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: move to device gpu_id "
    //           << gpu_id << " layer_idx " << layer_idx << " expert_idx "
    //           << expert_idx << " node "
    //           << expert_node->node->device.str() << std::endl;

    // int expert_type = expert_type_;
    // torch::Tensor input;
    // auto token_indices =
    //     router_mask_.index({"...", expert_idx}).to(torch::kBool);
    // switch (expert_type) {
    //   case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
    //   case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
    //   case NLLB_MOE_DENSE_ACT_DENSE:
    //   case FSGPT_MOE_DENSE_ACT_DENSE:
    //   case MIXTRAL_MOE_DENSE_ACT_DENSE:
    //   case DEEPSEEK_MOE_DENSE_ACT_DENSE:
    //     input =
    //         hidden_states_.index({token_indices}).to(expert_node->node->device);
    //     break;
    //   default:
    //     DLOG_FATAL("ExpertDispatcher::expert_type: unknown expert type ",
    //                expert_type);
    // }

    // DLOG_TRACE("ExpertDispatcher::GPUFetchFunc gpu_id ", gpu_id, "layer_idx
    // ",
    //            layer_idx, "expert_idx ", expert_idx, "input ",
    //            input.device().str(), "node ",
    //            expert_node->node->device.str());
    {
      ExecArgs exec_args;
      // exec_args.hidden_states = std::move(input);
      exec_args.expert_node = expert_node;
      exec_args.out_gpu_id = original_device.index();
      exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
      exec_args.evict = overload_fetch;
      exec_args.hit = cache_hit;
      // std::lock_guard<std::mutex> lock(exec_mutex_[gpu_id]);
      // exec_queue_[gpu_id].emplace_back(std::move(exec_args));
      exec_queue_[gpu_id].Push(exec_args);
    }
    // exec_cv_[gpu_id].notify_all();
  }

  cudaStreamDestroy(stream);
}

void ExpertDispatcher::GPUExecFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  while (!main_thread_stop_flag_.load()) {
    // std::unique_lock<std::mutex> lock(exec_mutex_[gpu_id]);
    // exec_cv_[gpu_id].wait(lock, [&] { return !exec_queue_[gpu_id].empty();
    // });

    // ExecArgs args = std::move(exec_queue_[gpu_id].front());
    // exec_queue_[gpu_id].pop_front();

    // lock.unlock();

    ExecArgs args;
    exec_queue_[gpu_id].Pop(args);
    exec_dequeue_count_.fetch_add(1);

    if (args.expert_node == nullptr) {
      if (main_thread_stop_flag_.load()) {
        break;
      }
      continue;
    }

    int64_t batch_size = hidden_states_.size(0);
    auto device = CUDA_DEVICE(gpu_id);
    auto expert_idx = args.expert_node->expert_idx;

    auto token_mask = router_mask_.index({"...", expert_idx});
    if (token_mask.device() != hidden_states_.device()) {
      token_mask = token_mask.to(hidden_states_.device(), /*non_blocking=*/true);
    }
    torch::Tensor input = (batch_size == 1)
                              ? hidden_states_.to(device)
                              : hidden_states_.index({token_mask}).to(device);

    // args.hidden_states = std::move(input);
    // assert(args.hidden_states.sum().to(torch::kCPU).item<float>() != 0);
    // at::InferenceMode infer_guard(true);

    // // prepare jit input vector
    // std::vector<torch::jit::IValue> jit_inputs;
    // jit_inputs.push_back(input);

    // cudaDeviceSynchronize();

    modules_[gpu_id]->SetTensorsFromIds(args.expert_node->node->tensor_ids);

    // random int [0,8)
    // int rnd = std::rand() % kNumDevices;
    c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromExternal(stream, gpu_id);
    c10::cuda::CUDAStreamGuard guard(torch_stream);
    // auto start = TIME_NOW;
    // c10::cuda::CUDAStreamGuard guard(stream);

    // auto* expert_module = args.expert_node->module;
    // int expert_type = expert_type_;
    // cudaStreamSynchronize(stream);  // make sure the input is ready

    auto output = modules_[gpu_id]->forward(input, stream);
    OutputFunc(args, output, token_mask, gpu_id);
  }

  cudaStreamDestroy(stream);
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  torch::Tensor token_mask, int gpu_id) {
  auto output_device =
      (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
  torch::Tensor output_tensor = output.to(output_device).to(torch::kFloat32);

  DLOG_TRACE("ExpertDispatcher::OutputFunc: output_tensor ",
             output_tensor.sizes().vec(), "(", output_tensor.device().str(),
             ")");

  // args.expert_node->node->mutex.unlock();
  int64_t expert_idx = args.expert_node->expert_idx;
  int64_t layer_idx = args.expert_node->layer_idx;
  int64_t batch_size = hidden_states_.size(0);

  args.expert_node->node->mutex.unlock();
  if (args.evict) {
    // pop out overloaded expert such that cache is not polluted
    args.expert_node->node->SetDevice(args.expert_node->node->default_host,
                                      true, nullptr);
    // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
    // uint64_t key = (layer_idx << 32) + expert_idx;
    // auto it = cached_experts_[gpu_id].find(key);
    // if (it != cached_experts_[gpu_id].end()) {
    //   cached_experts_[gpu_id].erase(it);
    // } else {
    //   DLOG_FATAL(
    //       "ExpertDispatcher::OutputFunc: expert not found in cache. gpu_id",
    //       gpu_id, "layer_idx ", layer_idx, "expert_idx ", expert_idx);
    // }
    // cache_sizes_[gpu_id] += args.expert_node->node->byte_size;
    DLOG_DEBUG("pop out overloaded expert cache_sizes_[gpu_id] ",
               cache_sizes_[gpu_id], "gpu_id ", gpu_id, "layer_idx ", layer_idx,
               "expert_idx ", expert_idx);
    // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
    // gpu_overload_[gpu_id].set_and_wake(true);
    {
      std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
      gpu_overload_[gpu_id].store(false, std::memory_order_release);
    }
  }
  cache_cv_[gpu_id].notify_all();

  // if (args.evict) {
  //   args.expert_node->node->SetDevice(args.expert_node->node->default_host,
  //                                     true, nullptr);
  //   {
  //     std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
  //     gpu_overload_[gpu_id] = false;
  //   }
  // }

  if (batch_size == 1) {
    auto router_weight = router_weight_;
    if (router_weight.device() != output_tensor.device()) {
      router_weight = router_weight.to(output_tensor.device(), /*non_blocking=*/true);
    }
    final_hidden_states_.add_(
        output_tensor *
        router_weight.index({torch::indexing::Slice(), expert_idx}));
  } else {
    auto mask_for_weight = token_mask;
    if (mask_for_weight.device() != router_weight_.device()) {
      mask_for_weight = mask_for_weight.to(
          router_weight_.device(), /*non_blocking=*/true);
    }
    auto token_indices = torch::nonzero(token_mask).squeeze(1);
    if (token_indices.device() != final_hidden_states_.device()) {
      token_indices =
          token_indices.to(final_hidden_states_.device(), /*non_blocking=*/true);
    }
    auto weights = router_weight_.index({mask_for_weight, expert_idx}).unsqueeze(1);
    if (weights.device() != output_tensor.device()) {
      weights = weights.to(output_tensor.device(), /*non_blocking=*/true);
    }
    auto weighted_output = output_tensor * weights;
    final_hidden_states_.index_add_(0, token_indices, weighted_output);
  }
  // {
  //   std::lock_guard<std::mutex> lock(output_mutex_);
  //   output_queue_.emplace_back(std::move(output_tensor),
  //                              args.expert_node->layer_idx,
  //                              args.expert_node->expert_idx, args.hit);
  //   DLOG_TRACE("ExpertDispatcher::OutputFunc: output_queue_",
  //              output_queue_.size(), "output",
  //              std::get<0>(output_queue_.back()).device().str(), "evict",
  //              args.evict, "(", args.expert_node->layer_idx,
  //              args.expert_node->expert_idx, gpu_id, args.hit, ")");
  // }

  // stream.synchronize();
  output_count_.fetch_add(1);
  pending_.fetch_sub(1);
  if (pending_.load() == 0) {
    pending_cv_.notify_all();
  }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait() {
  // int wait_count = 0;

  WaitForPendingZero("Wait");

  num_enqueued_.store(0);
  std::vector<CallResult> output_queue;
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue.swap(output_queue_);
  }

  return output_queue;
}

torch::Tensor ExpertDispatcher::WaitHiddenStates() {
  WaitForPendingZero("WaitHiddenStates");
  num_enqueued_.store(0);
  return final_hidden_states_;
}

void ExpertDispatcher::WaitForPendingZero(const char* caller) {
  if (pending_.load() == 0) {
    return;
  }
  pending_wait_count_.fetch_add(1);

  constexpr auto kPollInterval = std::chrono::milliseconds(1000);
  constexpr auto kLogInterval = std::chrono::seconds(30);
  const auto kStallTimeout = PendingStallTimeout();

  auto wait_start = std::chrono::steady_clock::now();
  auto last_log = wait_start;
  auto last_progress = wait_start;
  auto progress_signature = [&]() -> std::uint64_t {
    // A no-victim retry proves the fetch thread is alive, but it does not
    // retire a pending expert. Treat only actual queue/eviction/output movement
    // as forward progress for the pending-stall guard.
    return fetch_dequeue_count_.load() + exec_dequeue_count_.load() +
           output_count_.load() + eviction_count_.load();
  };
  std::uint64_t last_signature = progress_signature();

  std::unique_lock<std::mutex> lock(pending_mutex_);
  while (pending_.load() != 0) {
    if (pending_cv_.wait_for(lock, kPollInterval,
                             [&] { return pending_.load() == 0; })) {
      break;
    }
    auto now = std::chrono::steady_clock::now();
    auto signature = progress_signature();
    if (signature != last_signature) {
      last_signature = signature;
      last_progress = now;
    }
    if (now - last_log >= kLogInterval) {
      DLOG_WARN("ExpertDispatcher::", caller,
                ": waiting for pending experts. pending ", pending_.load(),
                " enqueue ", enqueue_count_.load(), " fetch_dequeue ",
                fetch_dequeue_count_.load(), " exec_dequeue ",
                exec_dequeue_count_.load(), " output ", output_count_.load(),
                " eviction ", eviction_count_.load(), " no_victim_wait ",
                no_victim_wait_count_.load());
      last_log = now;
    }
    if (now - last_progress >= kStallTimeout) {
      pending_stall_count_.fetch_add(1);
      auto idle_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         now - last_progress)
                         .count();
      std::ostringstream oss;
      oss << "ExpertDispatcher::" << caller
          << " progress stall: pending=" << pending_.load()
          << " enqueue=" << enqueue_count_.load()
          << " fetch_dequeue=" << fetch_dequeue_count_.load()
          << " exec_dequeue=" << exec_dequeue_count_.load()
          << " output=" << output_count_.load()
          << " eviction=" << eviction_count_.load()
          << " no_victim_wait=" << no_victim_wait_count_.load()
          << " idle_us=" << idle_us;
      throw std::runtime_error(oss.str());
    }
  }

  auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - wait_start)
                     .count();
  auto wait_us_u64 = static_cast<std::uint64_t>(wait_us);
  pending_wait_total_us_.fetch_add(wait_us_u64);
  RecordAtomicMax(pending_wait_max_us_, wait_us_u64);
}

void ExpertDispatcher::SetInputs(const torch::Tensor& hidden_states,
                                 const torch::Tensor& router_mask,
                                 const torch::Tensor& router_weight) {
  int device = at::cuda::current_device();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(CUDA_DEVICE(device));
  hidden_states_ = hidden_states;
  router_mask_ = router_mask;
  router_weight_ = router_weight;  // this can be float32
  final_hidden_states_ = torch::zeros_like(hidden_states, options);
}
