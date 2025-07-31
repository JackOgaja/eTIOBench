#pragma once

#include <vector>
#include <functional>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace etiobench {
namespace threading {

/**
 * Threading utilities for parallel processing.
 * Provides OpenMP-style parallel_for and thread pool implementation.
 */

// Get optimal number of threads for current system
size_t get_optimal_thread_count();

// Simple parallel for loop (similar to OpenMP parallel for)
template<typename Func>
void parallel_for(size_t start, size_t end, size_t num_threads, Func&& func) {
    if (start >= end) return;
    
    if (num_threads == 0) {
        num_threads = get_optimal_thread_count();
    }
    
    if (num_threads == 1 || (end - start) < num_threads) {
        // Serial execution for small ranges or single thread
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    size_t chunk_size = (end - start) / num_threads;
    size_t remainder = (end - start) % num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t thread_start = start + t * chunk_size;
        size_t thread_end = thread_start + chunk_size;
        
        // Distribute remainder among first threads
        if (t < remainder) {
            thread_end += 1;
            thread_start += t;
        } else {
            thread_start += remainder;
            thread_end += remainder;
        }
        
        if (thread_start < end) {
            threads.emplace_back([=, &func]() {
                for (size_t i = thread_start; i < std::min(thread_end, end); ++i) {
                    func(i);
                }
            });
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

// Parallel for with automatic thread count detection
template<typename Func>
void parallel_for(size_t start, size_t end, Func&& func) {
    parallel_for(start, end, get_optimal_thread_count(), std::forward<Func>(func));
}

/**
 * Simple thread pool for task-based parallelism
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();
    
    // Submit a task to the thread pool
    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) -> std::future<decltype(func(args...))> {
        using return_type = decltype(func(args...));
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            if (stop_) {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool");
            }
            
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }
    
    // Get number of worker threads
    size_t size() const { return threads_.size(); }
    
    // Check if thread pool is stopped
    bool is_stopped() const { return stop_; }

private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    
    void worker_thread();
};

// Parallel map operation
template<typename Container, typename Func>
auto parallel_map(const Container& container, Func&& func, size_t num_threads = 0) 
    -> std::vector<decltype(func(*container.begin()))> {
    
    using result_type = decltype(func(*container.begin()));
    std::vector<result_type> results(container.size());
    
    parallel_for(0, container.size(), num_threads, [&](size_t i) {
        auto it = container.begin();
        std::advance(it, i);
        results[i] = func(*it);
    });
    
    return results;
}

// Parallel reduce operation
template<typename Container, typename Func, typename T>
T parallel_reduce(const Container& container, T init_value, Func&& func, size_t num_threads = 0) {
    if (container.empty()) return init_value;
    
    if (num_threads == 0) {
        num_threads = get_optimal_thread_count();
    }
    
    std::vector<T> partial_results(num_threads, init_value);
    
    parallel_for(0, container.size(), num_threads, [&](size_t i) {
        size_t thread_id = i % num_threads;
        auto it = container.begin();
        std::advance(it, i);
        partial_results[thread_id] = func(partial_results[thread_id], *it);
    });
    
    T result = init_value;
    for (const auto& partial : partial_results) {
        result = func(result, partial);
    }
    
    return result;
}

} // namespace threading
} // namespace etiobench
