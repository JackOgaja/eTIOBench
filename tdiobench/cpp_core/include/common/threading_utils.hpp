#pragma once

#include <functional>
#include <cstddef>
#include <vector>
#include <thread>
#include <future>

namespace etiobench {
namespace core {
namespace threading {

/**
 * @brief Threading utilities for parallel processing
 */

/**
 * @brief Get optimal number of threads for current system
 * @return Number of hardware threads
 */
size_t get_optimal_thread_count();

/**
 * @brief Parallel for loop implementation
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to execute for each index
 * @param num_threads Number of threads to use (0 = auto)
 */
template<typename Func>
void parallel_for(size_t start, size_t end, Func&& func, size_t num_threads = 0);

/**
 * @brief Parallel for loop with chunk size
 * @param start Start index
 * @param end End index  
 * @param chunk_size Size of each chunk
 * @param func Function to execute for each chunk
 * @param num_threads Number of threads to use (0 = auto)
 */
template<typename Func>
void parallel_for_chunked(size_t start, size_t end, size_t chunk_size, 
                         Func&& func, size_t num_threads = 0);

/**
 * @brief Simple thread pool for task execution
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();
    
    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) -> std::future<decltype(func(args...))>;
    
    void wait_all();
    size_t get_thread_count() const { return threads_.size(); }
    
private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
    
    void worker_thread();
};

// Template implementations

template<typename Func>
void parallel_for(size_t start, size_t end, Func&& func, size_t num_threads) {
    if (end <= start) return;
    
    if (num_threads == 0) {
        num_threads = get_optimal_thread_count();
    }
    
    if (num_threads == 1 || (end - start) < num_threads) {
        // Single-threaded execution
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    // Multi-threaded execution
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    size_t chunk_size = (end - start) / num_threads;
    size_t remainder = (end - start) % num_threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t thread_start = start + t * chunk_size;
        size_t thread_end = thread_start + chunk_size;
        
        // Distribute remainder among first threads
        if (t < remainder) {
            thread_end++;
        }
        
        threads.emplace_back([thread_start, thread_end, &func]() {
            for (size_t i = thread_start; i < thread_end; ++i) {
                func(i);
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

template<typename Func>
void parallel_for_chunked(size_t start, size_t end, size_t chunk_size, 
                         Func&& func, size_t num_threads) {
    if (end <= start || chunk_size == 0) return;
    
    if (num_threads == 0) {
        num_threads = get_optimal_thread_count();
    }
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    std::atomic<size_t> next_chunk{start};
    
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&next_chunk, end, chunk_size, &func]() {
            while (true) {
                size_t chunk_start = next_chunk.fetch_add(chunk_size);
                if (chunk_start >= end) break;
                
                size_t chunk_end = std::min(chunk_start + chunk_size, end);
                func(chunk_start, chunk_end);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

template<typename Func, typename... Args>
auto ThreadPool::submit(Func&& func, Args&&... args) -> std::future<decltype(func(args...))> {
    using return_type = decltype(func(args...));
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
    );
    
    std::future<return_type> result = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Cannot submit task to stopped thread pool");
        }
        tasks_.emplace([task](){ (*task)(); });
    }
    
    condition_.notify_one();
    return result;
}

} // namespace threading
} // namespace core
} // namespace etiobench
