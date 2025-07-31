#include "threading_utils.hpp"
#include <algorithm>

namespace etiobench {
namespace threading {

size_t get_optimal_thread_count() {
    size_t hw_threads = std::thread::hardware_concurrency();
    return hw_threads > 0 ? hw_threads : 4; // Default to 4 if detection fails
}

ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    if (num_threads == 0) {
        num_threads = get_optimal_thread_count();
    }
    
    threads_.reserve(num_threads);
    
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });
            
            if (stop_ && tasks_.empty()) {
                return;
            }
            
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        
        task();
    }
}

} // namespace threading
} // namespace etiobench
