#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>


class QueueInterruptedException: public std::runtime_error
{
public:
    QueueInterruptedException(std::string const& message)
        : std::runtime_error(message)
    {}
};


template <class T>
class BlockingQueue {

public:
    BlockingQueue() {

    }

    BlockingQueue(size_t capacity) : _capacity(capacity) {

    }

    ~BlockingQueue() {

    }

    void waitAccept() {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.size() > _capacity && !interrupted) {
            _condition.wait(lock);
        }
        if (interrupted) throw QueueInterruptedException("queue interrupted");
    }

    bool offer(T t) {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_queue.size() > _capacity) {
            return false;
        }
        _queue.push(t);
        _condition.notify_all();
        return true;
    }

    void push(T t) {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.size() > _capacity && !interrupted) {
            _condition.wait(lock);
        }
        if (interrupted) throw QueueInterruptedException("queue interrupted");
        _queue.push(t);
        _condition.notify_all();
    }

    T peek() {
        std::unique_lock<std::mutex> lock(_mutex);
        while(_queue.empty() && !interrupted) {
            _condition.wait(lock);
        }
        if (interrupted && _queue.empty()) {
            throw QueueInterruptedException("queue interrupted");
        }
        return _queue.front();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(_mutex);
        while(_queue.empty() && !interrupted) {
            _condition.wait(lock);
        }
        if (interrupted && _queue.empty()) {
            throw QueueInterruptedException("queue interrupted");
        }
        T val = _queue.front();
        _queue.pop();
         _condition.notify_all();
        return val;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

    void interrupt() {
        std::lock_guard<std::mutex> lock(_mutex);
        interrupted = true;
        _condition.notify_all();
    }

private:
    std::queue<T> _queue;
    mutable std::mutex _mutex;
    std::condition_variable _condition;
    size_t _capacity = 1;
    bool interrupted = false;
};
