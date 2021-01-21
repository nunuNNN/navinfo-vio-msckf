/*
* @filename mutex.h
* @brief 互斥量操作类
*/

#ifndef _BASE_MUTEX_H_
#define _BASE_MUTEX_H_

#include <pthread.h>

#include <thread_run/common.h>

class Mutex
{
public:
    Mutex()
        : _threadId(0)
    {
        pthread_mutex_init(&_mutex, NULL);
    }
    ~Mutex()
    {
                pthread_mutex_destroy(&_mutex);
    }

    void lock()
    {
        pthread_mutex_lock(&_mutex);
        _threadId = static_cast<uint32_t>(pthread_self());
    }

    void unlock()
    {
        _threadId = 0;
        pthread_mutex_unlock(&_mutex);
    }

        pthread_mutex_t *getMutex()
        {
                return &_mutex;
        }

private:
    DISALLOW_EVIL_CONSTRUCTORS(Mutex);

    uint32_t _threadId;
    pthread_mutex_t _mutex;
};

class MutexGuard
{
public:
    MutexGuard(Mutex &mutex)
        : _mutex(mutex)
    {
        _mutex.lock();
    }
    ~MutexGuard()
    {
        _mutex.unlock();
    }
private:
    DISALLOW_EVIL_CONSTRUCTORS(MutexGuard);
    Mutex &_mutex;
};

#define MutexGuard(x) error "Missing MutexGuard object name"

#endif //_BASE_MUTEXLOCK_H_
