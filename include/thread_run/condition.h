/**
* @filename condition.h
* @brief 对条件变量的封装
* @author devon.hao
*/

#ifndef _BASE_CONDITION_H_
#define _BASE_CONDITION_H_

#include <pthread.h>
#include <thread_run/common.h>
#include <thread_run/mutex.h>
#include <sys/time.h>
#include <stdio.h>
#include <thread_run/semaphevent.h>

class Condition
{
public:
    static const uint32_t kInfinite = 0xffffffff;

    Condition(Mutex &mutex)
        : _mutex(mutex)
    {
        pthread_cond_init(&_cond, NULL);
    }
    virtual ~Condition()
    {
        pthread_cond_destroy(&_cond);
    }
    
    bool wait(uint32_t millisecond = kInfinite)
    {
        int32_t ret = 0;
        
        if (kInfinite == millisecond) {
            ret = pthread_cond_wait(&_cond, _mutex.getMutex());
        } else {
            struct timespec ts = {0, 0};
            getAbsTimespec(&ts, millisecond);
            ret = pthread_cond_timedwait(&_cond, _mutex.getMutex(), &ts);
        }
        
        return 0 == ret;
    }
    
    bool notify()
    {
        return 0 == pthread_cond_signal(&_cond);
    }
    
    bool notifyAll()
    {
        return 0 == pthread_cond_broadcast(&_cond);
    }

    static int32_t getAbsTimespec(struct timespec *ts, int32_t millisecond)
    {
        if (NULL == ts)
            return EINVAL;

        struct timeval tv;
        int32_t ret = gettimeofday(&tv, NULL);
        if (0 != ret)
            return ret;

        ts->tv_sec = tv.tv_sec;
        ts->tv_nsec = tv.tv_usec * 1000UL;

        ts->tv_sec += millisecond / 1000UL;
        ts->tv_nsec += millisecond % 1000UL * 1000000UL;

        ts->tv_sec += ts->tv_nsec/(1000UL * 1000UL *1000UL);
        ts->tv_nsec %= (1000UL * 1000UL *1000UL);

        return 0;
    }
    
private:
    DISALLOW_EVIL_CONSTRUCTORS(Condition);
    
    Mutex& _mutex;
    pthread_cond_t _cond;
};


class ConditionExt : public Condition{
public:
    ConditionExt(Mutex &mutex): Condition(mutex){}
    ~ConditionExt(){}

    bool wait(uint32_t millisecond = Condition::kInfinite){

       _sem_mutex.lock();
       bool res = _semaphore.timedWait(50);
       _sem_mutex.unlock();
       if(!res){           
           res = Condition::wait(millisecond);

           if(res){
               _sem_mutex.lock();
               _semaphore.timedWait(50);
               _sem_mutex.unlock();
           }

       }
       return res;
    }

    bool notify(){
        _sem_mutex.lock();
        _semaphore.post();
        _sem_mutex.unlock();
        Condition::notify();

        return true;
    }

private:
    Semaphore    _semaphore;
    Mutex        _sem_mutex;
};

/*条件变量 locker*/
class cond_locker
{
private:
    pthread_mutex_t m_mutex;
    pthread_cond_t m_cond;
    static const uint32_t kInfinite = 0xffffffff;
public:
    // 初始化 m_mutex and m_cond
    cond_locker()
    {
        if(pthread_mutex_init(&m_mutex, NULL) != 0)
            printf("mutex init error");
        if(pthread_cond_init(&m_cond, NULL) != 0)
        {   //条件变量初始化是被，释放初始化成功的mutex
            pthread_mutex_destroy(&m_mutex);
            printf("cond init error");
        }
    }
    // destroy mutex and cond
    ~cond_locker()
    {
        pthread_mutex_destroy(&m_mutex);
        pthread_cond_destroy(&m_cond);
    }
    //等待条件变量
    bool wait(uint32_t millisecond = kInfinite)
    {
        int32_t ret = 0;

        if (kInfinite == millisecond) {
            pthread_mutex_lock(&m_mutex);
            ret = pthread_cond_wait(&m_cond, &m_mutex);
            pthread_mutex_unlock(&m_mutex);
        } else {
            pthread_mutex_lock(&m_mutex);
            struct timespec ts = {0, 0};
            getAbsTimespec(&ts, millisecond);
            ret = pthread_cond_timedwait(&m_cond,&m_mutex, &ts);
            pthread_mutex_unlock(&m_mutex);
        }

        return 0 == ret;
    }
    //唤醒等待条件变量的线程
    bool signal()
    {
        return pthread_cond_signal(&m_cond) == 0;
    }

    //唤醒all等待条件变量的线程
    bool broadcast()
    {
            return pthread_cond_broadcast(&m_cond) == 0;
    }

    static int32_t getAbsTimespec(struct timespec *ts, int32_t millisecond)
      {
          if (NULL == ts)
              return EINVAL;

          struct timeval tv;
          int32_t ret = gettimeofday(&tv, NULL);
          if (0 != ret)
              return ret;

          ts->tv_sec = tv.tv_sec;
          ts->tv_nsec = tv.tv_usec * 1000UL;

          ts->tv_sec += millisecond / 1000UL;
          ts->tv_nsec += millisecond % 1000UL * 1000000UL;

          ts->tv_sec += ts->tv_nsec/(1000UL * 1000UL *1000UL);
          ts->tv_nsec %= (1000UL * 1000UL *1000UL);

          return 0;
      }
};

#endif //_BASE_CONDITION_H_
