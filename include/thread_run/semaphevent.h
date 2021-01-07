/**
* @filename Semaphore.h
* @brief 对信号量的封装
* @author devon.hao
*/

#ifndef _BASE_SEMAPHORE_H_
#define _BASE_SEMAPHORE_H_

#include <pthread.h>
#include <common.h>
#include <mutex.h>
#include<sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <semaphore.h>
#include <mutex.h>

class Semaphore {
public:
    Semaphore()
    {
        // 2st paramater not equal to zero indicate shared in process
        // otherwise shared in all thread in current process
        // 3st paramater is the initialization value of semaphore
        sem_init(&m_sem, 0, 0);
    }
    ~Semaphore() {
        sem_destroy(&m_sem);
    }

    /** * @brief wait resouce number > 0 * @return true:successed, false:don't waited resouce */
    bool wait(){
        return sem_wait(&m_sem) == 0 ? true : false;
    }

    /** * @brief try wait inc resouce number * @return true:successed, false:don't waited resouce */
    bool tryWait(){
        return sem_trywait(&m_sem) == 0 ? true : false;
    }

    /** * @brief wait for timeout * @parma [in] timeout : millisecond * @return true:successed, false:don't waited resouce */
    bool timedWait(const uint64_t timeout) {
        struct timespec ts;
        if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return false;
        }

        ts.tv_sec += timeout / 1000;
        ts.tv_nsec += (timeout % 1000) * 1000000;
        return sem_timedwait(&m_sem, &ts) == 0 ? true : false; 
    }

    /** * @brief post semaphore, inc resouce number * @return true:successed, false:don't waited resouce */
    bool post() {

        return sem_post(&m_sem) == 0 ? true : false;
    }

private:
    sem_t   m_sem;
};

#endif
