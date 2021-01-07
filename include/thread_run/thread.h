/**
 * @filename thread.h
 * @brief 线程类 Linux
 * @author devon.hao
 */

#ifndef _BASE_THREAD_H_
#define _BASE_THREAD_H_

#include <vector>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <syslog.h>
#include <sys/prctl.h>

#include "mutex.h"
#include "common.h"

/// @brief 线程类
class Thread
{
public:
    enum StateT
    {
        kInit,
        kStart,
        kJoined,
        kStop
    };

    explicit Thread()
        : _state(kInit), _thread(-1), _cpu_id(-1)
    {
    }

    explicit Thread(const char thread_name[], int cpu_id)
        : _state(kInit), _thread(-1), _cpu_id(cpu_id)
    {
        strcpy(_thread_name, thread_name);
    }

    virtual ~Thread()
    {
        if (isRunning())
        {
            join();
            _state = kStop;
        }
    }

    void bindCore(const char *thread_name, const int coreid)
    {
        if (thread_name != NULL)
            strcpy(_thread_name, thread_name);
        _cpu_id = coreid;
    }

    /// @brief 启动线程
    /// @return 成功返回true，失败返回false
    bool start()
    {
        if (kInit != _state)
        {
            return false;
        }

        pthread_attr_t *pAttr = NULL;
        pthread_attr_t threadattr;
        if (_cpu_id >= 0)
        {            
            pthread_attr_init(&threadattr);
            pAttr = &threadattr;
            cpu_set_t cpu_info;
            CPU_ZERO(&cpu_info);
            CPU_SET(_cpu_id, &cpu_info);
            if (0 != pthread_attr_setaffinity_np(pAttr, sizeof(cpu_set_t), &cpu_info))
            {
                perror("set affinity error:");
                return false;
            }
            printf("thread bind core id:%d \r\n", _cpu_id);
        }

        bool result = false;
        int ret = pthread_create(&_thread, pAttr, threadProc, (void *)this);
        result = (0 == ret);
        _state = kStart;
        return result;
    }

    /// @brief 结束线程
    /// @return 成功返回true，失败返回false
    bool stop()
    {
        if (kStop == _state || kInit == _state)
        {
            return true;
        }

        bool result = true;
        //        if (0 != pthread_cancel(_thread)) {
        //            result = false;
        //        }

        if (isRunning())
        {
            _state = kStop;
            result = join();
        }
        _state = kInit;
        _thread = -1;
        return result;
    }

    bool isRunning()
    {
        return !(_state == kJoined);
    }

    /// @brief 在当前线程中等待该线程结束
    /// @return 成功返回true，失败返回false
    bool join()
    {
        if (_thread == -1)
            return false;
        bool result = false;
        int ret = pthread_join(_thread, NULL);
        if (0 == ret)
        {
            result = true;
        }
        _state = kJoined;
        return result;
    }

    pthread_t tid() const { return _thread; }

    /// @brief 实现线程函数功能函数
    /// @return 成功返回true，失败返回false
    virtual bool run() { return true; }

    static int lock_file(const int fd)
    {
        struct flock flk;
        flk.l_type = F_WRLCK;
        flk.l_start = 0;
        flk.l_whence = SEEK_SET;
        flk.l_len = 0;

        return (fcntl(fd, F_SETLK, &flk));
    }

    static bool checkAppOnly(const std::string &strAppName)
    {
        int fd = 0;
        char buf[16] = {0};

        const int nlockmode = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        const char *lockfile = (std::string("/var/run/") + strAppName + ".pid").c_str();

        fd = open(lockfile, O_RDWR | O_CREAT, nlockmode);

        if (fd < 0)
            return true;

        if (lock_file(fd))
        {
            if (errno == EACCES || errno == EAGAIN)
            {
                close(fd);
                return false;
            }
            return true;
        }

        ftruncate(fd, 0);
        sprintf(buf, "%ld", (long)getpid());
        write(fd, buf, strlen(buf) + 1);
        return true;
    }

private:
    DISALLOW_EVIL_CONSTRUCTORS(Thread);

    void threadLoop()
    {
        if (strlen(_thread_name) > 0)
        {
            prctl(PR_SET_NAME, _thread_name);
        }
        
        while (_state != kStop)
        {

            if (!run())
                break;
        }
    }

    static void *threadProc(void *param)
    {
        Thread *pThis = reinterpret_cast<Thread *>(param);
        if (pThis)
            pThis->threadLoop();
        return 0;
    }

    StateT _state;
    int _cpu_id;
    char _thread_name[256];
    pthread_t _thread;
};

#endif //
