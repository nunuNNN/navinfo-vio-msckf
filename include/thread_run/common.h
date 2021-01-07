/**
* @filename common.h
* @brief 这里放一些最常用的头文件、宏、函数等，需要C++11的支持才行
* @author devon.hao
*/

#ifndef _BASE_COMMON_H_
#define _BASE_COMMON_H_

#include <stdint.h>
#include <unistd.h>
#include <string>
#include <memory>
#include <functional>

using std::string;
using namespace std::placeholders; // for std::bind

// disallow copy ctor and assign opt
#undef DISALLOW_EVIL_CONSTRUCTORS
#define DISALLOW_EVIL_CONSTRUCTORS(TypeName)    \
    TypeName(const TypeName&);                         \
    void operator=(const TypeName&)

// delete object safe
#define SAFE_DELETE(p)        \
    if (NULL != p) {          \
        delete p;             \
        p = NULL;             \
    }
    
// delete object array safe
#define SAFE_DELETE_ARRAY(p)  \
    if (NULL != p) {          \
        delete []p;           \
        p = NULL;             \
    }

#define SAFE_FREE(p)\
    if (NULL != p) {          \
        free(p);             \
        p = NULL;             \
    }

#define SAFE_CLOSE(p)\
    if (p != -1) {          \
        close(p);             \
        p = -1;             \
    }

#define MIN(a,b) ((a)<(b)) ? (a) : (b)
#define MAX(a,b) ((a)>(b)) ? (a) : (b)

#define DECL_EXPORT     __attribute__((visibility("default")))
#define DECL_IMPORT     __attribute__((visibility("default")))

#endif //
