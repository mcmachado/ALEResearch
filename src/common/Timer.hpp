#include <sys/time.h>
#include <time.h>

#ifndef TIMER_H
#define TIMER_H

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

void timeval_print(struct timeval *tv);

#endif
