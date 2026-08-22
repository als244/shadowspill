#include "../internal.h"

int shadowspill_idle_wakeup_initialize(ShadowSpillIdleWakeup *wakeup) {
    if (wakeup == NULL || pthread_mutex_init(&wakeup->lock, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&wakeup->condition, NULL) != 0) {
        pthread_mutex_destroy(&wakeup->lock);
        return -1;
    }
    wakeup->epoch = 0U;
    wakeup->initialized = 1U;
    return 0;
}

void shadowspill_idle_wakeup_destroy(ShadowSpillIdleWakeup *wakeup) {
    if (wakeup == NULL || !wakeup->initialized) {
        return;
    }
    pthread_cond_destroy(&wakeup->condition);
    pthread_mutex_destroy(&wakeup->lock);
    *wakeup = (ShadowSpillIdleWakeup){0};
}

void shadowspill_idle_notify(ShadowSpillRuntime *runtime) {
    if (runtime == NULL || !runtime->idle_wakeup.initialized) {
        return;
    }
    ShadowSpillIdleWakeup *wakeup = &runtime->idle_wakeup;
    pthread_mutex_lock(&wakeup->lock);
    ++wakeup->epoch;
    pthread_cond_broadcast(&wakeup->condition);
    pthread_mutex_unlock(&wakeup->lock);
}
