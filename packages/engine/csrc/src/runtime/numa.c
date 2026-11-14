/* Bind once, before pools are faulted/pinned and workers are created. */
#define _GNU_SOURCE
#include "numa.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static void warn_error(const char *operation, int node) {
    const int saved_errno = errno;
    (void)fprintf(stderr, "ShadowSpill NUMA warning: %s for node %d: %s; "
                  "continuing with OS placement.\n",
                  operation, node, strerror(saved_errno));
    (void)fflush(stderr);
}

/* Read kernel CPU/node range lists without a fixed CPU_SETSIZE limit. */
static cpu_set_t *parse_list(const char *text, size_t *bytes) {
    unsigned long largest = 0U;
    for (const char *p = text; *p != '\0';) {
        if (isspace((unsigned char)*p) || *p == ',' || *p == '-') { ++p; continue; }
        char *end = NULL;
        const unsigned long value = strtoul(p, &end, 10);
        if (end == p || value > INT_MAX / 2U) { return NULL; }
        if (value > largest) { largest = value; }
        p = end;
    }
    /* sched_getaffinity wants space for the kernel's complete mask. */
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    const size_t count = (size_t)(configured > 0 ? configured : 1);
    const size_t bits = largest + 1U > count ? largest + 1U : count;
    *bytes = CPU_ALLOC_SIZE(bits > 1024U ? bits : 1024U);
    cpu_set_t *mask = calloc(1U, *bytes);
    if (mask == NULL) { return NULL; }
    for (const char *p = text; *p != '\0';) {
        if (isspace((unsigned char)*p) || *p == ',') { ++p; continue; }
        char *end = NULL;
        const unsigned long first = strtoul(p, &end, 10);
        unsigned long last = first;
        if (end == p) { free(mask); return NULL; }
        p = end;
        if (*p == '-') {
            last = strtoul(p + 1, &end, 10);
            if (end == p + 1 || last < first) { free(mask); return NULL; }
            p = end;
        }
        for (unsigned long bit = first; bit <= last; ++bit) {
            CPU_SET_S((size_t)bit, *bytes, mask);
        }
    }
    return mask;
}

static cpu_set_t *read_list(const char *path, size_t *bytes) {
    FILE *file = fopen(path, "r");
    if (file == NULL) { return NULL; }
    char *line = NULL;
    size_t capacity = 0U;
    cpu_set_t *mask = NULL;
    if (getline(&line, &capacity, file) >= 0) { mask = parse_list(line, bytes); }
    free(line);
    (void)fclose(file);
    return mask;
}

static int single_node(void) {
    size_t bytes = 0U;
    cpu_set_t *nodes = read_list("/sys/devices/system/node/online", &bytes);
    int result = -1;
    if (nodes != NULL && CPU_COUNT_S(bytes, nodes) == 1) {
        for (size_t bit = 0U; bit < bytes * CHAR_BIT; ++bit) {
            if (CPU_ISSET_S(bit, bytes, nodes)) { result = (int)bit; break; }
        }
    }
    free(nodes);
    return result;
}

/* A cpuset's memory restriction is independent of CPU affinity. */
static int memory_allowed(int node) {
    /* Read the complete permitted list, including memory-only nodes. */
    cpu_set_t *mask = NULL;
    FILE *file = fopen("/proc/self/status", "r");
    if (file == NULL) { return 0; }
    char *line = NULL;
    size_t capacity = 0U;
    int allowed = 0;
    while (getline(&line, &capacity, file) >= 0) {
        const char prefix[] = "Mems_allowed_list:";
        if (strncmp(line, prefix, sizeof(prefix) - 1U) == 0) {
            size_t length = 0U;
            mask = parse_list(line + sizeof(prefix) - 1U, &length);
            allowed = mask != NULL && CPU_ISSET_S((size_t)node, length, mask);
            free(mask);
            break;
        }
    }
    free(line);
    (void)fclose(file);
    return allowed;
}

static void bind_threads(int node) {
    char path[128];
    (void)snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    size_t bytes = 0U;
    cpu_set_t *local = read_list(path, &bytes);
    cpu_set_t *allowed = local == NULL ? NULL : malloc(bytes);
    if (local == NULL || allowed == NULL) {
        warn_error("cannot read local CPU set", node);
        free(allowed); free(local); return;
    }
    DIR *tasks = opendir("/proc/self/task");
    if (tasks == NULL) {
        warn_error("cannot enumerate process threads", node);
        free(allowed); free(local); return;
    }
    unsigned int restricted = 0U, failed = 0U, bound = 0U;
    struct dirent *entry;
    while ((entry = readdir(tasks)) != NULL) {
        char *end = NULL;
        const long value = strtol(entry->d_name, &end, 10);
        if (*end != '\0' || value <= 0 || value > INT_MAX) { continue; }
        const pid_t tid = (pid_t)value;
        if (sched_getaffinity(tid, bytes, allowed) != 0) {
            if (errno != ESRCH) { ++failed; }
            continue;
        }
        CPU_AND_S(bytes, allowed, allowed, local);
        if (CPU_COUNT_S(bytes, allowed) == 0) { ++restricted; continue; }
        if (sched_setaffinity(tid, bytes, allowed) != 0) {
            if (errno != ESRCH) { ++failed; }
        } else { ++bound; }
    }
    (void)closedir(tasks);
    free(allowed); free(local);
    if (restricted != 0U || failed != 0U) {
        (void)fprintf(stderr, "ShadowSpill NUMA warning: node %d CPU binding: "
                      "%u threads bound, %u have no permitted local CPUs, "
                      "%u could not be bound; retaining their existing affinity.\n",
                      node, bound, restricted, failed);
        (void)fflush(stderr);
    }
    if (getenv("SHADOWSPILL_RUNTIME_PROGRESS") != NULL) {
        (void)fprintf(stderr, "shadowspill runtime: NUMA node %d, %u threads bound "
                      "within their existing CPU masks; host memory prefers this node\n",
                      node, bound);
        (void)fflush(stderr);
    }
}

int shadowspill_numa_initialize(const ShadowSpillBackend *backend, int disabled) {
    if (disabled || backend->host_numa_node == NULL) { return -1; }
    int32_t node = -1;
    if (backend->host_numa_node(backend->state, &node) != 0 || node < 0) {
        node = single_node();
    }
    if (node < 0) {
        (void)fprintf(stderr, "ShadowSpill NUMA warning: device-local host node "
                      "could not be discovered; retaining OS CPU/memory placement.\n");
        (void)fflush(stderr);
        return -1;
    }
    bind_threads(node);
    if (!memory_allowed(node)) {
        (void)fprintf(stderr, "ShadowSpill NUMA warning: memory node %d is not "
                      "permitted or its restrictions could not be read; "
                      "retaining OS memory placement.\n", node);
        (void)fflush(stderr);
        return -1;
    }
    size_t bytes = CPU_ALLOC_SIZE((size_t)node + 1U);
    cpu_set_t *mask = calloc(1U, bytes);
    if (mask == NULL) { warn_error("cannot create memory policy", node); return node; }
    CPU_SET_S((size_t)node, bytes, mask);
    /* Linux's nodemask copy subtracts one from maxnode. Include every bit
       in the allocated word (and the API's extra one), including node zero. */
    if (syscall(SYS_set_mempolicy, MPOL_PREFERRED | MPOL_F_STATIC_NODES,
                mask, (unsigned long)(bytes * CHAR_BIT) + 1UL) != 0) {
        warn_error("cannot set initializing thread's preferred memory policy", node);
    }
    free(mask);
    return node;
}

void shadowspill_numa_place(void *address, uint64_t bytes, int node) {
    if (node < 0) { return; }
    /* A distinct VMA name prevents a pool from merging with an adjacent pool,
       allowing the one-time residency audit to account for exactly its pages.
       Older kernels ignore this best-effort name; auditing still checks size. */
#ifndef PR_SET_VMA
#define PR_SET_VMA 0x53564d41
#define PR_SET_VMA_ANON_NAME 0
#endif
    char name[64];
    (void)snprintf(name, sizeof(name), "shadowspill.spill.%p", address);
    (void)prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, address, (size_t)bytes, name);
    const size_t length = CPU_ALLOC_SIZE((size_t)node + 1U);
    cpu_set_t *mask = calloc(1U, length);
    if (mask == NULL) { warn_error("cannot create pool memory policy", node); return; }
    CPU_SET_S((size_t)node, length, mask);
    if (syscall(SYS_mbind, address, (unsigned long)bytes,
                MPOL_PREFERRED | MPOL_F_STATIC_NODES, mask,
                (unsigned long)(length * CHAR_BIT) + 1UL, 0UL) != 0) {
        warn_error("cannot prefer local pages for pinned pool", node);
    }
    free(mask);
}

static void count_placement(
    FILE *file, void *address, uint64_t bytes, int node,
    unsigned long long *total, unsigned long long *remote
) {
    char *line = NULL;
    size_t capacity = 0U;
    const unsigned long long begin = (uintptr_t)address;
    if (file != NULL) {
        while (getline(&line, &capacity, file) >= 0) {
            char *end = NULL;
            const unsigned long long start = strtoull(line, &end, 16);
            if (end == line || start < begin || start >= begin + bytes) { continue; }
            unsigned long long pages = 0U, other = 0U, page_kib = 0U;
            char *context = NULL;
            for (char *word = strtok_r(end, " \n", &context); word != NULL;
                 word = strtok_r(NULL, " \n", &context)) {
                int actual = -1;
                unsigned long long count = 0U;
                if (sscanf(word, "N%d=%llu", &actual, &count) == 2) {
                    pages += count;
                    if (actual != node) { other += count; }
                } else if (sscanf(word, "kernelpagesize_kB=%llu", &count) == 1) {
                    page_kib = count;
                }
            }
            *total += pages * page_kib * 1024U;
            *remote += other * page_kib * 1024U;
        }
    }
    free(line);
}

void shadowspill_numa_verify(void *address, uint64_t bytes, int node) {
    if (node < 0) { return; }
    FILE *file = fopen("/proc/self/numa_maps", "r");
    unsigned long long total = 0U, remote = 0U;
    count_placement(file, address, bytes, node, &total, &remote);
    if (file != NULL) { (void)fclose(file); }
    const long page_size = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_size > 0 ? (uint64_t)page_size : 4096U;
    const uint64_t rounded = (bytes + page - 1U) / page * page;
    if (total != rounded) {
        (void)fprintf(stderr, "ShadowSpill NUMA warning: could not verify all "
                      "%llu bytes of pinned-pool placement near node %d "
                      "(observed %llu); placement may have fallen back.\n",
                      (unsigned long long)bytes, node, total);
    } else if (remote != 0U) {
        (void)fprintf(stderr, "ShadowSpill NUMA warning: pinned pool fell back "
                      "outside preferred node %d: %llu of %llu bytes are remote.\n",
                      node, remote, total);
    } else if (getenv("SHADOWSPILL_RUNTIME_PROGRESS") != NULL) {
        (void)fprintf(stderr, "shadowspill runtime: pinned pool %llu bytes "
                      "verified on NUMA node %d\n", total, node);
    }
    (void)fflush(stderr);
}
#else
int shadowspill_numa_initialize(const ShadowSpillBackend *backend, int disabled) {
    if (!disabled && backend->host_numa_node != NULL) {
        (void)fprintf(stderr, "ShadowSpill NUMA warning: automatic placement is "
                      "unavailable on this OS; retaining OS placement.\n");
        (void)fflush(stderr);
    }
    return -1;
}
void shadowspill_numa_place(void *address, uint64_t bytes, int node) {
    (void)address; (void)bytes; (void)node;
}
void shadowspill_numa_verify(void *address, uint64_t bytes, int node) {
    (void)address; (void)bytes; (void)node;
}
#endif
