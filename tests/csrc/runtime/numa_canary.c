/* CPU-only checks; all affinity changes live in this short-lived process. */
#include "../../../csrc/src/runtime/numa.c"
#undef NDEBUG
#include <assert.h>

static int queried;
static int node_query(void *state, int32_t *node) {
    (void)state;
    ++queried;
    *node = 0;
    return 0;
}

int main(void) {
    const ShadowSpillBackend backend = {.host_numa_node = node_query};
    assert(shadowspill_numa_initialize(&backend, 1) == -1);
    assert(queried == 0);
    const ShadowSpillBackend simulated = {0};
    assert(shadowspill_numa_initialize(&simulated, 0) == -1);
#if defined(__linux__)
    char residency[] =
        "1000 prefer=static:0 anon=4 N0=1 N1=3 kernelpagesize_kB=4\n"
        "5000 prefer=static:0 anon=2 N0=2 kernelpagesize_kB=4\n"
        "7000 default anon=7 N1=7 kernelpagesize_kB=4\n";
    FILE *pages = fmemopen(residency, sizeof(residency) - 1U, "r");
    assert(pages != NULL);
    unsigned long long total = 0U, remote = 0U;
    count_placement(pages, (void *)(uintptr_t)0x1000, 6U * 4096U, 0, &total, &remote);
    assert(total == 6U * 4096U && remote == 3U * 4096U);
    (void)fclose(pages);
    total = remote = 0U;
    count_placement(NULL, NULL, 4096U, 0, &total, &remote);
    assert(total == 0U && remote == 0U);
    size_t bytes = 0U;
    cpu_set_t *mask = parse_list("0-3,8,24-27,2048-2050\n", &bytes);
    assert(mask != NULL);
    assert(CPU_COUNT_S(bytes, mask) == 12);
    assert(CPU_ISSET_S(2050, bytes, mask));
    assert(!CPU_ISSET_S(2047, bytes, mask));
    free(mask);
    assert(parse_list("0-3,garbage", &bytes) == NULL);
    assert(parse_list("8-2", &bytes) == NULL);
    /* Even a node with all the process's CPUs must not widen a stricter
       preexisting mask on an individual thread. */
    const long cpus = sysconf(_SC_NPROCESSORS_CONF);
    bytes = CPU_ALLOC_SIZE(cpus > 1024 ? (size_t)cpus : 1024U);
    mask = calloc(1U, bytes);
    assert(mask != NULL && sched_getaffinity(0, bytes, mask) == 0);
    size_t cpu = 0U;
    while (!CPU_ISSET_S(cpu, bytes, mask)) { ++cpu; }
    CPU_ZERO_S(bytes, mask);
    CPU_SET_S(cpu, bytes, mask);
    assert(sched_setaffinity(0, bytes, mask) == 0);
    /* Bind to whatever node owns this CPU; this also exercises thread
       enumeration and a CPU mask containing a single permitted CPU. */
    char directory[128];
    (void)snprintf(directory, sizeof(directory), "/sys/devices/system/cpu/cpu%zu", cpu);
    DIR *entries = opendir(directory);
    assert(entries != NULL);
    struct dirent *entry;
    int node = -1;
    while ((entry = readdir(entries)) != NULL) {
        if (sscanf(entry->d_name, "node%d", &node) == 1) { break; }
    }
    (void)closedir(entries);
    if (node >= 0) {
        bind_threads(node);
        assert(sched_getaffinity(0, bytes, mask) == 0);
        assert(CPU_COUNT_S(bytes, mask) == 1 && CPU_ISSET_S(cpu, bytes, mask));
    }
    free(mask);
#endif
    (void)puts("NUMA discovery opt-out, masks, and scheduler restriction checks passed");
    return 0;
}
