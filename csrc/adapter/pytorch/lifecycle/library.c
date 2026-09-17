/* Loading the extension libraries whose pool kinds and lanes a runtime uses. */
#include "internal.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * The same shape as backend.c, for the same reason: the runtime loads nothing,
 * so whatever embeds it does. The difference is what comes back. A backend
 * exports a *create*, because a provider object has driver state to build and
 * a failure to report. A library exports a *descriptor*, because everything it
 * offers is a table of function pointers that nobody has called yet -- so
 * there is nothing to fail at, and every hardware- or network-dependent
 * decision waits for a pool's `acquire` or a lane's `create`, where a failure
 * path and an unwind exist.
 *
 * A loaded library must outlive the runtime that uses its entries: a pool's
 * `release` and a lane's `destroy` are called at close, from code that lives
 * here. So these are held beside the backend and closed after the runtime is.
 */

static ShadowSpillStatus load_one(
    const char *path, ShadowSpillPytorchLoadedLibrary *loaded
) {
    void *const library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        /* dlerror() is the only account of why, and it is gone after the next
           call. A bootstrap that fails here otherwise reports one status code
           for a missing file, an unresolved symbol and a wrong architecture
           alike. */
        fprintf(
            stderr, "ShadowSpill: could not load %s: %s\n", path, dlerror()
        );
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    /* The handle is stored before anything can fail, so unload is
       unconditional and never has to guess whether dlopen succeeded. */
    *loaded = (ShadowSpillPytorchLoadedLibrary){.library = library};
    union {
        void *object;
        ShadowSpillLibraryDescribe describe;
    } describe = {
        .object = dlsym(library, SHADOWSPILL_LIBRARY_DESCRIBE_SYMBOL)
    };
    if (describe.object == NULL) {
        fprintf(
            stderr, "ShadowSpill: %s exports no %s\n", path,
            SHADOWSPILL_LIBRARY_DESCRIBE_SYMBOL
        );
        shadowspill_pytorch_libraries_unload(loaded, 1U);
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    loaded->description = describe.describe();
    if (loaded->description == NULL ||
        loaded->description->abi_version != SHADOWSPILL_ABI_VERSION ||
        (loaded->description->pool_memory == NULL &&
         loaded->description->pool_memory_count != 0U) ||
        (loaded->description->lanes == NULL &&
         loaded->description->lane_count != 0U)) {
        fprintf(
            stderr,
            "ShadowSpill: %s describes itself incompatibly (abi %u, this build "
            "is %u)\n",
            path,
            loaded->description == NULL ? 0U : loaded->description->abi_version,
            (unsigned)SHADOWSPILL_ABI_VERSION
        );
        shadowspill_pytorch_libraries_unload(loaded, 1U);
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_pytorch_libraries_load(
    const char *const *paths,
    uint32_t count,
    ShadowSpillPytorchLoadedLibrary **loaded
) {
    *loaded = NULL;
    if (count == 0U) {
        return SHADOWSPILL_STATUS_OK;
    }
    if (paths == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillPytorchLoadedLibrary *libraries = calloc(
        count, sizeof(*libraries)
    );
    if (libraries == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        if (paths[index] == NULL || paths[index][0] == '\0') {
            shadowspill_pytorch_libraries_unload(libraries, index);
            free(libraries);
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        const ShadowSpillStatus status = load_one(
            paths[index], &libraries[index]
        );
        if (status != SHADOWSPILL_STATUS_OK) {
            /* Unwind in reverse over what did load; the failed one already
               closed itself. */
            shadowspill_pytorch_libraries_unload(libraries, index);
            free(libraries);
            return status;
        }
    }
    *loaded = libraries;
    return SHADOWSPILL_STATUS_OK;
}

void shadowspill_pytorch_libraries_unload(
    ShadowSpillPytorchLoadedLibrary *loaded, uint32_t count
) {
    if (loaded == NULL) {
        return;
    }
    for (uint32_t index = count; index > 0U; --index) {
        ShadowSpillPytorchLoadedLibrary *library = &loaded[index - 1U];
        if (library->library != NULL) {
            (void)dlclose(library->library);
        }
        *library = (ShadowSpillPytorchLoadedLibrary){0};
    }
}

/*
 * Flatten what the loaded libraries offer into the two lists the runtime
 * config wants. The runtime copies these at create, but `acquire`, `release`,
 * `create` and `destroy` are called for as long as the runtime lives, so the
 * *libraries* must stay open -- only these two arrays are transient.
 */
ShadowSpillStatus shadowspill_pytorch_library_entries(
    const ShadowSpillPytorchLoadedLibrary *loaded,
    uint32_t count,
    ShadowSpillPoolMemoryDescription **pool_memory,
    uint32_t *pool_memory_count,
    ShadowSpillLaneDescription **lanes,
    uint32_t *lane_count
) {
    *pool_memory = NULL;
    *lanes = NULL;
    *pool_memory_count = 0U;
    *lane_count = 0U;
    uint32_t memory_total = 0U;
    uint32_t lane_total = 0U;
    for (uint32_t index = 0U; index < count; ++index) {
        memory_total += loaded[index].description->pool_memory_count;
        lane_total += loaded[index].description->lane_count;
    }
    if (memory_total == 0U && lane_total == 0U) {
        return SHADOWSPILL_STATUS_OK;
    }
    ShadowSpillPoolMemoryDescription *memory_entries = NULL;
    ShadowSpillLaneDescription *lane_entries = NULL;
    if (memory_total != 0U) {
        memory_entries = calloc(memory_total, sizeof(*memory_entries));
        if (memory_entries == NULL) {
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
    }
    if (lane_total != 0U) {
        lane_entries = calloc(lane_total, sizeof(*lane_entries));
        if (lane_entries == NULL) {
            free(memory_entries);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
    }
    uint32_t memory_next = 0U;
    uint32_t lane_next = 0U;
    for (uint32_t index = 0U; index < count; ++index) {
        const ShadowSpillLibraryDescription *description =
            loaded[index].description;
        for (uint32_t entry = 0U; entry < description->pool_memory_count;
             ++entry) {
            memory_entries[memory_next++] = description->pool_memory[entry];
        }
        for (uint32_t entry = 0U; entry < description->lane_count; ++entry) {
            lane_entries[lane_next++] = description->lanes[entry];
        }
    }
    /* A kind or a pair two libraries both claim is refused by the runtime,
       not here: one rule, in the place that already has to enforce it against
       its own built-ins. */
    *pool_memory = memory_entries;
    *pool_memory_count = memory_total;
    *lanes = lane_entries;
    *lane_count = lane_total;
    return SHADOWSPILL_STATUS_OK;
}
