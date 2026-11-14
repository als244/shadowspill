/* What a separately loaded library offers a runtime, and the one symbol it
   exports to say so. */

#ifndef SHADOWSPILL_RUNTIME_LIBRARY_H
#define SHADOWSPILL_RUNTIME_LIBRARY_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/runtime/lane.h>
#include <shadowspill/runtime/pool_memory.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Extension libraries
 *
 * A pool kind and a lane are contracts, and a library implements either or
 * both. What it hands over is exactly what a caller would otherwise write out
 * by hand: entries for `pool_memory` and `lanes` on the runtime config.
 *
 * THE RUNTIME LOADS NOTHING. It is handed a config whose lists are already
 * filled in, and `libshadowspill` links libc and nothing else. The dlopen
 * lives one layer up, in whatever embeds the runtime -- the same layer that
 * already loads a backend, and by the same means. This header declares the
 * types so that a library, the loader, and the runtime agree on them without
 * any of the three depending on the others.
 *
 * NOTHING RUNS AT LOAD. The exported symbol is a *descriptor*, not a create:
 * it is read, its entries are copied into a config, and no code of the
 * library's has run yet. That is deliberate. A library has no way to report a
 * failure at load -- there is no runtime, no unwind, and nothing to fail back
 * to -- so everything that depends on what the hardware or the network can
 * actually do happens later, in `acquire` or in a lane's `create`, where a
 * failure path and a strict unwind already exist.
 *
 * The descriptor and everything it points at must have static storage: the
 * runtime copies the entries at create, but a loader may read the descriptor
 * at any time while the library is open.
 */

typedef struct ShadowSpillLibraryDescription {
    /* Compared against SHADOWSPILL_ABI_VERSION before anything else is read.
       A library built against a different runtime is refused whole. */
    uint32_t abi_version;

    /* A short name for diagnostics. Borrowed, never freed. */
    const char *name;

    /* Pool kinds this library serves, appended to the runtime config's list.
       NULL with a zero count when it serves none. */
    const ShadowSpillPoolMemoryDescription *pool_memory;
    uint32_t pool_memory_count;

    /* Lanes this library serves, appended likewise. */
    const ShadowSpillLaneDescription *lanes;
    uint32_t lane_count;
} ShadowSpillLibraryDescription;

/* The one symbol an extension library exports. It reads nothing, allocates
   nothing and cannot fail; it returns a pointer to static storage. */
typedef const ShadowSpillLibraryDescription *(*ShadowSpillLibraryDescribe)(void);

#define SHADOWSPILL_LIBRARY_DESCRIBE_SYMBOL "shadowspill_library_describe"

#ifdef __cplusplus
}
#endif

#endif /* SHADOWSPILL_RUNTIME_LIBRARY_H */
