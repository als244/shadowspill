#ifndef SHADOWSPILL_PYTORCH_STORAGE_INTERNAL_H
#define SHADOWSPILL_PYTORCH_STORAGE_INTERNAL_H

/*
 * PyTorch storages over runtime leases. objects.c holds the C primitives --
 * validate a CPU view against its lease, acquire objects for a stream, hand
 * one to the caller and take it back -- and the torch operators that wrap
 * them compile beside it when libtorch is found.
 */

#include "../internal.h"

#endif
