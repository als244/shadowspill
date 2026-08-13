#include "portfolio_internal.h"

#include <stdint.h>
#include <string.h>

typedef struct Sha256 {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    uint32_t used;
} Sha256;

static uint32_t rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32U - count));
}

static void sha256_transform(Sha256 *hash, const uint8_t block[64]) {
    static const uint32_t constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    uint32_t words[64];
    for (uint32_t index = 0U; index < 16U; ++index) {
        uint32_t offset = index * 4U;
        words[index] = ((uint32_t)block[offset] << 24U) |
            ((uint32_t)block[offset + 1U] << 16U) |
            ((uint32_t)block[offset + 2U] << 8U) |
            block[offset + 3U];
    }
    for (uint32_t index = 16U; index < 64U; ++index) {
        uint32_t left = words[index - 15U];
        uint32_t right = words[index - 2U];
        uint32_t small0 = rotate_right(left, 7U) ^ rotate_right(left, 18U) ^
            (left >> 3U);
        uint32_t small1 = rotate_right(right, 17U) ^ rotate_right(right, 19U) ^
            (right >> 10U);
        words[index] = words[index - 16U] + small0 + words[index - 7U] + small1;
    }

    uint32_t a = hash->state[0];
    uint32_t b = hash->state[1];
    uint32_t c = hash->state[2];
    uint32_t d = hash->state[3];
    uint32_t e = hash->state[4];
    uint32_t f = hash->state[5];
    uint32_t g = hash->state[6];
    uint32_t h = hash->state[7];
    for (uint32_t index = 0U; index < 64U; ++index) {
        uint32_t big1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
            rotate_right(e, 25U);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temporary1 = h + big1 + choose + constants[index] + words[index];
        uint32_t big0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
            rotate_right(a, 22U);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = big0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    hash->state[0] += a;
    hash->state[1] += b;
    hash->state[2] += c;
    hash->state[3] += d;
    hash->state[4] += e;
    hash->state[5] += f;
    hash->state[6] += g;
    hash->state[7] += h;
}

static void sha256_init(Sha256 *hash) {
    *hash = (Sha256){
        .state = {
            0x6a09e667U,
            0xbb67ae85U,
            0x3c6ef372U,
            0xa54ff53aU,
            0x510e527fU,
            0x9b05688cU,
            0x1f83d9abU,
            0x5be0cd19U,
        },
    };
}

static void sha256_update(Sha256 *hash, const void *data_value, size_t size) {
    const uint8_t *data = data_value;
    hash->bytes += size;
    while (size != 0U) {
        uint32_t available = 64U - hash->used;
        uint32_t take = size < available ? (uint32_t)size : available;
        memcpy(hash->block + hash->used, data, take);
        hash->used += take;
        data += take;
        size -= take;
        if (hash->used == 64U) {
            sha256_transform(hash, hash->block);
            hash->used = 0U;
        }
    }
}

static void sha256_text(Sha256 *hash, const char *value) {
    sha256_update(hash, value, strlen(value));
}

static void sha256_finish(
    Sha256 *hash,
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES]
) {
    uint64_t bits = hash->bytes * 8U;
    uint8_t marker = 0x80U;
    sha256_update(hash, &marker, 1U);
    uint8_t zero = 0U;
    while (hash->used != 56U) {
        sha256_update(hash, &zero, 1U);
    }
    uint8_t length[8];
    for (uint32_t index = 0U; index < 8U; ++index) {
        length[7U - index] = (uint8_t)(bits >> (index * 8U));
    }
    sha256_update(hash, length, sizeof(length));
    for (uint32_t index = 0U; index < 8U; ++index) {
        digest[index * 4U] = (uint8_t)(hash->state[index] >> 24U);
        digest[index * 4U + 1U] = (uint8_t)(hash->state[index] >> 16U);
        digest[index * 4U + 2U] = (uint8_t)(hash->state[index] >> 8U);
        digest[index * 4U + 3U] = (uint8_t)hash->state[index];
    }
}

static const char *action_name(uint8_t kind) {
    switch (kind) {
        case SHADOWSPILL_MEMORY_RELEASE:
            return "release";
        case SHADOWSPILL_MEMORY_OFFLOAD:
            return "offload";
        case SHADOWSPILL_MEMORY_PREFETCH:
            return "prefetch";
    }
    return "invalid";
}

static const char *location_name(uint8_t location) {
    return location == SHADOWSPILL_MEMORY_DEVICE ? "device" : "host";
}

static void append_json_string(Sha256 *hash, const char *escaped_payload) {
    sha256_text(hash, "\"");
    sha256_text(hash, escaped_payload);
    sha256_text(hash, "\"");
}

static void append_residency(
    Sha256 *hash,
    const ShadowSpillPressureFitContext *context,
    const uint32_t *aliases,
    const uint8_t *locations,
    uint32_t count
) {
    sha256_text(hash, "[");
    for (uint32_t index = 0U; index < count; ++index) {
        if (index != 0U) {
            sha256_text(hash, ",");
        }
        sha256_text(hash, "{\"alias_group_id\":");
        append_json_string(hash, context->alias_json_names[aliases[index]]);
        sha256_text(hash, ",\"location\":\"");
        sha256_text(hash, location_name(locations[index]));
        sha256_text(hash, "\"}");
    }
    sha256_text(hash, "]");
}

void shadowspill_schedule_digest(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillDenseSchedule *schedule,
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES]
) {
    Sha256 hash;
    sha256_init(&hash);
    sha256_text(&hash, "{\"actions\":[");
    for (uint32_t index = 0U; index < schedule->action_count; ++index) {
        if (index != 0U) {
            sha256_text(&hash, ",");
        }
        sha256_text(&hash, "{\"alias_group_id\":");
        append_json_string(
            &hash,
            context->alias_json_names[schedule->action_aliases[index]]
        );
        sha256_text(&hash, ",\"kind\":\"");
        sha256_text(&hash, action_name(schedule->action_kinds[index]));
        sha256_text(&hash, "\",\"trigger_task_id\":");
        append_json_string(
            &hash,
            context->task_json_names[schedule->action_trigger_tasks[index]]
        );
        sha256_text(&hash, "}");
    }
    sha256_text(&hash, "],\"final_residency\":");
    append_residency(
        &hash,
        context,
        schedule->final_aliases,
        schedule->final_locations,
        schedule->final_count
    );
    sha256_text(&hash, ",\"initial_residency\":");
    append_residency(
        &hash,
        context,
        schedule->initial_aliases,
        schedule->initial_locations,
        schedule->initial_count
    );
    sha256_text(&hash, ",\"schema\":\"shadowspill.memory_schedule/v1\"}");
    sha256_finish(&hash, digest);
}
