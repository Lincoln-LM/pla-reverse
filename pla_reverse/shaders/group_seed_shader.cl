struct xoroshiro {
    unsigned long seed_0;
    unsigned long seed_1;
};

inline unsigned long rotl(unsigned long x, const int k)
{
    return (x << k) | (x >> (64 - k));
}

inline unsigned long advance(struct xoroshiro* rng)
{
    const unsigned long rand = rng->seed_0 + rng->seed_1;
    unsigned long seed_1 = rng->seed_1 ^ rng->seed_0;
    const unsigned long seed_0 = rotl(rng->seed_0, 24) ^ seed_1 ^ (seed_1 << 16);
    seed_1 = rotl(seed_1, 37);
    rng->seed_0 = seed_0;
    rng->seed_1 = seed_1;

    return rand;
}

inline bool verify(__global ulong *fixed_seeds, const int fixed_seeds_length, ulong generator_seed) {
    struct xoroshiro rng = {generator_seed - 0x82A2B175229D6A5B, 0x82A2B175229D6A5B};
    advance(&rng); // generator 0 = generator_seed
    advance(&rng); // generator 1 is unused
    rng.seed_0 = advance(&rng); // reseed group
    rng.seed_1 = 0x82A2B175229D6A5B;
    rng.seed_0 = advance(&rng); // seed generator
    rng.seed_1 = 0x82A2B175229D6A5B;
    advance(&rng); // encounter slot
    unsigned long fixed_seed = advance(&rng);

    // binary search
    int left = 0;
    int right = fixed_seeds_length - 1;
    while (left <= right) {
        int middle = (left + right) / 2;
        if (fixed_seed == fixed_seeds[middle]) {
            return true;
        } else if (fixed_seed < fixed_seeds[middle]) {
            right = middle - 1;
        } else {
            left = middle + 1;
        }
    }
    return false;
}

__kernel void find_group_seed(__global ulong *results, __global ulong *generator_seeds, __global ulong *fixed_seeds, const int fixed_seeds_length) {
    unsigned long generator_seed = generator_seeds[get_global_id(0)];
    if (verify(fixed_seeds, fixed_seeds_length, generator_seed)) {
        results[0] = generator_seed - 0x82A2B175229D6A5B;
    }
}