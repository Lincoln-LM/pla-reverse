__constant uint SEED_MAT[32] = {}; // REPLACE: __constant uint SEED_MAT[32] = {SEED_MAT_REPLACE};
__constant ulong NULL_SPACE[] = {}; // REPLACE: __constant ulong NULL_SPACE[] = {NULL_SPACE_REPLACE};
__constant ulong TARGET_RAND = 0; // REPLACE: __constant ulong TARGET_RAND = TARGET_RAND_REPLACE;
__constant ulong SHINY_ROLLS = 0; // REPLACE: __constant ulong SHINY_ROLLS = SHINY_ROLLS_REPLACE;
__constant uint FIXED_SEED_LOW = 0; // REPLACE: __constant ulong FIXED_SEED_LOW = FIXED_SEED_LOW_REPLACE;
__constant uint PID = 0; // REPLACE: __constant ulong PID = PID_REPLACE;
__constant uint XORO_CONST = 0; // REPLACE: __constant ulong XORO_CONST = XORO_CONST_REPLACE;

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

bool verify(unsigned long seed) {
    struct xoroshiro rng = {seed, 0x82A2B175229D6A5B};
    for (int i = 0; i < SHINY_ROLLS + 1; i++) {
        advance(&rng);
    }

    return (advance(&rng) & 0xFFFFFFFF) == PID;
}

__kernel void find_fixed_seeds(
    __global uint *cnt, __global ulong *res_g)
{
    uint pid_s1 = get_global_id(0);
    uint pid_s0 = (PID - pid_s1) & 0xFFFF;
    uint vec = (pid_s0 | (pid_s1 << 16)) ^ XORO_CONST;

    unsigned long base_seed = 0;
    for (int i = 0; vec; vec >>= 1, i++) {
        if (vec & 1) {
            base_seed ^= SEED_MAT[i];
        }
    }

    for (int i = 0; i < sizeof(NULL_SPACE) / sizeof(ulong); i++) {
        unsigned long seed = FIXED_SEED_LOW | ((ulong)(base_seed ^ NULL_SPACE[i]) << 32);
        if (verify(seed)) {
            res_g[atomic_inc(&cnt[0])] = seed;
        }
    }
}