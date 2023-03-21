__constant unsigned char IVS[6] = {}; // REPLACE: __constant unsigned char IVS[6] = {IVS_REPLACE};
__constant bool TWO_ABILITIES = 0; // REPLACE: __constant bool TWO_ABILITIES = TWO_ABILITIES_REPLACE;
__constant unsigned char ABILITY = 0; // REPLACE: __constant unsigned char ABILITY = ABILITY_REPLACE;
__constant unsigned char GENDER_RATIO = 0; // REPLACE: __constant unsigned char GENDER_RATIO = GENDER_RATIO_REPLACE;
__constant unsigned char GENDER = 0; // REPLACE: __constant unsigned char GENDER = GENDER_REPLACE;
__constant unsigned char NATURE = 0; // REPLACE: __constant unsigned char NATURE = NATURE_REPLACE;
__constant unsigned char SIZES[8192] = {}; // REPLACE: __constant unsigned short SIZES[8192] = {SIZES_REPLACE};
__constant unsigned char SHINY_ROLLS = 0; // REPLACE: __constant unsigned char SHINY_ROLLS = SHINY_ROLLS_REPLACE;
__constant ulong IV_CONST = 0; // REPLACE: __constant ulong IV_CONST = IV_CONST_REPLACE;
__constant ulong SEED_MAT[64] = {}; // REPLACE: __constant ulong SEED_MAT[64] = {SEED_MAT_REPLACE};
__constant ulong NULL_SPACE[16] = {}; // REPLACE: __constant ulong NULL_SPACE[16] = {NULL_SPACE_REPLACE};

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

inline unsigned long rand(struct xoroshiro* rng, const unsigned long maximum, const unsigned long mask) {
    unsigned long random;
    do {
        random = advance(rng) & mask;
    } while (random >= maximum);
    return random;
}

inline unsigned long rand_mask(struct xoroshiro* rng, const unsigned long mask) {
    return advance(rng) & mask;
}

bool verify(unsigned long fixed_seed) {
    struct xoroshiro rng = {fixed_seed, 0x82A2B175229D6A5B};

    for (int i = 0; i < 2 + SHINY_ROLLS + 6; i++) {
        advance(&rng);
    }

    unsigned char ability = rand(&rng, 2, 1);
    if (TWO_ABILITIES && (ability != ABILITY)) {
        return false;
    }

    if (1 <= GENDER_RATIO && GENDER_RATIO <= 253) {
        unsigned char gender_val = rand(&rng, 253, 255);
        unsigned char gender = (gender_val + 1) < GENDER_RATIO;
        if (gender != GENDER) {
            return false;
        }
    }

    unsigned char nature = rand(&rng, 25, 31);
    if (nature != NATURE) {
        return false;
    }

    unsigned short height = rand(&rng, 129, 255) + rand(&rng, 128, 127);
    unsigned short weight = rand(&rng, 129, 255) + rand(&rng, 128, 127);
    unsigned short size = (height << 8) | weight;

    if (((SIZES[size >> 3] >> (size & 7)) & 1) == 0) {
        return false;
    }

    return true;
}

__kernel void find_fixed_seeds(
    __global uint *cnt, __global ulong *res_g)
{
    unsigned int ab = get_global_id(0);
    unsigned int cd = get_global_id(1);
    unsigned int ef = get_global_id(2);

    unsigned long a = ab & 31;
    unsigned long b = ab >> 5;
    unsigned long c = cd & 31;
    unsigned long d = cd >> 5;
    unsigned long e = ef & 31;
    unsigned long f = ef >> 5;

    unsigned long  vec = (
        (((IVS[0] - a) & 31) << 0) |
        (a << 5) |
        (((IVS[1] - b) & 31) << 10) |
        (b << 15) |
        (((IVS[2] - c) & 31) << 20) |
        (c << 25) |
        (((IVS[3] - d) & 31) << 30) |
        (d << 35) |
        (((IVS[4] - e) & 31) << 40) |
        (e << 45) |
        (((IVS[5] - f) & 31) << 50) |
        (f << 55)
    ) ^ IV_CONST;

    unsigned long  base_seed = 0;
    for (int i = 0; vec; vec >>= 1, i++) {
        if (vec & 1) {
            base_seed ^= SEED_MAT[i];
        }
    }

    for (int i = 0; i < 16; i++) {
        unsigned long seed = base_seed ^ NULL_SPACE[i];
        if (verify(seed)) {
            res_g[atomic_inc(&cnt[0])] = seed;
        }
    }
}