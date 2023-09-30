inline unsigned long rotl(unsigned long x, const int k) {
  return (x << k) | (x >> (64 - k));
}

inline unsigned long x0_from_x1(unsigned long x1) {
  x1 = rotl(x1, 27);
  return rotl(0x82A2B175229D6A5B, 24) ^ x1 ^ (x1 << 16) ^ rotl(x1, 24);
}

inline unsigned long seed_from_x1(unsigned long x1) {
  return rotl(x1, 27) ^ 0x82A2B175229D6A5B;
}

__kernel void find_generator_seeds(__global uint *cnt, __global ulong *res_g,
                                   __global ulong *slices,
                                   __global ulong *seeds, const int index) {
  unsigned long random = seeds[index];
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
  unsigned int z = get_global_id(2);
  unsigned long x1_slice = slices[x] | (slices[y] << 1) | (slices[z] << 2);
  unsigned long x0_slice =
      x0_from_x1(x1_slice) &
      0b0011100000111000001110000011100000111000001110000011100000111000;
  unsigned long sub = random - x0_slice - x1_slice;
  unsigned long base_x1_slice_3 =
      sub & 0b0000100000001000000010000000100000001000000010000000100000001000;
  unsigned long sub_carry =
      sub - 0b1100000011000000110000001100000011000000110000001100000011000000;
  unsigned long changed = (sub_carry ^ sub) & 0x808080808080808;
  for (unsigned long i0 = 0; i0 <= ((changed >> 11) & 1); i0++) {
    unsigned long part_0 = i0 << 11;
    for (unsigned long i1 = 0; i1 <= ((changed >> 19) & 1); i1++) {
      unsigned long part_1 = part_0 | (i1 << 19);
      for (unsigned long i2 = 0; i2 <= ((changed >> 27) & 1); i2++) {
        unsigned long part_2 = part_1 | (i2 << 27);
        for (unsigned long i3 = 0; i3 <= ((changed >> 35) & 1); i3++) {
          unsigned long part_3 = part_2 | (i3 << 35);
          for (unsigned long i4 = 0; i4 <= ((changed >> 43) & 1); i4++) {
            unsigned long part_4 = part_3 | (i4 << 43);
            for (unsigned long i5 = 0; i5 <= ((changed >> 51) & 1); i5++) {
              unsigned long part_5 = part_4 | (i5 << 51);
              for (unsigned long i6 = 0; i6 <= ((changed >> 59) & 1); i6++) {
                unsigned long part_6 = part_5 | (i6 << 59);
                unsigned long x1_slice_3 = base_x1_slice_3 ^ part_6;
                unsigned long x0_slice_3 =
                    x0_from_x1(x1_slice_3) &
                    0b0100000001000000010000000100000001000000010000000100000001000000;
                unsigned long x0_slice_ = x0_slice | x0_slice_3;
                unsigned long x1_slice_ = x1_slice | x1_slice_3;
                unsigned long x1_slice_4 =
                    (random - x0_slice_ - x1_slice_) &
                    0b0001000000010000000100000001000000010000000100000001000000010000;
                unsigned long x0_slice_4 =
                    x0_from_x1(x1_slice_4) &
                    0b1000000010000000100000001000000010000000100000001000000010000000;
                x0_slice_ = x0_slice_ | x0_slice_4;
                x1_slice_ = x1_slice_ | x1_slice_4;
                unsigned long x1_slice_567 =
                    (random - x0_slice_ - x1_slice_) &
                    0b1110000011100000111000001110000011100000111000001110000011100000;
                unsigned long x0_slice_567 =
                    x0_from_x1(x1_slice_567) &
                    0b0000011100000111000001110000011100000111000001110000011100000111;
                unsigned long x0 = x0_slice_ | x0_slice_567;
                unsigned long x1 = x1_slice_ | x1_slice_567;
                if ((x0 + x1) == random) {
                  unsigned long seed = seed_from_x1(x1);
                  res_g[atomic_inc(&cnt[0])] = seed;
                }
              }
            }
          }
        }
      }
    }
  }
}