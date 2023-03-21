"""Test generator script for the budew/gastly spawner in crimson mirelands"""

import numpy as np
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from numba_pokemon_prngs.data.personal import PERSONAL_INFO_LA


TABLE = {
    "Day": {
        "Total": 82,
        "Slots": {
            ("Budew", 406): 80,
            ("BudewAlpha", 406): 2,
        },
    },
    "Night": {
        "Total": 82,
        "Slots": {
            ("Gastly", 92): 80,
            ("GastlyAlpha", 92): 2,
        },
    },
}


def generate(group_rng: Xoroshiro128PlusRejection) -> tuple:
    generator_seed = group_rng.next()
    generator_rng = Xoroshiro128PlusRejection(np.uint64(generator_seed))
    slot = (generator_rng.next() / (2**64)) * TABLE[time_of_day]["Total"]
    for species, slot_threshold in TABLE[time_of_day]["Slots"].items():
        if slot < slot_threshold:
            break
        slot -= slot_threshold
    fixed_rng = Xoroshiro128PlusRejection(np.uint64(generator_rng.next()))
    ec = fixed_rng.next_rand(0xFFFFFFFF)
    tidsid = fixed_rng.next_rand(0xFFFFFFFF)
    for _ in range(shiny_rolls):
        pid = fixed_rng.next_rand(0xFFFFFFFF)
        temp = tidsid ^ pid
        shiny = ((temp & 0xFFFF) ^ (temp >> 16)) < 16
        if shiny:
            break
    ivs = tuple(fixed_rng.next_rand(32) for _ in range(6))
    ability = fixed_rng.next_rand(2)
    gender_ratio = PERSONAL_INFO_LA[species[1]].gender_ratio
    gender = None
    if 1 <= gender_ratio <= 253:
        gender = ("Male", "Female")[int((fixed_rng.next_rand(253) + 1) < gender_ratio)]
    nature = fixed_rng.next_rand(25)
    group_rng.next()  # generator 2, unused
    group_rng.re_init(np.uint64(group_rng.next()))
    return shiny, species, ec, tidsid, pid, ivs, ability, gender, nature


seed = int(input("Group Seed (hex): "), 16)
time_of_day = ("Day", "Night")[int(input("Day or Night? (0/1): "))]
shiny_rolls = int(input("Shiny Rolls: "))
go_until_shiny = int(input("Go until first shiny? (0/1): "))
group_rng = Xoroshiro128PlusRejection(np.uint64(seed))
if go_until_shiny:
    advance = -1
    shiny = False
    while not shiny:
        shiny, species, ec, tidsid, pid, ivs, ability, gender, nature = generate(
            group_rng
        )
        advance += 1
    print(
        f"{advance=} {shiny=} {species=} {ec=} {tidsid=} {pid=} {ivs=} {ability=} {gender=} {nature=}"
    )
else:
    max_advances = int(input("Max Advances (will print out all advances): "))
    for advance in range(-2, max_advances):
        if advance == -2:
            print("First seed pokemon:")
        elif advance == -1:
            print("Second seed pokemon:")
        shiny, species, ec, tidsid, pid, ivs, ability, gender, nature = generate(
            group_rng
        )
        print(
            f"{advance=} {shiny=} {species=} {ec=} {tidsid=} {pid=} {ivs=} {ability=} {gender=} {nature=}"
        )
