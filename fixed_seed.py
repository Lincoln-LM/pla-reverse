"""Reverse from data to fixed seed"""


import numpy as np
import pyopencl as cl
import tqdm
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from numba_pokemon_prngs.data.personal import PERSONAL_INFO_LA
import pla_reverse

CONSTANTS = {}

context = cl.create_some_context()
queue = cl.CommandQueue(context)

CONSTANTS["SHINY_ROLLS"] = int(input("Amount of Shiny Rolls: "))
dex_number = int(input("National Dex Number: "))
personal_info = PERSONAL_INFO_LA[dex_number]
CONSTANTS["IV_CONST"] = pla_reverse.matrix.vec_to_int(
    pla_reverse.matrix.iv_const(CONSTANTS["SHINY_ROLLS"])
)
CONSTANTS["SEED_MAT"] = ",".join(
    str(pla_reverse.matrix.vec_to_int(row))
    for row in pla_reverse.matrix.generalized_inverse(
        pla_reverse.matrix.iv_matrix(CONSTANTS["SHINY_ROLLS"])
    )
)
CONSTANTS["NULL_SPACE"] = ",".join(
    str(pla_reverse.matrix.vec_to_int(row))
    for row in pla_reverse.matrix.nullspace(
        pla_reverse.matrix.iv_matrix(CONSTANTS["SHINY_ROLLS"])
    )
)
CONSTANTS["IVS"] = input("IVs (comma seperated, ex. '31,31,31,31,31,31'): ")
ivs_array = tuple(int(iv) for iv in CONSTANTS["IVS"].split(","))
two_abilities_bool = bool(int(input("Has two unique non-hidden abilities? (0/1): ")))
CONSTANTS["TWO_ABILITIES"] = "true" if two_abilities_bool else "false"
if two_abilities_bool:
    CONSTANTS["ABILITY"] = int(input("Ability (0/1): "))
else:
    CONSTANTS["ABILITY"] = 2
CONSTANTS["GENDER_RATIO"] = personal_info.gender_ratio
if 1 <= CONSTANTS["GENDER_RATIO"] <= 254:
    CONSTANTS["GENDER"] = int(input("Gender (M=0, F=1): "))
else:
    CONSTANTS["GENDER"] = 2
CONSTANTS["NATURE"] = int(input("Nature (0-24): "))
imperial = bool(int(input("Are your sizes in imperial units? (0/1): ")))
if imperial:
    height = input("Height (ex 5'2): ").replace('"', "").split("'")
    height = int(height[0]), int(height[1])
    weight = float(input("Weight (ex. 3.8): "))
else:
    height = input("Height (ex 0.92): ")
    height = float(height)
    weight = float(input("Weight (ex. 3.80): "))
sizes_array = pla_reverse.size.all_possible_sizes(dex_number, height, weight, imperial)
sizes_set = {pla_reverse.size.scalars_to_ushort(*size) for size in sizes_array}
sizes_table = tuple((1 if i in sizes_set else 0) for i in range(0x10000))
CONSTANTS["SIZES"] = str(sizes_table)[1:-1]
print(f"{len(sizes_set)} possible sizes.")

expected_seeds = pla_reverse.odds.calc_expected_seeds(
    CONSTANTS["TWO_ABILITIES"],
    CONSTANTS["GENDER"],
    CONSTANTS["GENDER_RATIO"],
    sizes_array,
)
print(f"Expecting around {expected_seeds} to be found.")

program = cl.Program(
    context, pla_reverse.shaders.build_shader_code("fixed_seed_shader", CONSTANTS)
).build()

host_results = np.zeros(round(expected_seeds * 1.5), np.uint64)
host_count = np.zeros(1, np.int32)

device_results = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_results.nbytes)
device_count = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_count.nbytes)

kernel = program.find_fixed_seeds
kernel(
    queue,
    (32**2, 32**2, 32**2),
    (4, 4, 4),
    device_count,
    device_results,
)


host_results = np.empty_like(host_results)
host_count = np.empty_like(host_count)

print("Processing ....")

cl.enqueue_copy(queue, host_results, device_results)
cl.enqueue_copy(queue, host_count, device_count)

host_results = host_results[: host_count[0]]

print(f"{host_count[0]} total fixed seeds found!")
verify = bool(int(input("Verify all fixed seeds on cpu (slow)? (0/1): ")))
if verify:
    print("Verifying all fixed seeds ...")

    for fixed_seed in tqdm.tqdm(host_results):
        rng = Xoroshiro128PlusRejection(fixed_seed)
        rng.advance(2 + CONSTANTS["SHINY_ROLLS"])
        ivs = tuple(rng.next_rand(32) for _ in range(6))
        ability = rng.next_rand(2)
        if 1 <= CONSTANTS["GENDER_RATIO"] <= 253:
            gender_val = rng.next_rand(253)
            gender = (gender_val + 1) < CONSTANTS["GENDER_RATIO"]
        nature = rng.next_rand(25)
        height = rng.next_rand(0x81) + rng.next_rand(0x80)
        weight = rng.next_rand(0x81) + rng.next_rand(0x80)
        assert ivs == ivs_array, f"IVs were wrong! {ivs} {ivs_array}"
        assert (
            ability == CONSTANTS["ABILITY"]
        ), f"Ability was wrong! {ability} {CONSTANTS['ABILITY']}"
        if 1 <= CONSTANTS["GENDER_RATIO"] <= 253:
            assert (
                gender == CONSTANTS["GENDER"]
            ), f"Gender was wrong! {gender} {CONSTANTS['GENDER']}"
        assert (
            nature == CONSTANTS["NATURE"]
        ), f"Nature was wrong! {nature} {CONSTANTS['NATURE']}"
        assert (
            height,
            weight,
        ) in sizes_array, f"Height/Weight was wrong! {height} {weight} {sizes_array}"

    print("All fixed seeds found were valid!")
if filename := input("Filename to save (empty to not save): "):
    np.save(filename, host_results)
