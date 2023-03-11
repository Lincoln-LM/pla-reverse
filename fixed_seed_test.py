"""Test fixed seed reversal"""

import numpy as np
import pyopencl as cl
import tqdm
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
import pla_reverse

CONSTANTS = {}

context = cl.create_some_context()
queue = cl.CommandQueue(context)

CONSTANTS["SHINY_ROLLS"] = int(input("Amount of Shiny Rolls: "))
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
CONSTANTS["ABILITY"] = int(input("Ability (0/1): "))
CONSTANTS["GENDER_RATIO"] = int(input("Gender Ratio (255/254/225/191/127/63/31/0): "))
CONSTANTS["GENDER"] = int(input("Gender (M=0, F=1, G=2): "))
CONSTANTS["NATURE"] = int(input("Nature (0-24): "))
CONSTANTS["SIZES"] = input(
    "Sizes (comma seperated, {height, weight}, ex. '{127,128},{200,255},{28,100}'): "
)
sizes_array = tuple(
    tuple(
        int(size_val) for size_val in size.replace("{", "").replace("}", "").split(",")
    )
    for size in CONSTANTS["SIZES"].split("},")
)
CONSTANTS["SIZES_COUNT"] = len(sizes_array)

expected_seeds = pla_reverse.odds.calc_expected_seeds(
    CONSTANTS["GENDER"], CONSTANTS["GENDER_RATIO"], sizes_array
)
print(f"Expecting around {expected_seeds} to be found")

program = cl.Program(
    context, pla_reverse.shaders.build_shader_code("fixed_seed_shader", CONSTANTS)
).build()

host_results = np.zeros(51200, np.uint64)
host_count = np.zeros(1, np.int32)

device_results = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_results.nbytes)
device_count = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_count.nbytes)

kernel = program.find_fixed_seeds
kernel(queue, (32**2, 32**2, 32**2), None, device_count, device_results)

host_results = np.empty_like(host_results)
host_count = np.empty_like(host_count)

cl.enqueue_copy(queue, host_results, device_results)
cl.enqueue_copy(queue, host_count, device_count)

host_results = host_results[: host_count[0]]

print(f"{host_count[0]} total fixed seeds found!")
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
