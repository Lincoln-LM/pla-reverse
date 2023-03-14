"""Reverse from fixed seed to generator seed"""


import numpy as np
import pyopencl as cl
import pla_reverse

context = cl.create_some_context()
queue = cl.CommandQueue(context)

input_filename = input("Generator seed dump filename: ")
if not input_filename.endswith(".npy"):
    input_filename += ".npy"

generator_seeds = np.load(input_filename)
print(f"{len(generator_seeds)} total generator seeds loaded.")

input_filename = input("Second fixed seed dump filename: ")
if not input_filename.endswith(".npy"):
    input_filename += ".npy"

fixed_seeds = np.sort(np.load(input_filename))
print(f"{len(fixed_seeds)} total fixed seeds loaded.")

program = cl.Program(
    context, pla_reverse.shaders.build_shader_code("group_seed_shader", {})
).build()

host_results = np.zeros(1, np.uint64)
host_generator_seeds = generator_seeds
host_fixed_seeds = fixed_seeds

device_results = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, host_results.nbytes)
device_generator_seeds = cl.Buffer(
    context, cl.mem_flags.READ_ONLY, host_generator_seeds.nbytes
)
device_fixed_seeds = cl.Buffer(context, cl.mem_flags.READ_ONLY, host_fixed_seeds.nbytes)

print("Processing ....")

cl.enqueue_copy(queue, device_results, host_results)
cl.enqueue_copy(queue, device_generator_seeds, host_generator_seeds)
cl.enqueue_copy(queue, device_fixed_seeds, host_fixed_seeds)

kernel = program.find_group_seed
kernel(
    queue,
    (len(host_generator_seeds),),
    None,
    device_results,
    device_generator_seeds,
    device_fixed_seeds,
    np.int32(len(host_fixed_seeds)),
)

host_results = np.empty_like(host_results)

cl.enqueue_copy(queue, host_results, device_results)

group_seed = host_results[0]

print(f"{group_seed=:016X}")
