"""Reverse from fixed seed to generator seed"""


import numpy as np
import pyopencl as cl
import tqdm
import pla_reverse

context = cl.create_some_context()
queue = cl.CommandQueue(context)

input_filename = input("Fixed seed dump filename: ")
if not input_filename.endswith(".npy"):
    input_filename += ".npy"

fixed_seeds = np.load(input_filename)
TOTAL_SEEDS = len(fixed_seeds)
STEP_SIZE = TOTAL_SEEDS // 100
print(f"{TOTAL_SEEDS} total fixed seeds loaded.")

program = cl.Program(
    context, pla_reverse.shaders.build_shader_code("generator_seed_shader", {})
).build()

host_count = np.zeros(1, np.uint64)
host_results = np.zeros(round(len(fixed_seeds) * 1.5), np.uint64)
host_slices = np.zeros(256, np.uint64)
host_seeds = fixed_seeds
for i in range(256):
    for k in range(8):
        if (i >> k) & 1:
            host_slices[i] |= np.uint64(1) << np.uint64(k * 8)

device_count = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_count.nbytes)
device_results = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, host_results.nbytes)
device_slices = cl.Buffer(context, cl.mem_flags.READ_ONLY, host_slices.nbytes)
device_seeds = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_seeds.nbytes)

print("Processing ....")

cl.enqueue_copy(queue, device_count, host_count)
cl.enqueue_copy(queue, device_results, host_results)
cl.enqueue_copy(queue, device_slices, host_slices)
cl.enqueue_copy(queue, device_seeds, host_seeds)

kernel = program.find_generator_seeds
with tqdm.tqdm(total=TOTAL_SEEDS) as progress_bar:
    i = 0
    while i < TOTAL_SEEDS:
        for j in range(min(STEP_SIZE, TOTAL_SEEDS - i)):
            kernel(
                queue,
                (256, 256, 256),
                None,
                device_count,
                device_results,
                device_slices,
                device_seeds,
                np.int32(i + j),
            ).wait()
        progress_bar.update(STEP_SIZE)
        i += STEP_SIZE


host_count = np.empty_like(host_count)
host_results = np.empty_like(host_results)

cl.enqueue_copy(queue, host_results, device_results)
cl.enqueue_copy(queue, host_count, device_count)

host_results = host_results[: host_count[0]]

print(f"{host_count[0]} total generator seeds found!")
if filename := input("Filename to save (empty to not save): "):
    np.save(filename, host_results)
