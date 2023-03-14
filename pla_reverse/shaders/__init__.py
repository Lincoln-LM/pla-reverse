"""OpenCL Shader Files"""

import importlib.resources as pkg_resources
from .. import shaders

SHADER_NAMES = ("fixed_seed_shader", "generator_seed_shader", "group_seed_shader")
SHADERS = {
    filename: pkg_resources.read_text(shaders, f"{filename}.cl")
    for filename in SHADER_NAMES
}


def build_shader_code(name: str, constants: dict) -> str:
    """Build shader code from filename and constants"""
    code = SHADERS[name].split("\n")
    for i, line in enumerate(code):
        if line.startswith("__constant"):
            for constant_name, constant_value in constants.items():
                line = line.replace(f"{constant_name}_REPLACE", str(constant_value))
            if "// REPLACE: " in line:
                cl_value = line.split("// REPLACE: ")[1]
                line = cl_value
            code[i] = line
    return "\n".join(code)
