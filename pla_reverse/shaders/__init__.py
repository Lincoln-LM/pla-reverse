"""OpenCL Shader Files"""

import importlib.resources as pkg_resources
from .. import shaders

SHADER_NAMES = ("fixed_seed_shader",)
SHADERS = {
    filename: pkg_resources.read_text(shaders, f"{filename}.c")
    for filename in SHADER_NAMES
}


def build_shader_code(name: str, constants: dict) -> str:
    """Build shader code from filename and constants"""
    code = SHADERS[name]
    for constant_name, constant_value in constants.items():
        code = code.replace(f"{{{constant_name}_REPLACE}}", str(constant_value))
    return code
