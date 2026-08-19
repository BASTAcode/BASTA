import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

MODULES = ["glitch_fq", "glitch_sd", "icov_sd", "sd"]


class F2PyBuildHook(BuildHookInterface):
    PLUGIN_NAME = "f2py-build"

    def initialize(self, version, build_data):
        src_dir = Path(self.root) / "src" / "basta"

        for name in MODULES:
            source = src_dir / f"{name}.f95"
            subprocess.check_call(
                [sys.executable, "-m", "numpy.f2py", "-c", str(source), "-m", name],
                cwd=src_dir,
            )

        # Make sure the compiled extensions get bundled into the wheel
        build_data["artifacts"] = build_data.get("artifacts", []) + [
            f"src/basta/{name}*.so" for name in MODULES
        ]
        # The wheel now contains compiled code, so it's platform-specific
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
