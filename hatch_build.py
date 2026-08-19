import re
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

MODULES = ["glitch_fq", "glitch_sd", "icov_sd", "sd"]


def get_gfortran_major_version():
    """Return gfortran's major version number, or None if it can't be determined."""
    try:
        output = subprocess.check_output(
            ["gfortran", "-dumpversion"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    match = re.match(r"(\d+)", output)
    return int(match.group(1)) if match else None


class F2PyBuildHook(BuildHookInterface):
    PLUGIN_NAME = "f2py-build"

    def initialize(self, version, build_data):
        src_dir = Path(self.root) / "src" / "basta"

        # The combination of new systems (newer glibc) and old Fortran code can create issues.
        # --> https://discourse.nixos.org/t/fortran-and-executable-stack/78108
        #     * Background info. The fix do not work for Fortran code.
        # --> https://groups.google.com/a/chromium.org/g/crashpad-dev/c/rqk3YjICT6M
        #     * This is what is implemented.
        #
        # We therefore append the required flag for gfortran 14 and above!
        # --> Note that for systems with old glibc it should work without extra flags
        # --> The combination of gfortran <= 13 and new glibc will *not* work!!
        f90flags = []
        gfortran_version = get_gfortran_major_version()
        if gfortran_version is not None and gfortran_version >= 14:
            f90flags.append("-ftrampoline-impl=heap")

        for name in MODULES:
            source = src_dir / f"{name}.f95"
            cmd = [
                sys.executable, "-m", "numpy.f2py",
                "-c", str(source), "-m", name,
            ]
            if f90flags:
                cmd.append(f"--f90flags={' '.join(f90flags)}")

            subprocess.check_call(cmd, cwd=src_dir)

        build_data["artifacts"] = build_data.get("artifacts", []) + [
            f"src/basta/{name}*.so" for name in MODULES
        ]
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
