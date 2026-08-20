import re
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# We will compile these Fortran modules
MODULES = ["glitch_fq", "glitch_sd", "icov_sd", "sd"]


def get_gfortran_major_version():
    """
    Return gfortran's major version number, or None if it can't be determined.
    """
    try:
        output = subprocess.check_output(
            ["gfortran", "-dumpversion"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    match = re.match(r"(\d+)", output)
    return int(match.group(1)) if match else None


# This defines the hook to be triggered in hatchling
class F2PyBuildHook(BuildHookInterface):
    PLUGIN_NAME = "f2py-build"

    def initialize(self, version, build_data):
        src_dir = Path(self.root) / "src" / "basta"

        # Raise a warning if gfortran is not present but allow BASTA to be installed
        if shutil.which("gfortran") is None:
            self.app.display_warning(
                "gfortran not found on this system. Skipping compilation of "
                f"Fortran extension modules ({', '.join(MODULES)}). "
                "BASTA will install and run, but the glitch fitting relying on "
                "these modules will be unavailable. Install gfortran and "
                "reinstall BASTA if you need to fit glitches."
            )
            return

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

        # Loop the loop!
        built = []
        for name in MODULES:
            source = src_dir / f"{name}.f95"
            cmd = [
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                str(source),
                "-m",
                name,
            ]
            if f90flags:
                cmd.append(f"--f90flags={' '.join(f90flags)}")

            try:
                subprocess.check_call(cmd, cwd=src_dir)
                built.append(name)
            except subprocess.CalledProcessError:
                self.app.display_warning(
                    f"Failed to compile Fortran module '{name}'. "
                    "Features relying on it will be unavailable."
                )

        if not built:
            self.app.display_warning(
                "No Fortran extension modules were built. BASTA will install "
                "and run, but glitch fitting will be unavailable."
            )
            return

        # Finalise and then we are done
        build_data["artifacts"] = build_data.get("artifacts", []) + [
            f"src/basta/{name}*.so" for name in built
        ]
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
