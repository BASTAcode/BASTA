# Convert reference figures from the examples to png to be rendered by the documentation
import os
import subprocess
from contextlib import contextmanager

from tqdm import tqdm


@contextmanager
def cd(newdir):
    """Change directory"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


# Settings for imagemagic (change name on Windows)
imname = ["convert"]
settings = ["-density", "96", "-quality", "85"]
imgcmd = imname + settings

# Paths
indir = os.path.abspath("../examples/reference")
outdir = os.path.abspath("figures")

# Transverse all example cases and process all pdf -> png, and store in output dir
with cd(indir):
    cases = next(os.walk("."))[1]
    for case in tqdm(cases):
        os.makedirs(os.path.join(outdir, case), exist_ok=True)
        with cd(os.path.join(indir, case)):
            files = next(os.walk("."))[2]
            for file in files:
                if not file.endswith(".pdf"):
                    continue
                outname = file[:-4] + ".png"
                outfile = os.path.join(outdir, case, outname)
                imgcall = imgcmd + [f"{file}", f"{outfile}"]
                subprocess.check_call(imgcall)
