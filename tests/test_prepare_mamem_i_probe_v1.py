"""Generated archive semantics only, no real dataset reads."""
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("mamem_prepare", SCRIPTS / "prepare_mamem_i_probe_v1.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def listing(path="EEG-SSVEP-Part1/S001a.mat", size=120):
    return ("Type = Rar\nSolid = -\nMultivolume = -\nVolumes = 1\n\n----------\n"
            f"Path = {path}\nFolder = -\nSize = {size}\nEncrypted = -\nSolid = -\n"
            "Split Before = -\nSplit After = -\nHost OS = Win32\nAttributes = A\nCRC = 12345678\n")


@pytest.mark.parametrize("path", ["../S001a.mat", "/root/S001a.mat", "x\\S001a.mat"])
def test_unsafe_member(path):
    with pytest.raises(p.Stop):
        p.parse_listing(listing(path))


def test_solid_rejected():
    with pytest.raises(p.Stop):
        p.parse_listing(listing().replace("Solid = -", "Solid = +"))


def test_first_global_and_no_size_based_replacement():
    def entry(path, size=120):
        _, members = p.parse_listing(listing(path, size))
        return {"archive": {"Path": "generated.rar"}, "members": members}
    entries = [entry("EEG-SSVEP-Part2/S007a.mat"), entry("EEG-SSVEP-Part1/S001a.mat")]
    assert p.select_member(entries)["subject"] == "S001"
    with pytest.raises(p.Stop, match="no_replacement"):
        p.select_member([entry("EEG-SSVEP-Part1/S001a.mat", p.CAP + 1), entries[0]])
    with pytest.raises(p.Stop, match="duplicate"):
        p.select_member([entries[0], entries[0]])


def test_installed_unar_stdout_exact_member_generated_zip(tmp_path):
    archive = tmp_path / "generated.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("folder/chosen.mat", b"generated selected bytes")
        z.writestr("folder/forbidden.mat", b"must not extract")
    r = subprocess.run(["/usr/bin/unar", "-q", "-nr", "-k", "skip", "-o", "-",
                        str(archive), "folder/chosen.mat"], capture_output=True, timeout=10)
    assert r.returncode == 0 and r.stdout == b"generated selected bytes"
    assert list(tmp_path.iterdir()) == [archive]


def test_parent_stdout_cap():
    with pytest.raises(p.Stop, match="stdout_capture_cap"):
        p.bounded_run([sys.executable, "-c", "print('x'*1024)"], 64, 5)


def test_parent_stderr_cap():
    with pytest.raises(p.Stop, match="stderr_capture_cap"):
        p.bounded_run([sys.executable, "-c", "import sys; sys.stderr.write('x'*2000000)"], 64, 5)


def test_parent_wall_timeout():
    with pytest.raises(p.Stop, match="wall_timeout"):
        p.bounded_run([sys.executable, "-c", "import time; time.sleep(5)"], 64, 0.05)
