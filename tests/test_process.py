def test_supervised_child_stops_with_parent(tmp_path):
    """A child started through config.supervised() must not outlive its parent (even on SIGKILL)."""
    import os
    import signal
    import subprocess
    import sys
    import time

    pidfile = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time\n"
        "from app.config import supervised\n"
        "p=subprocess.Popen(supervised(['sleep','300']))\n"
        "time.sleep(1.5)\n"
        "out=subprocess.run(['pgrep','-P',str(p.pid)],capture_output=True,text=True).stdout.split()\n"
        f"open({str(pidfile)!r},'w').write(out[0])\n"
        "time.sleep(300)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=os.getcwd())
    for _ in range(50):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    child = int(pidfile.read_text())
    assert os.path.exists(f"/proc/{child}")
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    for _ in range(40):
        if not os.path.exists(f"/proc/{child}") or open(f"/proc/{child}/stat").read().split()[2] == "Z":
            break
        time.sleep(0.1)
    else:
        os.kill(child, signal.SIGKILL)
        raise AssertionError("child outlived its parent")
