"""Submit a packaged .vcc from the Modal Volume.

Uploading from Modal avoids pulling multi-GB .vcc files down to the laptop and
back up again on every iteration.

Needs a Modal Secret named `vcc-token` holding VCC_TOKEN.

Run:  modal run --detach submit.py --vcc-name baseline.vcc -m "message"
"""

import modal

from common import CPU_CORES, WORK, app, cpu_image, VOL, VOL_PATH

secret = modal.Secret.from_name("vcc-token")


@app.function(
    image=cpu_image,
    volumes={VOL_PATH: VOL},
    secrets=[secret],
    cpu=CPU_CORES,
    memory=4096,
    timeout=60 * 60 * 2,
)
def submit(vcc_name: str = "baseline.vcc", message: str = "",
           dry_run: bool = False, final_ok: bool = False, wait: bool = False):
    import os
    import subprocess

    path = f"{WORK}/{vcc_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"{path}: {os.path.getsize(path) / 1e9:.2f} GB")

    # Confirm the token works and report the phase before spending a submission.
    who = subprocess.run(["vcc", "whoami"], capture_output=True, text=True)
    print(who.stdout, who.stderr)
    if who.returncode != 0:
        raise RuntimeError("vcc whoami failed -- check the vcc-token secret")
    if "phase:    final" in who.stdout and not final_ok:
        # Final-phase submissions are capped at 2/day and silently replace the
        # team's prize entry, so never let an automated path spend one by
        # accident. --final-ok is the deliberate opt-in.
        raise RuntimeError(
            "Refusing to auto-submit in the FINAL phase without --final-ok. "
            "This replaces the team's prize entry and spends 1 of 2 daily "
            "submissions, and the score will be hidden."
        )
    if dry_run:
        print("dry run -- not submitting")
        return None

    # Default to NOT waiting: the upload is the fragile part, and `--wait`
    # keeps the container alive through server-side scoring for no benefit --
    # the entry id is enough to poll `vcc status` afterwards.
    cmd = ["vcc", "submit", path]
    if wait:
        cmd.append("--wait")
    if message:
        cmd += ["-m", message]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"vcc submit failed with code {r.returncode}")
    return r.stdout


@app.local_entrypoint()
def main(vcc_name: str = "baseline.vcc", message: str = "",
         dry_run: bool = False, final_ok: bool = False, wait: bool = False):
    submit.remote(vcc_name, message, dry_run, final_ok, wait)
