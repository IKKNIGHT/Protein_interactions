"""Shared RFdiffusion command-building + streaming-run helpers.

CANONICAL SOURCE: Phase5_RFdiffusion_Runner.ipynb cell 5 and
Phase5_RFdiffusion_Sweep.ipynb cell 4 -- confirmed byte-for-byte identical
(same functions, same docstring-less inline definitions) before this
module existed. Both notebooks' "Cell 3: Smoke test" cells now import from
here instead of each carrying its own copy.

NOT included here: the conda-env-build logic (each notebook's "Cell 2").
Checked line-by-line (per the plan's explicit instruction, since no GPU
execution is possible tonight to verify a merge): the two cells are
genuinely different, not just cosmetically -- Runner downloads and
verifies THREE weight checkpoints (Base_ckpt.pt, Complex_base_ckpt.pt,
ActiveSite_ckpt.pt), Sweep downloads only Complex_base_ckpt.pt and also
removes a stale dgl shim from the kernel python that Runner's cell
doesn't check for. Unifying these would mean either parameterizing away
a real difference in what each notebook actually needs, or silently
changing one notebook's runtime behavior -- neither verifiable without a
GPU. Left as-is in each notebook; only the genuinely identical smoke-test
helpers are centralized here.
"""
import subprocess


def rfd_cmd(pybin, rfd_script, spec, out_prefix, n_designs, weights, start=0):
    """Build the RFdiffusion inference command for one design spec.
    Byte-identical to the formula both notebooks previously inlined."""
    return (f"{pybin} {rfd_script} inference.input_pdb={spec['target_pdb']} inference.output_prefix={out_prefix} "
            f"inference.num_designs={n_designs} inference.design_startnum={start} "
            f"\"contigmap.contigs=['{spec['contig']}']\" "
            f"ppi.hotspot_res=\"[{','.join(spec['hotspots'])}]\" inference.ckpt_override_path={weights} "
            f"denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0")


def run_stream(cmd, log_path, cwd, env):
    """Run `cmd`, streaming stdout/stderr, printing lines matching a few key
    markers, and writing the full log to `log_path`. Byte-identical logic to
    what both notebooks previously inlined -- cwd/env are now explicit
    parameters (previously closed over notebook globals RFD/ENV)."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=cwd, env=env)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if any(k in line for k in ('Error', 'error', 'Traceback', 'Finished design', 'Making design')):
            print('   ', line.rstrip())
    p.wait()
    open(log_path, 'w').writelines(lines)
    return p.returncode, ''.join(lines[-40:])
