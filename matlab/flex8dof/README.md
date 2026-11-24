# DX300 flexible-joint model (4 joints, 8 DOF)

Put this folder on the MATLAB path (or `cd` into it) and run the scripts in order.

| Script | Option | What you should see (checked in Python) |
|---|---|---|
| run_01_boom2dof.m | 1. Boom only, 2 DOF | boom 30 → 35 deg, about 7.1 Hz ripple |
| run_02_flex8dof.m | 2. Full 8 DOF | swing 10 deg, boom 35 deg, spring twist up to about 0.9 deg |
| run_03_rigid_vs_flex.m | 3. Rigid vs flexible | tip differs by up to about 10 cm during the move |
| run_04_modes.m | 4. Natural frequencies | about 1.5 / 3.0 / 5.9 / 25 Hz |

Shared files: dx300_params.m, kin.m, Mmat.m, Gvec.m, hvec.m, dyn_flex.m, dyn_rigid.m.

## Things to discuss with Can
- Boom alone is ~7 Hz, but in the full model it is ~3 Hz, because the boom spring carries the arm and bucket too.
- The modes are almost flat vs pose because k is fixed at mid-stroke. Upgrade: k(q) from V_A(q), V_B(q) using the pin geometry.
- Values marked (PH) in dx300_params.m are placeholders.

## Option 5 - Simscape Multibody (on the laptop)
1. Open your dx300_excavator copy and save it as dx300_excavator_flex.
2. Between each cylinder and its link, add a Translational Spring + Translational Damper (Simscape Foundation) in series with the cylinder rod; set k_lin from dx300_params (before multiplying by arm^2).
3. Swing: add a Rotational Spring + Damper between the swing motor shaft and the upper structure.
4. Run the Lift test and compare the boom angle with run_02_flex8dof.m.

## Option 6 - thesis link (later)
Use the spring twist / oscillation as signals for fault detection (Stage 4) and as an extra cost for the SAC vs PID comparison.
