"""DX300 boom parameters, SI units. Single source of truth for the code.

Mirrors 03-Machine/DX300 Boom Parameters (Obsidian). Pin names A/B/C/D and
the frame (origin A, x forward, y up, theta = angle of A->D) are defined there.
Status tags: official (Doosan/Develon document), ESTIMATED (drawing, 4.4 mm/px,
to be replaced by quarry measurements).
"""
import math

# --- Pin coordinates, measured on the drawing at theta_ref (ESTIMATED) ---
THETA_REF_RAD = math.atan2(-0.682, 6.208)    # pose of the drawing, about -6.3 deg
B_XY_M = (0.571, -0.695)                     # cylinder base on frame (fixed)
C_REF_XY_M = (2.629, 0.850)                  # rod eye on boom at theta_ref
D_REF_XY_M = (6.208, -0.682)                 # boom-arm pin at theta_ref

# --- Derived boom constants (do not edit, computed from the pins above) ---
C_M = math.hypot(*C_REF_XY_M)                                    # |AC|, about 2.763 m
DELTA_RAD = THETA_REF_RAD - math.atan2(C_REF_XY_M[1], C_REF_XY_M[0])  # about -24.2 deg
BOOM_LENGTH_M = math.hypot(*D_REF_XY_M)                          # |AD|, about 6.245 m

# --- Boom (official) ---
BOOM_LENGTH_OFFICIAL_M = 6.245   # S, M p.62
BOOM_MASS_KG = 2330.0            # M p.14, p.66

# --- Boom cylinders (official, Develon DX300LCA spec sheet sec. 8) ---
N_CYL = 2
BORE_M = 0.140
ROD_M = 0.095
STROKE_M = 1.450
CLOSED_LENGTH_M = 2.010
EXTENDED_LENGTH_M = CLOSED_LENGTH_M + STROKE_M   # 3.460 m
CYL_MASS_KG = 260.0
