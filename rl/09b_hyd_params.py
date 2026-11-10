"""Lab 09b hydraulic parameters, SI units. Loaded by rl/lab09_common.py.

Both values are assumptions supplied by the user, not DX300 data. Replace them when the
shop manual and the pump datasheet are available. dx300sim/dx300_params.py is not touched.
"""

# ASSUMED - not from manual
# Typical valve pressure margin for mobile hydraulics; to be replaced from the DX300 shop manual.
DP_VALVE_PA = 20e5      # 20 bar

# ASSUMED - not from manual
# Typical overall efficiency of an axial piston pump; to be replaced from the pump datasheet.
ETA_PUMP = 0.85
