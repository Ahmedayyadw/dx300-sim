# Acceptable error for DX300 boom control

Use case: coarse boom positioning (lifting to a truck, moving into position before a dig), not fine grading; fine grade work is done with the arm and bucket under machine control.
Tolerance: about 10-20 cm at the boom tip is acceptable for coarse positioning. Source: <add one real link, e.g. a Trimble/Leica/Topcon machine-control page>.
Conversion: tip error = 6.245 m (DX300 boom length) x angle error in radians, so 1 deg = 10.9 cm and 2 deg = 21.8 cm.
Choice: w = 0.3 (1.10 deg = 12.0 cm, 32% less energy than w = 0) gives safe margin; w = 0.5 (1.69 deg = 18.4 cm, 37% less energy) is at the edge of the tolerance.
