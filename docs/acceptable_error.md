# Acceptable error for DX300 boom control

Use case: coarse boom positioning (lifting to a truck, moving into position before a dig), not fine grading; fine grade work is done with the arm and bucket under machine control.
Tolerance: fine grading with machine control works at centimetre level (GNSS) to millimetre level (laser/total station, ~3-5 mm) [1], far tighter than boom-only control can reach (6-25 cm at the boom tip). This study therefore targets coarse boom positioning, using 20 cm at the boom tip as a design assumption (not a published standard).
Conversion: tip error = 6.245 m (DX300 boom length) x angle error in radians, so 1 deg = 10.9 cm and 2 deg = 21.8 cm.
Choice: w = 0.3 (1.10 deg = 12.0 cm, 32% less energy than w = 0) gives safe margin; w = 0.5 (1.69 deg = 18.4 cm, 37% less energy) is at the edge of the tolerance.

[1] For Construction Pros, ‘Match 3D Grade Control Solution to the Job’, https://www.forconstructionpros.com/construction-technology/machine-grade-control-gps-laser-other/article/20983470/match-3d-grade-control-solution-to-the-job
