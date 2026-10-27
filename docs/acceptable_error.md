# Acceptable error for DX300 boom control

Use case: coarse boom positioning (lifting to a truck, moving into position before a dig), not fine grading; fine grade work is done with the arm and bucket under machine control.
Tolerance: fine grading with machine control works at centimetre level (GNSS) to millimetre level (laser/total station, ~3-5 mm) [1], far tighter than boom-only control can reach (6-25 cm at the boom tip). This study therefore targets coarse boom positioning, using 20 cm at the boom tip as a design assumption (not a published standard).
Conversion: tip error = 6.245 m (DX300 boom length) x angle error in radians, so 1 deg = 10.9 cm and 2 deg = 21.8 cm.
Choice: w = 0.1. On 100 unseen scenarios its worst move is 17.2 cm (0% over 20 cm) at ~15% less energy than w = 0. w = 0.3 fails 9.7% of moves (worst 31.7 cm) and w = 0.5 fails 26.7% (worst 47.2 cm), see rl/05_generalization.csv.

[1] For Construction Pros, ‘Match 3D Grade Control Solution to the Job’, https://www.forconstructionpros.com/construction-technology/machine-grade-control-gps-laser-other/article/20983470/match-3d-grade-control-solution-to-the-job
