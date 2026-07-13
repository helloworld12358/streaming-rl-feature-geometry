# Experiment summary

Current status: implementation and local smoke/validation scaffolding are complete. Full remote run is not executed in this environment because no remote server or self-hosted runner credentials are available.

Expected analysis compares final-window accuracy, cumulative reward, GVF TD error, cue decodability, covariance condition number, effective rank, isotropy error, high-order moment errors, and update norms across feature-property conditions.

Known limitations: the recurrent state uses a fixed linear trace-memory fallback, and approximate Gaussian matching is a simple signed-power moment-shaping extension that cannot guarantee a full isotropic Gaussian distribution.
