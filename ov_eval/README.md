# OpenVINS evaluation tools

This package retains the native OpenVINS trajectory and timing utilities. The
project-specific initialization evaluator is deliberately small: it reads the
versioned, segmented `trajectory.tum` produced by this release and evaluates
each reset segment in its own local coordinate frame.

```bash
rosrun ov_eval eval_initialization trajectory.tum groundtruth.tum
rosrun ov_eval eval_initialization trajectory.tum mav0/state_groundtruth_estimate0/data.csv
```

The optional third argument is `posyaw` (the default), `se3`, or `sim3`.
`posyaw` is recommended because full-rotation alignment hides gravity-direction
error. EuRoC ground-truth CSV timestamps are converted from nanoseconds by the
native OpenVINS loader.

The evaluator prints and writes the same compact table: one row per sequence
and a final `Mean` row. By default, `trajectory.tum` produces
`trajectory_init_eval.md` in the same directory. Override the destination
when needed:

```bash
rosrun ov_eval eval_initialization trajectory.tum groundtruth.tum \
  --output /absolute/path/init_eval.md
```

The report file contains only the table; diagnostics remain on stderr.

The result must use schema version 2 and contain these file-level records:

```text
# openvins_result_version 2
# auto_reset_interval_seconds 15.000000000
```

Each sequence starts with a strictly increasing segment ID and its compact
initialization record, followed by ordinary eight-column TUM poses:

```text
# segment 0 reason startup
# init_timing first_attempt_oldest_time 1.0 success_oldest_time 2.5 init_window_time 0.5
# init_linear_state <timestamp tx ty tz qx qy qz qw vx vy vz>
# init_nonlinear_state <timestamp tx ty tz qx qy qz qw vx vy vz>
<timestamp tx ty tz qx qy qz qw>
```

`Time(s)` preserves the earlier initialization evaluator's definition:

```text
success_oldest_time - first_attempt_oldest_time + init_window_time
```

It is a sensor-timestamp span across initialization attempts, not inference or
wall-clock runtime. Missing per-segment metadata, fewer than three poses, or
insufficient ground-truth associations are reported as `N/A`, never zero.
