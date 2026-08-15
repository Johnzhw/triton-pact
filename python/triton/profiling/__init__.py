# PACT Online Profiling Framework (PGO branch)
#
# Submodules:
#   pact_profile_db.py       — persistent measured-facts database
#   pact_profile_collector.py — Proton instrumentation / cupti fallback collector
#   pact_kernel_swapper.py   — variant compilation + atomic kernel swap
#   pact_pgo_controller.py   — facts -> numeric pass inputs -> candidate -> swap
#
# Policy: no bottleneck-to-pass-enable mapping.  Passes always run; measured
# facts replace theory-only inputs where available.
