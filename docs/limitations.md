# Limitations 
GEMLA is a research SDK. The current release is intentionally scope-limited.

# Current Limitations
- Uses finite-dimensional observable trajectories.
- Current demos use synthetic or proxy data.
- Current gates are diagnostic, not universal guarantees.
- Real deployment requires domain-specific validation.
- The V-JEPA-style adapter does not download or redistribute third-party model weights.
- The current SDK evaluates transport signatures; it does not make autonomous operational decisions.

## What a PASS means
A PASS means that the supplied trajectory satisfied the current GEMLA gate stack and rejected the included adversarial controls.

## What a PASS does not mean
A PASS does not prove that the input system is safe, optimal, causal in a physical sense, or ready for high-stakes deployment.

## What a FAIL means
A FAIL means one or more gate conditions, control rejection checks, anchor diagnostics, or winding diagnostics did not pass under the current configuration.

## Practical deployment requirements: 
Before deployment, users should add:

- real domain data
- calibrated thresholds
- baseline model comparisons
- out-of-sample validation
- domain expert review
- monitoring and rollback plans

