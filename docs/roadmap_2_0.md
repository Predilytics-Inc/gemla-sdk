# GEMLA SDK Roadmap to 2.0.0

This roadmap keeps GEMLA 1.x as the public SDK and introduces stable contracts for the future GEMLA 2.x architecture engine.

## Version ladder

| Version | Milestone | Meaning |
|---|---|---|
| 1.1.0 | Version cleanup and API contracts | Stable public objects, config, exceptions, and docs. |
| 1.2.0 | Γ–EML–α core | Action, Gamma/MAP, and path-measure implementation. |
| 1.3.0 | Surrogate maps | Causal source-to-surrogate layer. |
| 1.4.0 | Formal lifted gates | RevA, flatness, anchors, winding, and acceptance objects. |
| 1.5.0 | Controls and ablations | Expanded adversarial and structure-sensitivity suite. |
| 1.6.0 | ADL/operator layer | Finite operator family and determinant observable. |
| 1.7.0 | Cross-resolution scaling | Multi-P stability runner. |
| 1.8.0 | T4f audit | Eigenvalue tracking, braid words, parity, and monodromy. |
| 1.9.0 | Calibration and bundles | Nulls, bootstrap intervals, p-values, and audit bundles. |
| 2.0.0 | Integrated GEMLA2 engine | Full pipeline and evidence export. |

## 1.1.0 scope

GEMLA 1.1.0 does not change the mathematical behavior of the existing 1.0 pipeline. It stabilizes the SDK surface by adding:

- consistent versioning,
- public data contracts,
- shared configuration,
- shared exceptions,
- API contract documentation,
- tests for the new public objects.

## 2.0.0 completion target

GEMLA 2.0.0 is complete when the SDK can evaluate:

```text
X
→ surrogate map R_P
→ EML action A_P
→ Gibbs/path-measure kernel
→ Gamma/MAP update
→ ADL operator family K_P(t)
→ lifted phase / flatness / anchors / winding
→ controls + ablations
→ cross-resolution stability
→ T4f spectral-braid audit
→ statistical calibration
→ GEMLA2 verdict + audit bundle
```
