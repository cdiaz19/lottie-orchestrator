# Lottie — Development Methodology

> The implementation methodology used throughout previous Lottie versions has been one of
> the project's greatest strengths and **must be preserved in V3**.
>
> Every architectural improvement must follow the same engineering discipline already
> established within the project.

---

## Design First

Before writing implementation code:

- Study the existing architecture.
- Identify extension points.
- Avoid breaking existing abstractions.
- Prefer extending modules over modifying core behavior.
- Preserve backward compatibility whenever possible.

No feature should be implemented by bypassing the current architecture.

---

## Modular Development

Every major feature should be developed as an independent module.

Avoid creating monolithic implementations.

Examples:

```
Runtime
 ├── Event Runtime
 ├── Lifecycle Hooks
 ├── Context Compiler
 ├── Module Orchestrator
 ├── Execution Planner
 ├── Provider Router
 ├── Policy Engine
 ├── Memory Harness
 ├── Plugin SDK
```

Each module should expose clean interfaces and minimize coupling with the rest of the
runtime.

Whenever possible:

- depend on abstractions
- expose interfaces
- avoid circular dependencies
- isolate responsibilities

---

## Incremental Evolution

V3 should evolve through small architectural iterations.

Avoid giant refactors.

Each improvement should:

- compile
- pass tests
- preserve compatibility
- be independently reviewable

One completed module is preferred over five partially implemented features.

---

## Architecture Reviews

Before implementing each Epic:

1. Analyze the existing implementation.
2. Compare it with the V3 vision.
3. Identify gaps.
4. Produce an architectural proposal.
5. Validate the proposal before implementation.

Only after the architecture is approved should implementation begin.

---

## Lottie Lab Validation

Every architectural improvement must include dedicated validation inside Lottie Lab —
**https://github.com/cdiaz19/lottie-lab**.

The implementation is **not complete** until it can be demonstrated through practical
scenarios.

Each feature should include:

- realistic examples
- playground scenarios
- interactive validation
- manual experimentation

Lottie Lab should remain the primary place to validate runtime behavior.

**Hard gate:** one slice = one PR = one lab round. **A slice does not merge until its lab
round is green.** Round numbering is continuous across versions — V1 used R15–R21, V2 uses
R22–R27, V3 starts at R28.

---

## Testing Requirements

Every new module must include comprehensive tests.

Testing is a first-class requirement.

Include:

- Unit Tests
- Integration Tests
- Regression Tests
- Edge Cases
- Failure Scenarios
- Compatibility Tests

If applicable:

- Performance Benchmarks
- Load Tests
- Concurrency Tests

No module should be considered complete without its corresponding test suite.

---

## Multi-Round Development

Every Epic should be developed through multiple engineering rounds.

Suggested workflow:

```
Round 1  Architecture Analysis
   ↓
Round 2  Design Proposal
   ↓
Round 3  Implementation
   ↓
Round 4  Unit Tests
   ↓
Round 5  Integration Tests
   ↓
Round 6  Lottie Lab Validation
   ↓
Round 7  Documentation
   ↓
Round 8  Performance Review
   ↓
Round 9  Final Refactoring
```

Each round should end with a review before moving to the next.

Do not skip validation phases.

---

## Documentation

Every module should include:

- architecture overview
- responsibilities
- public interfaces
- extension points
- lifecycle
- examples
- migration notes (if applicable)

Documentation should evolve alongside the implementation.

---

## Code Quality

Prefer:

- composition over inheritance
- interfaces over concrete implementations
- dependency injection
- immutable data where possible
- explicit contracts
- small cohesive modules

Avoid:

- hidden side effects
- provider-specific logic inside runtime
- business logic inside infrastructure
- duplicated execution paths
- tightly coupled components

---

## Definition of Done

A V3 module is only considered complete when:

- ✓ Architecture reviewed
- ✓ Implementation completed
- ✓ Fully tested
- ✓ Lottie Lab scenarios implemented
- ✓ Documentation updated
- ✓ Performance evaluated
- ✓ Backward compatibility verified
- ✓ Public APIs reviewed
- ✓ Ready for enterprise usage

Feature complete does not mean production ready.

Production ready requires passing the complete engineering workflow above.

---

## Engineering Assistant Responsibilities

The primary engineering assistant's responsibility is not only to generate code, but to
help preserve the architectural integrity of the project.

When implementing any feature:

- Challenge architectural decisions if a cleaner abstraction exists.
- Recommend reusable components instead of one-off solutions.
- Keep the runtime provider-agnostic.
- Keep agents infrastructure-agnostic.
- Prefer long-term maintainability over short-term implementation speed.
- Respect the existing modular architecture of Lottie.
- Preserve consistency with previous versions of the project.

Always think like a Staff/Principal Software Architect, not just a code generator.

Every decision should move Lottie closer to becoming a true Agent Operating System.
