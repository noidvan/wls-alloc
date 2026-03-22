//! Weighted least-squares (WLS) constrained control allocator.
//!
//! Solves `min ‖Au − b‖²` subject to `umin ≤ u ≤ umax` using an active-set
//! method with incremental QR updates. Designed for real-time motor mixing on
//! flight controllers: `no_std`, fully stack-allocated, const-generic over
//! problem dimensions.
//!
//! # Quick start
//!
//! ```ignore
//! use wls_alloc::{setup_a, setup_b, solve};
//!
//! // Build LS problem from control-allocation parameters
//! let (a, gamma) = setup_a::<4, 6, 10>(&g, &wv, &mut wu, theta, cond_bound);
//! let b = setup_b::<4, 6, 10>(&v, &ud, &wv, &wu, gamma);
//!
//! // Solve
//! let stats = solve::<4, 6, 10>(&a, &b, &umin, &umax, &mut us, &mut ws, 100);
//! ```

#![no_std]
#![warn(missing_docs)]

/// Low-level linear algebra: Householder QR, back-substitution, constraint checking.
pub mod linalg;
/// Problem setup: convert WLS control-allocation parameters into LS form.
pub mod setup;
/// Active-set constrained least-squares solver.
pub mod solver;
pub mod types;

pub use setup::{setup_a, setup_b};
pub use solver::solve;
pub use types::{ExitCode, MatA, SolverStats, VecN};
