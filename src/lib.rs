//! Weighted least-squares (WLS) constrained control allocator.
//!
//! Solves `min ‖Au − b‖²` subject to `umin ≤ u ≤ umax` using an active-set
//! method with incremental QR updates. Designed for real-time motor mixing on
//! flight controllers: `no_std`, fully stack-allocated, const-generic over
//! problem dimensions.
//!
//! # Encapsulated API (recommended)
//!
//! [`wls::ControlAllocator`] owns the static problem (effectiveness matrix,
//! weights, `γ`) and the warm-start solver state. Build once, then call
//! [`solve`](wls::ControlAllocator::solve) on every control tick.
//!
//! ```no_run
//! use wls_alloc::wls::ControlAllocator;
//! use wls_alloc::{ExitCode, VecN, MatA};
//!
//! // Effectiveness matrix G (6 pseudo-controls × 4 motors)
//! let g: MatA<6, 4> = MatA::zeros(); // replace with real data
//! let wv: VecN<6> = VecN::from_column_slice(&[10.0, 10.0, 10.0, 1.0, 0.5, 0.5]);
//! let wu: VecN<4> = VecN::from_column_slice(&[1.0; 4]);
//!
//! let mut alloc = ControlAllocator::<4, 6, 10>::new(&g, &wv, wu, 2e-9, 4e5);
//!
//! // Per-tick solve
//! let v: VecN<6> = VecN::zeros();
//! let ud: VecN<4> = VecN::from_column_slice(&[0.5; 4]);
//! let umin: VecN<4> = VecN::from_column_slice(&[0.0; 4]);
//! let umax: VecN<4> = VecN::from_column_slice(&[1.0; 4]);
//!
//! let stats = alloc.solve(&v, &ud, &umin, &umax, 100);
//! assert_eq!(stats.exit_code, ExitCode::Success);
//! let u = alloc.solution();
//! ```
//!
//! # Raw building blocks
//!
//! The [`raw`] module exposes the underlying free functions
//! (`setup_a` / `setup_b` / `solve` / `solve_cls`) for advanced use: custom
//! `A` matrices, the unregularised CLS variant, or composing pieces of the
//! pipeline differently.
//!
//! ```no_run
//! use wls_alloc::raw::{setup_a_unreg, setup_b_unreg, solve_cls};
//! use wls_alloc::{ExitCode, VecN, MatA};
//!
//! let g: MatA<4, 4> = MatA::zeros();
//! let wv: VecN<4> = VecN::from_column_slice(&[1.0; 4]);
//!
//! let a = setup_a_unreg::<4, 4>(&g, &wv);
//! let v: VecN<4> = VecN::zeros();
//! let b = setup_b_unreg(&v, &wv);
//!
//! let umin: VecN<4> = VecN::from_column_slice(&[0.0; 4]);
//! let umax: VecN<4> = VecN::from_column_slice(&[1.0; 4]);
//! let mut us: VecN<4> = VecN::from_column_slice(&[0.5; 4]);
//! let mut ws = [0i8; 4];
//! let stats = solve_cls::<4, 4>(&a, &b, &umin, &umax, &mut us, &mut ws, 100);
//! assert_eq!(stats.exit_code, ExitCode::Success);
//! ```

#![no_std]
#![warn(missing_docs)]

/// Low-level linear algebra: Householder QR, back-substitution, constraint checking.
pub mod linalg;
/// Problem setup: convert WLS control-allocation parameters into LS form.
pub mod setup;
/// Active-set constrained least-squares solver.
pub mod solver;
/// Core types, constants, and nalgebra type aliases.
pub mod types;
/// High-level encapsulated WLS control allocator.
pub mod wls;

pub use types::{ExitCode, MatA, SolverStats, VecN};

/// Low-level free-function building blocks. Use [`wls::ControlAllocator`] for
/// typical flight-controller workloads; reach for `raw` when you need a custom
/// `A`, the unregularised CLS solver, or non-standard pipeline composition.
pub mod raw {
    pub use crate::setup::{setup_a, setup_a_unreg, setup_b, setup_b_unreg};
    pub use crate::solver::{solve, solve_cls};
}

pub use setup::{setup_a, setup_a_unreg, setup_b, setup_b_unreg};
pub use solver::{solve, solve_cls};
