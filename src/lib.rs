//! Weighted least-squares (WLS) constrained control allocator.
//!
//! Solves `min ‖Au − b‖²` subject to `umin ≤ u ≤ umax` using an active-set
//! method with incremental QR updates. Designed for real-time motor mixing on
//! flight controllers: `no_std`, fully stack-allocated, const-generic over
//! problem dimensions.
//!
//! # Regularised (standard WLS)
//!
//! ```no_run
//! use wls_alloc::{setup_a, setup_b, solve, ExitCode, VecN, MatA};
//!
//! // Effectiveness matrix G (6 pseudo-controls × 4 motors)
//! let g: MatA<6, 4> = MatA::zeros(); // replace with real data
//! let wv: VecN<6> = VecN::from_column_slice(&[10.0, 10.0, 10.0, 1.0, 0.5, 0.5]);
//! let mut wu: VecN<4> = VecN::from_column_slice(&[1.0; 4]);
//!
//! // Build LS problem
//! let (a, gamma) = setup_a::<4, 6, 10>(&g, &wv, &mut wu, 2e-9, 4e5);
//! let v: VecN<6> = VecN::zeros();
//! let ud: VecN<4> = VecN::from_column_slice(&[0.5; 4]);
//! let b = setup_b::<4, 6, 10>(&v, &ud, &wv, &wu, gamma);
//!
//! // Solve
//! let umin: VecN<4> = VecN::from_column_slice(&[0.0; 4]);
//! let umax: VecN<4> = VecN::from_column_slice(&[1.0; 4]);
//! let mut us: VecN<4> = VecN::from_column_slice(&[0.5; 4]);
//! let mut ws = [0i8; 4];
//! let stats = solve::<4, 6, 10>(&a, &b, &umin, &umax, &mut us, &mut ws, 100);
//! assert_eq!(stats.exit_code, ExitCode::Success);
//! ```
//!
//! # Unregularised (constrained least-squares)
//!
//! When the regularisation term `γ ‖Wu (u − u_pref)‖²` is not needed, use
//! [`setup_a_unreg`] / [`setup_b_unreg`] with [`solve_cls`]. The coefficient
//! matrix is `NV × NU` instead of `(NV + NU) × NU`, yielding a smaller QR
//! factorisation.
//!
//! ```no_run
//! use wls_alloc::{setup_a_unreg, setup_b_unreg, solve_cls, ExitCode, VecN, MatA};
//!
//! // Square system: 4 pseudo-controls × 4 motors
//! let g: MatA<4, 4> = MatA::zeros(); // replace with real data
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

pub use setup::{setup_a, setup_a_unreg, setup_b, setup_b_unreg};
pub use solver::{solve, solve_cls};
pub use types::{ExitCode, MatA, SolverStats, VecN};
