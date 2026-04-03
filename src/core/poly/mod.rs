pub mod access_map;
pub mod domain;
pub mod native;

#[cfg(test)]
mod tests;

pub use access_map::{AccessMap, strides_to_access};
pub use domain::{Aff, Constraint, Domain, PolyVar, shape_to_domain};
pub use native::NativeDependenceAnalyzer;
