pub mod access_map;
pub mod domain;
pub mod native;

#[cfg(test)]
mod tests;

pub use access_map::{strides_to_access, AccessMap};
pub use domain::{shape_to_domain, Aff, Constraint, Domain, PolyVar};
pub use native::NativeDependenceAnalyzer;
