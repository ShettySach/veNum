pub mod access_map;
pub mod analysis;
pub mod domain;
pub mod native;
pub mod sets;

#[cfg(test)]
mod tests;

pub use access_map::{AccessMap, strides_to_access};
pub use analysis::extract_instances;
#[allow(unused_imports)]
pub use domain::{
    Aff, Constraint, Domain, PolyVar, dim_to_aff, shape_to_domain, shape_to_domain_checked,
};
pub use native::NativeDependenceAnalyzer;
pub use sets::{ConstraintSystem, Relation, project_out};
