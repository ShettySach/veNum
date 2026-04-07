pub mod dependence;
pub mod extract;
pub mod legality;

pub use dependence::analyze_kernel_poly;
pub use extract::{StatementInstance, extract_instances};
