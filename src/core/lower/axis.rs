use anyhow::{Result, anyhow};

use crate::core::llir::loop_nest::Loop;

pub fn validate_axis(loops: &[Loop], axis: usize) -> Result<()> {
    if axis >= loops.len() {
        return Err(anyhow!(
            "opt axis {} out of bounds for loop nest of size {}",
            axis,
            loops.len()
        ));
    }
    Ok(())
}
