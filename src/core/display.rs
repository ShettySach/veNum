use prettytable::{
    format::consts::FORMAT_BOX_CHARS,
    {Cell, Row, Table},
};
use std::fmt::{Debug, Display, Formatter, Result};

use crate::core::shared::{
    dtype::{Buffer, RealizedTensor},
    tensor::{Context, Tensor},
};

// ==================== Tensor<C: Context> ====================

impl<C: Context> Debug for Tensor<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Tensor")
            .field("dtype", &self.dtype())
            .field("dims", &self.rank())
            .field("elems", &self.numel())
            .field("shape", &self.shape())
            .finish()
    }
}

// ==================== RealizedTensor ====================

impl Display for RealizedTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let sizes = self.sizes();
        let n = sizes.len();

        if (1..=8).contains(&n) {
            let table = if n % 2 == 1 {
                let row = realized_odd_dimensions(self.buffer(), sizes, n, 0);
                let table = Table::init(vec![row]);
                set_style(table)
            } else {
                realized_even_dimensions(self.buffer(), sizes, n, 0)
            };

            write!(f, "{}", table)?;
        }

        writeln!(
            f,
            "Tensor {{ dtype: {:?}, dims: {}, elems: {}, shape: {:?} }}",
            self.dtype(),
            n,
            self.numel(),
            sizes,
        )
    }
}

fn format_buffer_element(buffer: &Buffer, index: usize) -> String {
    match buffer {
        Buffer::F32(v) => format!("{}", v[index]),
        Buffer::F64(v) => format!("{}", v[index]),
        Buffer::I32(v) => format!("{}", v[index]),
        Buffer::I64(v) => format!("{}", v[index]),
    }
}

fn realized_odd_dimensions(buffer: &Buffer, sizes: &[usize], n: usize, flat_offset: usize) -> Row {
    let rank = sizes.len();
    let dim = rank - n;
    let size = sizes[dim];

    if n == 1 {
        Row::from((0..size).map(|i| {
            let s = format_buffer_element(buffer, flat_offset + i);
            Cell::new(&s)
        }))
    } else {
        let inner_numel: usize = sizes[dim + 1..].iter().product();
        Row::from((0..size).map(|i| {
            let offset = flat_offset + i * inner_numel;
            realized_even_dimensions(buffer, sizes, n - 1, offset)
        }))
    }
}

fn realized_even_dimensions(
    buffer: &Buffer,
    sizes: &[usize],
    n: usize,
    flat_offset: usize,
) -> Table {
    let rank = sizes.len();
    let dim = rank - n;
    let size = sizes[dim];
    let inner_numel: usize = sizes[dim + 1..].iter().product();

    let rows = (0..size)
        .map(|i| {
            let offset = flat_offset + i * inner_numel;
            realized_odd_dimensions(buffer, sizes, n - 1, offset)
        })
        .collect();

    let table = Table::init(rows);
    set_style(table)
}

// ==================== Shared ====================

fn set_style(mut table: Table) -> Table {
    table.set_format(*FORMAT_BOX_CHARS);
    table
}
