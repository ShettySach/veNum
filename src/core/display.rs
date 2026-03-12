use prettytable::{
    format::consts::FORMAT_BOX_CHARS,
    {Cell, Row, Table},
};
use std::{
    any::type_name,
    fmt::{Debug, Display, Formatter, Result},
};

use crate::{core::eager::ETensor, core::shape::offset_fn};

impl<T: Debug + Copy> Debug for ETensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("Tensor")
            .field("dtype", &type_name::<T>())
            .field("dims", &self.rank())
            .field("elems", &self.numel())
            .field("shape", &self.sizes())
            .finish()
    }
}

impl<T: Display + Debug + Copy> Display for ETensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let n = self.rank();

        if (1..=8).contains(&n) {
            let table = if n % 2 == 1 {
                let row = odd_dimensions(self, n, 0);
                let table = Table::init(vec![row]);
                set_style(table)
            } else {
                even_dimensions(self, n, 0)
            };

            write!(f, "{}", table)?;
        }

        writeln!(f, "{:?}", self)
    }
}

fn odd_dimensions<T>(tensor: &ETensor<T>, n: usize, stride_offset: usize) -> Row
where
    T: Copy + Display,
{
    let dim = tensor.rank() - n;
    let size = tensor.shape.sizes[dim];
    let stride = tensor.shape.strides[dim];

    if n == 1 {
        let offset = tensor.offset() + stride_offset;
        Row::from((0..size).map(|index| {
            let index = offset_fn(stride, index, size) + offset;
            let element = tensor.data[index];
            //let element = format!("{:.2}", element); // TODO: Handle precision without String
            Cell::from(&element)
        }))
    } else {
        Row::from((0..size).map(|index| {
            let offset = offset_fn(stride, index, size) + stride_offset;
            even_dimensions(tensor, n - 1, offset)
        }))
    }
}

fn even_dimensions<T>(tensor: &ETensor<T>, n: usize, stride_offset: usize) -> Table
where
    T: Copy + Display,
{
    let dim = tensor.rank() - n;
    let size = tensor.shape.sizes[dim];
    let stride = tensor.shape.strides[dim];

    let rows = (0..size)
        .map(|index| {
            let offset = offset_fn(stride, index, size) + stride_offset;
            odd_dimensions(tensor, n - 1, offset)
        })
        .collect();

    let table = Table::init(rows);
    set_style(table)
}

fn set_style(mut table: Table) -> Table {
    table.set_format(*FORMAT_BOX_CHARS);
    table
}
