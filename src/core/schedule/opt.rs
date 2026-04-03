#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Opt {
    pub op: OptOp,
    pub axis: usize,
    pub amt: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OptOp {
    Tile,
    Vectorize,
    Unroll,
    Parallelize,
    GroupReduce,
    PadTo,
}
