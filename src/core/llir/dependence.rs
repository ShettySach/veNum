use super::affine::AffineExpr;

#[derive(Clone, Debug, PartialEq)]
pub struct Dependence {
    pub from: usize,
    pub to: usize,
    pub kind: DepKind,
    pub distance: Option<Vec<i64>>,
    pub relation: DependenceRelation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DependenceRelation {
    pub source_vars: Vec<String>,
    pub sink_vars: Vec<String>,
    pub constraints: Vec<AffineConstraint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineConstraint {
    pub expr: AffineExpr,
    pub kind: ConstraintKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
    Eq,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepKind {
    Raw,
    War,
    Waw,
}
