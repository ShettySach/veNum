#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(pub u32);

impl From<u32> for Symbol {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Const(i64),
    Sym(Symbol),
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),
    Div(Box<Dim>, Box<Dim>),
    Mod(Box<Dim>, Box<Dim>),
}

impl Dim {
    pub fn constant(value: i64) -> Self {
        Self::Const(value)
    }

    pub fn symbol(symbol: Symbol) -> Self {
        Self::Sym(symbol)
    }

    pub fn add(lhs: Dim, rhs: Dim) -> Self {
        Self::Add(Box::new(lhs), Box::new(rhs))
    }

    pub fn mul(lhs: Dim, rhs: Dim) -> Self {
        Self::Mul(Box::new(lhs), Box::new(rhs))
    }

    pub fn div(lhs: Dim, rhs: Dim) -> Self {
        Self::Div(Box::new(lhs), Box::new(rhs))
    }

    pub fn modulo(lhs: Dim, rhs: Dim) -> Self {
        Self::Mod(Box::new(lhs), Box::new(rhs))
    }

    pub fn as_const(&self) -> Option<i64> {
        match self {
            Dim::Const(v) => Some(*v),
            _ => None,
        }
    }
}
