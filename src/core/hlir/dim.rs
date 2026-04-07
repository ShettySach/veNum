#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

impl std::ops::Add for Dim {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for Dim {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Div for Dim {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::Div(Box::new(self), Box::new(rhs))
    }
}
