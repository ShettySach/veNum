//! Extraction cost model definitions for the upcoming typed egglog pipeline.

use egglog::extract::{Cost, CostModel};
use egglog::{ArcSort, EGraph, Function, FunctionRow, Value};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalExprCost {
    total: u64,
    class_rank: u8,
    canonical: String,
    assoc_penalty: u32,
    order_penalty: u32,
    rendered: String,
    head: String,
    add_terms: Vec<String>,
    mul_terms: Vec<String>,
}

impl Ord for CanonicalExprCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.total,
            self.class_rank,
            &self.canonical,
            self.assoc_penalty,
            self.order_penalty,
            &self.rendered,
        )
            .cmp(&(
                other.total,
                other.class_rank,
                &other.canonical,
                other.assoc_penalty,
                other.order_penalty,
                &other.rendered,
            ))
    }
}

impl PartialOrd for CanonicalExprCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Cost for CanonicalExprCost {
    fn identity() -> Self {
        Self {
            total: 0,
            class_rank: u8::MAX,
            canonical: String::new(),
            assoc_penalty: 0,
            order_penalty: 0,
            rendered: String::new(),
            head: String::new(),
            add_terms: Vec::new(),
            mul_terms: Vec::new(),
        }
    }

    fn unit() -> Self {
        Self {
            total: 1,
            ..Self::identity()
        }
    }

    fn combine(self, other: &Self) -> Self {
        Self {
            total: self.total.saturating_add(other.total),
            ..Self::identity()
        }
    }
}

#[derive(Default, Clone)]
pub struct CanonicalExprCostModel;

impl CostModel<CanonicalExprCost> for CanonicalExprCostModel {
    fn fold(
        &self,
        head: &str,
        children_cost: &[CanonicalExprCost],
        _head_cost: CanonicalExprCost,
    ) -> CanonicalExprCost {
        let children_total = children_cost
            .iter()
            .fold(0_u64, |sum, child| sum.saturating_add(child.total));
        let mut total = base_head_cost(head).saturating_add(children_total);
        if head == "Const" && !is_scalar_shape(children_cost.get(1)) {
            total = total.saturating_add(8);
        }
        if head == "Cast"
            && children_cost
                .first()
                .is_some_and(|child| child.head == "Cast")
        {
            total = total.saturating_add(6);
        }

        match head {
            "Add" | "Mul" => self.fold_commutative_variadic(head, total, children_cost),
            _ => self.fold_generic(head, total, children_cost),
        }
    }

    fn enode_cost(
        &self,
        _egraph: &EGraph,
        _func: &Function,
        _row: &FunctionRow,
    ) -> CanonicalExprCost {
        CanonicalExprCost::identity()
    }

    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> CanonicalExprCost {
        let rendered = match sort.name() {
            "i64" => egraph.value_to_base::<i64>(value).to_string(),
            "bool" => egraph.value_to_base::<bool>(value).to_string(),
            "String" => format!("{:?}", egraph.value_to_base::<egglog::sort::S>(value)),
            _ => format!("<{sort}:{value:?}>", sort = sort.name()),
        };

        CanonicalExprCost {
            total: 0,
            class_rank: u8::MAX,
            canonical: rendered.clone(),
            assoc_penalty: 0,
            order_penalty: 0,
            rendered,
            head: sort.name().to_string(),
            add_terms: Vec::new(),
            mul_terms: Vec::new(),
        }
    }
}

impl CanonicalExprCostModel {
    fn fold_generic(
        &self,
        head: &str,
        total: u64,
        children: &[CanonicalExprCost],
    ) -> CanonicalExprCost {
        let rendered_children: Vec<_> = children
            .iter()
            .map(|child| child.rendered.clone())
            .collect();
        let canonical_children: Vec<_> = children
            .iter()
            .map(|child| child.canonical.clone())
            .collect();
        let class_rank = expr_class_rank(head, children);

        CanonicalExprCost {
            total,
            class_rank,
            canonical: format!(
                "{class_rank}:{term}",
                term = render_term(head, &canonical_children)
            ),
            assoc_penalty: 0,
            order_penalty: 0,
            rendered: render_term(head, &rendered_children),
            head: head.to_string(),
            add_terms: Vec::new(),
            mul_terms: Vec::new(),
        }
    }

    fn fold_commutative_variadic(
        &self,
        head: &str,
        total: u64,
        children: &[CanonicalExprCost],
    ) -> CanonicalExprCost {
        let rendered_children: Vec<_> = children
            .iter()
            .map(|child| child.rendered.clone())
            .collect();
        let actual_terms = flatten_terms(head, children);
        let mut sorted_terms = actual_terms.clone();
        sorted_terms.sort();

        let assoc_penalty = children
            .iter()
            .fold(0_u32, |sum, child| sum.saturating_add(child.assoc_penalty))
            .saturating_add(
                children
                    .first()
                    .map_or(0, |child| u32::from(child.head == head)),
            );
        let order_penalty = inversion_count(&actual_terms);

        CanonicalExprCost {
            total,
            class_rank: 6,
            canonical: format!("6:{head}[{}]", sorted_terms.join(",")),
            assoc_penalty,
            order_penalty,
            rendered: render_term(head, &rendered_children),
            head: head.to_string(),
            add_terms: if head == "Add" {
                actual_terms.clone()
            } else {
                Vec::new()
            },
            mul_terms: if head == "Mul" {
                actual_terms
            } else {
                Vec::new()
            },
        }
    }
}

fn flatten_terms(head: &str, children: &[CanonicalExprCost]) -> Vec<String> {
    let mut terms = Vec::new();
    for child in children {
        match head {
            "Add" if child.head == "Add" => terms.extend(child.add_terms.clone()),
            "Mul" if child.head == "Mul" => terms.extend(child.mul_terms.clone()),
            _ => terms.push(child.canonical.clone()),
        }
    }
    terms
}

fn inversion_count(terms: &[String]) -> u32 {
    let mut inversions = 0_u32;
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            if terms[i] > terms[j] {
                inversions = inversions.saturating_add(1);
            }
        }
    }
    inversions
}

fn expr_class_rank(head: &str, children: &[CanonicalExprCost]) -> u8 {
    match head {
        "Const" => {
            if is_scalar_shape(children.get(1)) {
                0
            } else {
                3
            }
        }
        "Broadcast" if children.first().is_some_and(is_scalar_const) => 1,
        "Leaf" => 2,
        "Neg" | "Recip" | "Exp" | "Log" | "Sqrt" | "Sin" => 4,
        "Cast" => 5,
        "Reshape" | "Permute" | "Expand" | "Broadcast" => 6,
        "Add" | "Mul" | "Max" | "Min" => 6,
        _ => u8::MAX,
    }
}

fn is_scalar_const(child: &CanonicalExprCost) -> bool {
    child.head == "Const" && child.class_rank == 1
}

fn is_scalar_shape(shape: Option<&CanonicalExprCost>) -> bool {
    shape.is_some_and(|shape| shape.rendered == "(ShapeNil)")
}

fn base_head_cost(head: &str) -> u64 {
    match head {
        "Leaf" => 0,
        "Const" => 1,
        "Reshape" => 0,
        "Permute" | "Expand" | "Broadcast" => 1,
        "Neg" | "Recip" | "Exp" | "Log" | "Sqrt" | "Sin" | "Cast" => 4,
        "Add" | "Mul" | "Max" | "Min" => 4,
        _ => 0,
    }
}

fn render_term(head: &str, children: &[String]) -> String {
    if children.is_empty() {
        format!("({head})")
    } else {
        format!("({head} {})", children.join(" "))
    }
}
