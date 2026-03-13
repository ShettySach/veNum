## User

[@Venum Lazy First Refactor Plan](zed:///agent/thread/420d8dc9-58d1-4375-88fb-d8c140f4b8c9?name=Venum+Lazy+First+Refactor+Plan)
Continue please

**Tool Call: List the `venum` directory's contents**
Status: Completed

# Folders:
venum/.direnv
venum/assets
venum/examples
venum/src
venum/target

# Files:
venum/.gitignore
venum/Cargo.lock
venum/Cargo.toml
venum/PLAN.md
venum/README.md
venum/TODO.md
venum/flake.lock
venum/flake.nix


**Tool Call: Check project diagnostics**
Status: Completed

/home/sword/Desktop/code/venum/src/core/lazy/tensor.rs: 0 error(s), 2 warning(s)
/home/sword/Desktop/code/venum/src/core/lazy/optimize.rs: 1 error(s), 2 warning(s)


**Tool Call: git status --short**
Status: Completed

```
M examples/dag_render.rs
 M src/core/eager/mod.rs
 M src/core/lazy/graph.rs
 M src/core/lazy/jit.rs
 M src/core/lazy/mod.rs
 M src/core/lazy/optimize.rs
 M src/core/lazy/render.rs
 M src/core/lazy/schedule.rs
 M src/core/lazy/tensor.rs
 M src/core/lazy/tests.rs
 M src/lib.rs
?? src/core/lazy/context.rs
```

**Tool Call: Read file `src/core/lazy/optimize.rs` (lines 1-400)**
Status: Completed

```json
{
  "Text": "use std::collections::{HashMap, HashSet};\n\nuse super::dtype::DType;\nuse super::graph::{Graph, Node, NodeId, Op};\n\n/// Run the egglog optimizer on the graph rooted at `root`.\n/// Returns a new, optimized Graph and the new root NodeId.\npub fn optimize(graph: &Graph, root: NodeId) -> (Graph, NodeId) {\n    // 1. Convert our Graph into egglog terms.\n    let mut egraph = egglog::EGraph::default();\n\n    // Define the tensor expression sort.\n    egraph\n        .parse_and_run_program(\n            None,\n            r#\"\n            (datatype TExpr\n                (tLoad i64)\n                (tConst f64)\n                (tAdd TExpr TExpr)\n                (tSub TExpr TExpr)\n                (tMul TExpr TExpr)\n                (tDiv TExpr TExpr)\n                (tExp TExpr)\n                (tLn  TExpr)\n                (tSqrt TExpr)\n                (tNeg TExpr)\n            )\n            \"#,\n        )\n        .unwrap();\n\n    // Algebraic simplification rules.\n    egraph\n        .parse_and_run_program(\n            None,\n            r#\"\n            ;; --- Identity rules ---\n            (rewrite (tAdd ?a (tConst 0.0)) ?a)\n            (rewrite (tAdd (tConst 0.0) ?a) ?a)\n            (rewrite (tSub ?a (tConst 0.0)) ?a)\n            (rewrite (tMul ?a (tConst 1.0)) ?a)\n            (rewrite (tMul (tConst 1.0) ?a) ?a)\n            (rewrite (tDiv ?a (tConst 1.0)) ?a)\n\n            ;; --- Zero rules ---\n            (rewrite (tMul ?a (tConst 0.0)) (tConst 0.0))\n            (rewrite (tMul (tConst 0.0) ?a) (tConst 0.0))\n\n            ;; --- Double negation ---\n            (rewrite (tNeg (tNeg ?a)) ?a)\n\n            ;; --- Inverse ops ---\n            (rewrite (tExp (tLn ?a)) ?a)\n            (rewrite (tLn (tExp ?a)) ?a)\n            (rewrite (tSqrt (tMul ?a ?a)) ?a)\n\n            ;; --- Self-cancellation ---\n            (rewrite (tSub ?a ?a) (tConst 0.0))\n            (rewrite (tDiv ?a ?a) (tConst 1.0))\n\n            ;; --- Commutativity ---\n            (rewrite (tAdd ?a ?b) (tAdd ?b ?a))\n            (rewrite (tMul ?a ?b) (tMul ?b ?a))\n\n            ;; --- Strength reduction ---\n            (rewrite (tAdd ?a ?a) (tMul (tConst 2.0) ?a))\n            \"#,\n        )\n        .unwrap();\n\n    // 2. Insert our DAG as egglog terms.\n    let term_str = node_to_egglog(graph, root);\n    let insert_cmd = format!(\"(let root {})\", term_str);\n    egraph.parse_and_run_program(None, &insert_cmd).unwrap();\n\n    // 3. Run equality saturation.\n    egraph.parse_and_run_program(None, \"(run 10)\").unwrap();\n\n    // 4. Extract the best term.\n    let outputs = egraph\n        .parse_and_run_program(None, \"(extract root)\")\n        .unwrap();\n\n    let extracted = outputs[0].to_string();\n\n    // 5. Parse the extracted term back into a new Graph.\n    let (new_graph, new_root) = parse_egglog_term(graph, &extracted);\n\n    (new_graph, new_root)\n}\n\n/// Convert a graph node to an egglog s-expression string.\nfn node_to_egglog(graph: &Graph, id: NodeId) -> String {\n    let node = graph.node(id);\n    match &node.op {\n        Op::Load => format!(\"(tLoad {})\", id.0),\n        Op::Const(v) => format!(\"(tConst {:.1})\", *v as f64),\n        Op::Add => format!(\n            \"(tAdd {} {})\",\n            node_to_egglog(graph, node.inputs[0]),\n            node_to_egglog(graph, node.inputs[1])\n        ),\n        Op::Sub => format!(\n            \"(tSub {} {})\",\n            node_to_egglog(graph, node.inputs[0]),\n            node_to_egglog(graph, node.inputs[1])\n        ),\n        Op::Mul => format!(\n            \"(tMul {} {})\",\n            node_to_egglog(graph, node.inputs[0]),\n            node_to_egglog(graph, node.inputs[1])\n        ),\n        Op::Div => format!(\n            \"(tDiv {} {})\",\n            node_to_egglog(graph, node.inputs[0]),\n            node_to_egglog(graph, node.inputs[1])\n        ),\n        Op::Exp => format!(\"(tExp {})\", node_to_egglog(graph, node.inputs[0])),\n        Op::Ln => format!(\"(tLn {})\", node_to_egglog(graph, node.inputs[0])),\n        Op::Sqrt => format!(\"(tSqrt {})\", node_to_egglog(graph, node.inputs[0])),\n        Op::Neg => format!(\"(tNeg {})\", node_to_egglog(graph, node.inputs[0])),\n\n        // Non-elementwise / shape ops are currently not modeled in egglog.\n        // Keep optimizer safe by treating them as opaque leaves.\n        Op::Reshape(_)\n        | Op::Permute(_)\n        | Op::Transpose(_, _)\n        | Op::Expand(_)\n        | Op::Slice(_)\n        | Op::Flip(_)\n        | Op::Squeeze\n        | Op::Unsqueeze(_)\n        | Op::Pad(_, _) => format!(\"(tLoad {})\", id.0),\n    }\n}\n\n/// Parse an egglog extracted term back into a Graph.\n/// Reuses Load buffers from the original graph.\nfn parse_egglog_term(original: &Graph, term: &str) -> (Graph, NodeId) {\n    let term = term.trim();\n    let mut graph = Graph::new();\n    let root = parse_sexpr(original, term, &mut graph);\n    (graph, root)\n}\n\n/// Recursive s-expression parser for egglog output.\nfn parse_sexpr(original: &Graph, s: &str, graph: &mut Graph) -> NodeId {\n    let s = s.trim();\n\n    // Not an s-expression — should be a bare token (shouldn't happen at top level).\n    if !s.starts_with('(') {\n        panic!(\"Expected s-expression, got: {}\", s);\n    }\n\n    // Strip outer parens.\n    let inner = &s[1..s.len() - 1];\n\n    // Split into head and args.\n    let (head, rest) = split_head(inner);\n\n    match head {\n        \"tLoad\" => {\n            let id: usize = rest.trim().parse().unwrap();\n            let orig_node = original.node(NodeId(id));\n            graph.add_node(Node {\n                op: Op::Load,\n                inputs: vec![],\n                shape: orig_node.shape.clone(),\n                dtype: orig_node.dtype,\n                buffer: orig_node.buffer.clone(),\n            })\n        }\n        \"tConst\" => {\n            let val: f64 = rest.trim().parse().unwrap();\n            // Infer shape from context — for now use [1] for scalar constants.\n            // The JIT handles Const nodes inline (f32const), so shape doesn't matter for codegen.\n            graph.add_node(Node {\n                op: Op::Const(val as f32),\n                inputs: vec![],\n                shape: vec![1],\n                dtype: DType::F32,\n                buffer: None,\n            })\n        }\n        \"tAdd\" | \"tSub\" | \"tMul\" | \"tDiv\" => {\n            let (arg1_str, arg2_str) = split_two_args(rest);\n            let lhs = parse_sexpr(original, arg1_str, graph);\n            let rhs = parse_sexpr(original, arg2_str, graph);\n            let op = match head {\n                \"tAdd\" => Op::Add,\n                \"tSub\" => Op::Sub,\n                \"tMul\" => Op::Mul,\n                \"tDiv\" => Op::Div,\n                _ => unreachable!(),\n            };\n            let shape = graph.node(lhs).shape.clone();\n            graph.add_node(Node {\n                op,\n                inputs: vec![lhs, rhs],\n                shape,\n                dtype: DType::F32,\n                buffer: None,\n            })\n        }\n        \"tExp\" | \"tLn\" | \"tSqrt\" | \"tNeg\" => {\n            let arg = parse_sexpr(original, rest.trim(), graph);\n            let op = match head {\n                \"tExp\" => Op::Exp,\n                \"tLn\" => Op::Ln,\n                \"tSqrt\" => Op::Sqrt,\n                \"tNeg\" => Op::Neg,\n                _ => unreachable!(),\n            };\n            let shape = graph.node(arg).shape.clone();\n            graph.add_node(Node {\n                op,\n                inputs: vec![arg],\n                shape,\n                dtype: DType::F32,\n                buffer: None,\n            })\n        }\n        _ => {\n            // Be conservative for any unknown extracted term:\n            // materialize it as an opaque load by reusing original root buffer.\n            let orig_node = original.node(NodeId(0));\n            graph.add_node(Node {\n                op: Op::Load,\n                inputs: vec![],\n                shape: orig_node.shape.clone(),\n                dtype: orig_node.dtype,\n                buffer: orig_node.buffer.clone(),\n            })\n        }\n    }\n}\n\n/// Split \"head rest...\" from an s-expression body.\nfn split_head(s: &str) -> (&str, &str) {\n    let s = s.trim();\n    if let Some(idx) = s.find(|c: char| c.is_whitespace()) {\n        (&s[..idx], &s[idx..])\n    } else {\n        (s, \"\")\n    }\n}\n\n/// Split two s-expression arguments from a string like \" (tAdd ...) (tMul ...)\".\nfn split_two_args(s: &str) -> (&str, &str) {\n    let s = s.trim();\n    let end_of_first = find_sexpr_end(s);\n    let first = &s[..end_of_first];\n    let second = s[end_of_first..].trim();\n    (first, second)\n}\n\n/// Find the end index of the first s-expression in the string.\nfn find_sexpr_end(s: &str) -> usize {\n    let s = s.trim();\n    if s.starts_with('(') {\n        let mut depth = 0;\n        for (i, c) in s.char_indices() {\n            match c {\n                '(' => depth += 1,\n                ')' => {\n                    depth -= 1;\n                    if depth == 0 {\n                        return i + 1;\n                    }\n                }\n                _ => {}\n            }\n        }\n        s.len()\n    } else {\n        // Bare token — find next whitespace or end.\n        s.find(|c: char| c.is_whitespace()).unwrap_or(s.len())\n    }\n}\n"
}
```

**Tool Call: Read file `src/core/lazy/tensor.rs` (lines 1-400)**
Status: Completed

```json
{
  "Text": "use anyhow::{anyhow, bail, Result};\nuse std::sync::Arc;\nuse std::{\n    cmp::Ordering,\n    iter::{Product, Sum},\n    ops::{Add, Div, Mul, Sub},\n};\n\nuse crate::core::{\n    eager::ETensor as EagerTensor,\n    errors::{ExpansionError, ReshapeError, TransposeError, UnsqueezeError},\n    iters::Indexer,\n};\n\nuse super::context::Context;\nuse super::graph::{Graph, NodeId, Op};\nuse super::jit::compile_kernel;\nuse super::optimize;\nuse super::render;\nuse super::schedule::{build_schedule, ScheduleItem};\n\n#[derive(Clone, Debug, PartialEq)]\npub struct RealizedTensor {\n    data: Vec<f32>,\n    shape: Vec<usize>,\n}\n\nimpl RealizedTensor {\n    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {\n        Self { data, shape }\n    }\n\n    pub fn data(&self) -> &[f32] {\n        &self.data\n    }\n\n    pub fn shape(&self) -> &[usize] {\n        &self.shape\n    }\n\n    pub fn numel(&self) -> usize {\n        self.shape.iter().product()\n    }\n\n    pub fn to_eager(&self) -> Result<EagerTensor<f32>> {\n        EagerTensor::new(&self.data, &self.shape)\n    }\n}\n\n#[derive(Clone)]\npub struct Tensor {\n    graph: Arc<std::sync::Mutex<Graph>>,\n    id: NodeId,\n    shape: Vec<usize>,\n}\n\nimpl Tensor {\n    pub fn from_tensor(cx: &Context, tensor: &EagerTensor<f32>) -> Self {\n        let data = Arc::new(tensor.data().to_vec());\n        let shape = tensor.sizes().to_vec();\n\n        let graph = cx.graph();\n        let id = {\n            let mut g = graph.lock().unwrap();\n            g.load(data, shape.clone())\n        };\n\n        Self { graph, id, shape }\n    }\n\n    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {\n        let data = Arc::new(data.to_vec());\n\n        let graph = cx.graph();\n        let id = {\n            let mut g = graph.lock().unwrap();\n            g.load(data, shape.clone())\n        };\n\n        Self { graph, id, shape }\n    }\n\n    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {\n        let graph = cx.graph();\n        let id = {\n            let mut g = graph.lock().unwrap();\n            g.constant(value, shape.clone())\n        };\n\n        Self { graph, id, shape }\n    }\n\n    pub fn shape(&self) -> &[usize] {\n        &self.shape\n    }\n\n    pub fn numel(&self) -> usize {\n        self.shape.iter().product()\n    }\n\n    // -------- elementwise lazy ops --------\n\n    fn binary_op(&self, rhs: &Tensor, op: Op) -> Tensor {\n        let same_graph = Arc::ptr_eq(&self.graph, &rhs.graph);\n        let mut graph = self.graph.lock().unwrap();\n\n        let rhs_id = if same_graph {\n            rhs.id\n        } else {\n            let rhs_graph = rhs.graph.lock().unwrap();\n            import_subgraph(&rhs_graph, rhs.id, &mut graph)\n        };\n\n        let id = graph.binary(op, self.id, rhs_id);\n        let shape = graph.node(id).shape.clone();\n        drop(graph);\n\n        Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        }\n    }\n\n    fn unary_op(&self, op: Op) -> Tensor {\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.unary(op, self.id);\n        let shape = graph.node(id).shape.clone();\n        drop(graph);\n\n        Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        }\n    }\n\n    pub fn exp(&self) -> Tensor {\n        self.unary_op(Op::Exp)\n    }\n\n    pub fn ln(&self) -> Tensor {\n        self.unary_op(Op::Ln)\n    }\n\n    pub fn sqrt(&self) -> Tensor {\n        self.unary_op(Op::Sqrt)\n    }\n\n    pub fn neg(&self) -> Tensor {\n        self.unary_op(Op::Neg)\n    }\n\n    // -------- shape lazy ops --------\n\n    pub fn reshape(&self, sizes: Vec<usize>) -> Result<Tensor> {\n        let old_numel: usize = self.shape.iter().product();\n        let new_numel: usize = sizes.iter().product();\n        if old_numel != new_numel {\n            bail!(ReshapeError {\n                current_shape: self.shape.clone(),\n                new_shape: sizes\n            });\n        }\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.reshape(self.id, sizes.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape: sizes,\n        })\n    }\n\n    pub fn permute(&self, permutation: Vec<usize>) -> Result<Tensor> {\n        if permutation.len() != self.shape.len() {\n            bail!(ReshapeError {\n                current_shape: self.shape.clone(),\n                new_shape: self.shape.clone()\n            });\n        }\n\n        let mut seen = vec![false; permutation.len()];\n        for &p in &permutation {\n            if p >= permutation.len() || seen[p] {\n                bail!(ReshapeError {\n                    current_shape: self.shape.clone(),\n                    new_shape: self.shape.clone()\n                });\n            }\n            seen[p] = true;\n        }\n\n        let shape = permutation\n            .iter()\n            .map(|&i| self.shape[i])\n            .collect::<Vec<_>>();\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.permute(self.id, permutation, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Tensor> {\n        let rank = self.shape.len();\n        if rank < 2 || dim_1 >= rank || dim_2 >= rank {\n            bail!(TransposeError);\n        }\n\n        let mut shape = self.shape.clone();\n        shape.swap(dim_1, dim_2);\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.transpose(self.id, dim_1, dim_2, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn expand(&self, expansions: Vec<usize>) -> Result<Tensor> {\n        if expansions.len() != self.shape.len() {\n            bail!(ExpansionError {\n                size: self.shape.len(),\n                expansion: expansions.len()\n            });\n        }\n\n        for (&size, &exp) in self.shape.iter().zip(expansions.iter()) {\n            if !(exp == size || size == 1) {\n                bail!(ExpansionError {\n                    size,\n                    expansion: exp\n                });\n            }\n        }\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.expand(self.id, expansions.clone(), expansions.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape: expansions,\n        })\n    }\n\n    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor> {\n        if ranges.len() != self.shape.len() {\n            bail!(anyhow!(\n                \"slice ranges rank mismatch: got {}, expected {}\",\n                ranges.len(),\n                self.shape.len()\n            ));\n        }\n\n        let mut out = Vec::with_capacity(self.shape.len());\n        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {\n            let size = self.shape[dim];\n            let end = if end_raw == 0 { size } else { end_raw };\n            if start > end || end > size {\n                bail!(anyhow!(\n                    \"slice range {:?} out of bounds for dim {} with size {}\",\n                    (start, end),\n                    dim,\n                    size\n                ));\n            }\n            out.push(end - start);\n        }\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.slice(self.id, ranges, out.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape: out,\n        })\n    }\n\n    pub fn flip(&self, flips: Vec<usize>) -> Result<Tensor> {\n        for &d in &flips {\n            if d >= self.shape.len() {\n                bail!(anyhow!(\n                    \"flip dimension {} out of bounds for rank {}\",\n                    d,\n                    self.shape.len()\n                ));\n            }\n        }\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.flip(self.id, flips, self.shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape: self.shape.clone(),\n        })\n    }\n\n    pub fn squeeze(&self) -> Result<Tensor> {\n        let mut shape: Vec<usize> = self.shape.iter().copied().filter(|&s| s != 1).collect();\n        if shape.is_empty() {\n            shape.push(1);\n        }\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.squeeze(self.id, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn unsqueeze(&self, new_rank: usize) -> Result<Tensor> {\n        let rank = self.shape.len();\n        if new_rank < rank {\n            bail!(UnsqueezeError {\n                current: rank,\n                new_rank\n            });\n        }\n        if new_rank == rank {\n            return Ok(self.clone());\n        }\n\n        let mut shape = vec![1; new_rank - rank];\n        shape.extend_from_slice(&self.shape);\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.unsqueeze(self.id, new_rank, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn pad(&self, constant: f32, padding: Vec<(usize, usize)>) -> Result<Tensor> {\n        let mut pad = padding;\n        pad.resize(self.shape.len(), (0, 0));\n\n        let mut shape = Vec::with_capacity(self.shape.len());\n        for (s, (l, r)) in self.shape.iter().copied().zip(pad.iter().copied()) {\n            shape.push(l + s + r);\n        }\n\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.pad(self.id, constant, pad, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    // -------- reduce lazy ops --------\n\n    pub fn sum_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {\n        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.sum(self.id, dimensions, keepdims, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn product_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {\n        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.prod(self.id, dimensions, keepdims, shape.clone());\n        drop(graph);\n\n"
}
```

**Tool Call: Read file `src/core/lazy/schedule.rs` (lines 1-400)**
Status: Completed

```json
{
  "Text": "use std::collections::{HashMap, HashSet};\n\nuse super::graph::{Graph, NodeId, Op};\n\n/// A fused elementwise kernel: a tree of ops over input buffers producing one output.\n#[derive(Debug)]\npub struct FusedKernel {\n    pub root: NodeId,\n    /// Leaf node ids that must be realized before this kernel runs.\n    pub input_buffers: Vec<NodeId>,\n    /// Number of elements in the output.\n    pub numel: usize,\n}\n\n/// Shape-op barrier item.\n/// These run as standalone schedule steps (not fused into elementwise kernels yet).\n#[derive(Debug, Clone)]\npub struct ShapeOpItem {\n    pub root: NodeId,\n    pub op: Op,\n    pub input: NodeId,\n    pub shape: Vec<usize>,\n}\n\n/// Reduce-op barrier item.\n/// Reductions are scheduled separately from elementwise fusion.\n#[derive(Debug, Clone)]\npub struct ReduceOpItem {\n    pub root: NodeId,\n    pub op: Op,\n    pub input: NodeId,\n    pub shape: Vec<usize>,\n}\n\n/// An item in the execution schedule.\n#[derive(Debug)]\npub enum ScheduleItem {\n    Fused(FusedKernel),\n    Shape(ShapeOpItem),\n    Reduce(ReduceOpItem),\n}\n\n/// Build a linear execution schedule from the graph, rooted at `root`.\n///\n/// Algorithm:\n/// 1. Topological sort from root (reverse post-order DFS).\n/// 2. Walk topo order:\n///    - Leaf nodes (`Load`, `Const`) are skipped.\n///    - Shape ops become `ScheduleItem::Shape` barriers.\n///    - Reduce ops become `ScheduleItem::Reduce` barriers.\n///    - Elementwise ops become `ScheduleItem::Fused`.\n/// 3. Fusing rule for elementwise kernels:\n///    - Input can be inlined iff:\n///      a) input is elementwise\n///      b) input has exactly one consumer\n///      c) input has same `numel` as consumer\npub fn build_schedule(graph: &Graph, root: NodeId) -> Vec<ScheduleItem> {\n    let topo = topo_sort(graph, root);\n    let consumer_counts = compute_consumer_counts(graph, &topo);\n\n    // Nodes that are inlined into fused elementwise kernels.\n    let mut inlined: HashSet<NodeId> = HashSet::new();\n\n    let mut schedule = Vec::new();\n\n    for &id in &topo {\n        if inlined.contains(&id) {\n            continue;\n        }\n\n        let node = graph.node(id);\n\n        // Leaves don't need a schedule item.\n        if matches!(node.op, Op::Load | Op::Const(_)) {\n            continue;\n        }\n\n        // Shape ops are explicit barriers.\n        if node.op.is_shape_op() {\n            let input = node.inputs[0];\n            schedule.push(ScheduleItem::Shape(ShapeOpItem {\n                root: id,\n                op: node.op.clone(),\n                input,\n                shape: node.shape.clone(),\n            }));\n            continue;\n        }\n\n        // Reduce ops are explicit barriers.\n        if node.op.is_reduce_op() {\n            let input = node.inputs[0];\n            schedule.push(ScheduleItem::Reduce(ReduceOpItem {\n                root: id,\n                op: node.op.clone(),\n                input,\n                shape: node.shape.clone(),\n            }));\n            continue;\n        }\n\n        // Elementwise ops can be fused.\n        if node.op.is_elementwise() {\n            let mut input_buffers = Vec::new();\n            collect_kernel_inputs(\n                graph,\n                id,\n                &consumer_counts,\n                &mut inlined,\n                &mut input_buffers,\n            );\n\n            schedule.push(ScheduleItem::Fused(FusedKernel {\n                root: id,\n                input_buffers,\n                numel: node.numel(),\n            }));\n            continue;\n        }\n\n        // Future non-elementwise ops (matmul/conv/etc.) are intentionally skipped here.\n    }\n\n    schedule\n}\n\n/// Recursively collect the leaf inputs of a fused kernel rooted at `id`.\n/// Mark intermediate fusible nodes as inlined.\nfn collect_kernel_inputs(\n    graph: &Graph,\n    id: NodeId,\n    consumer_counts: &HashMap<NodeId, usize>,\n    inlined: &mut HashSet<NodeId>,\n    inputs: &mut Vec<NodeId>,\n) {\n    let node = graph.node(id);\n\n    for &input_id in &node.inputs {\n        let input_node = graph.node(input_id);\n\n        // Const nodes are always inlined (they emit f32const in JIT).\n        if matches!(input_node.op, Op::Const(_)) {\n            inlined.insert(input_id);\n            continue;\n        }\n\n        let can_inline = input_node.op.is_elementwise()\n            && *consumer_counts.get(&input_id).unwrap_or(&0) == 1\n            && input_node.numel() == node.numel();\n\n        if can_inline {\n            inlined.insert(input_id);\n            collect_kernel_inputs(graph, input_id, consumer_counts, inlined, inputs);\n        } else {\n            // This is a barrier leaf input for the fused kernel.\n            if !inputs.contains(&input_id) {\n                inputs.push(input_id);\n            }\n        }\n    }\n}\n\n/// Topological sort via reverse post-order DFS, rooted at `root`.\nfn topo_sort(graph: &Graph, root: NodeId) -> Vec<NodeId> {\n    let mut visited = HashSet::new();\n    let mut order = Vec::new();\n    topo_dfs(graph, root, &mut visited, &mut order);\n    order\n}\n\nfn topo_dfs(graph: &Graph, id: NodeId, visited: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) {\n    if !visited.insert(id) {\n        return;\n    }\n    let node = graph.node(id);\n    for &input in &node.inputs {\n        topo_dfs(graph, input, visited, order);\n    }\n    order.push(id);\n}\n\n/// Count how many times each node in `topo` is referenced as an input.\nfn compute_consumer_counts(graph: &Graph, topo: &[NodeId]) -> HashMap<NodeId, usize> {\n    let mut counts: HashMap<NodeId, usize> = HashMap::new();\n    let topo_set: HashSet<NodeId> = topo.iter().copied().collect();\n\n    for &id in topo {\n        let node = graph.node(id);\n        for &input_id in &node.inputs {\n            if topo_set.contains(&input_id) {\n                *counts.entry(input_id).or_insert(0) += 1;\n            }\n        }\n    }\n\n    counts\n}\n"
}
```

**Tool Call: Read file `src/core/lazy/graph.rs` (lines 1-400)**
Status: Completed

```json
{
  "Text": "use std::sync::Arc;\n\nuse super::dtype::{Buffer, DType};\n\n#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]\npub struct NodeId(pub usize);\n\n#[derive(Clone, Debug)]\npub enum Op {\n    // Leaf\n    Const(f32),\n    Load,\n\n    // Binary elementwise\n    Add,\n    Sub,\n    Mul,\n    Div,\n\n    // Unary elementwise\n    Exp,\n    Ln,\n    Sqrt,\n    Neg,\n\n    // Shape ops (lazy graph nodes)\n    Reshape(Vec<usize>),\n    Permute(Vec<usize>),\n    Transpose(usize, usize),\n    Expand(Vec<usize>),\n    Slice(Vec<(usize, usize)>),\n    Flip(Vec<usize>),\n    Squeeze,\n    Unsqueeze(usize),\n    Pad(f32, Vec<(usize, usize)>),\n\n    // Reduce ops\n    Sum(Vec<usize>, bool),\n    Prod(Vec<usize>, bool),\n    Max(Vec<usize>, bool),\n    Min(Vec<usize>, bool),\n}\n\nimpl Op {\n    pub fn is_elementwise(&self) -> bool {\n        matches!(\n            self,\n            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Exp | Op::Ln | Op::Sqrt | Op::Neg\n        )\n    }\n\n    pub fn is_binary(&self) -> bool {\n        matches!(self, Op::Add | Op::Sub | Op::Mul | Op::Div)\n    }\n\n    pub fn is_unary(&self) -> bool {\n        matches!(self, Op::Exp | Op::Ln | Op::Sqrt | Op::Neg)\n    }\n\n    pub fn is_shape_op(&self) -> bool {\n        matches!(\n            self,\n            Op::Reshape(_)\n                | Op::Permute(_)\n                | Op::Transpose(_, _)\n                | Op::Expand(_)\n                | Op::Slice(_)\n                | Op::Flip(_)\n                | Op::Squeeze\n                | Op::Unsqueeze(_)\n                | Op::Pad(_, _)\n        )\n    }\n\n    pub fn is_reduce_op(&self) -> bool {\n        matches!(\n            self,\n            Op::Sum(_, _) | Op::Prod(_, _) | Op::Max(_, _) | Op::Min(_, _)\n        )\n    }\n}\n\n#[derive(Clone, Debug)]\npub struct Node {\n    pub op: Op,\n    pub inputs: Vec<NodeId>,\n    pub shape: Vec<usize>,\n    pub dtype: DType,\n    pub buffer: Option<Buffer>,\n}\n\nimpl Node {\n    pub fn numel(&self) -> usize {\n        self.shape.iter().product()\n    }\n}\n\npub struct Graph {\n    pub nodes: Vec<Node>,\n}\n\nimpl Graph {\n    pub fn new() -> Self {\n        Graph { nodes: Vec::new() }\n    }\n\n    pub fn add_node(&mut self, node: Node) -> NodeId {\n        let id = NodeId(self.nodes.len());\n        self.nodes.push(node);\n        id\n    }\n\n    pub fn node(&self, id: NodeId) -> &Node {\n        &self.nodes[id.0]\n    }\n\n    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {\n        &mut self.nodes[id.0]\n    }\n\n    pub fn load(&mut self, data: Arc<Vec<f32>>, shape: Vec<usize>) -> NodeId {\n        let buffer = Buffer::F32(data);\n        self.add_node(Node {\n            op: Op::Load,\n            inputs: vec![],\n            shape,\n            dtype: DType::F32,\n            buffer: Some(buffer),\n        })\n    }\n\n    pub fn constant(&mut self, value: f32, shape: Vec<usize>) -> NodeId {\n        self.add_node(Node {\n            op: Op::Const(value),\n            inputs: vec![],\n            shape,\n            dtype: DType::F32,\n            buffer: None,\n        })\n    }\n\n    pub fn binary(&mut self, op: Op, lhs: NodeId, rhs: NodeId) -> NodeId {\n        let shape = self.node(lhs).shape.clone();\n        self.add_node(Node {\n            op,\n            inputs: vec![lhs, rhs],\n            shape,\n            dtype: DType::F32,\n            buffer: None,\n        })\n    }\n\n    pub fn unary(&mut self, op: Op, input: NodeId) -> NodeId {\n        let shape = self.node(input).shape.clone();\n        self.add_node(Node {\n            op,\n            inputs: vec![input],\n            shape,\n            dtype: DType::F32,\n            buffer: None,\n        })\n    }\n\n    // --- Shape node builders ---\n\n    pub fn reshape(&mut self, input: NodeId, shape: Vec<usize>) -> NodeId {\n        self.add_node(Node {\n            op: Op::Reshape(shape.clone()),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn permute(&mut self, input: NodeId, permutation: Vec<usize>, shape: Vec<usize>) -> NodeId {\n        self.add_node(Node {\n            op: Op::Permute(permutation),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn transpose(\n        &mut self,\n        input: NodeId,\n        dim_1: usize,\n        dim_2: usize,\n        shape: Vec<usize>,\n    ) -> NodeId {\n        self.add_node(Node {\n            op: Op::Transpose(dim_1, dim_2),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn expand(&mut self, input: NodeId, expansions: Vec<usize>, shape: Vec<usize>) -> NodeId {\n        self.add_node(Node {\n            op: Op::Expand(expansions),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn slice(\n        &mut self,\n        input: NodeId,\n        ranges: Vec<(usize, usize)>,\n        shape: Vec<usize>,\n    ) -> NodeId {\n        self.add_node(Node {\n            op: Op::Slice(ranges),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn flip(&mut self, input: NodeId, flips: Vec<usize>, shape: Vec<usize>) -> NodeId {\n        self.add_node(Node {\n            op: Op::Flip(flips),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn squeeze(&mut self, input: NodeId, shape: Vec<usize>) -> NodeId {\n        self.add_node(Node {\n            op: Op::Squeeze,\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn unsqueeze(&mut self, input: NodeId, unsqueezed: usize, shape: Vec<usize>) -> NodeId {\n        self.add_node(Node {\n            op: Op::Unsqueeze(unsqueezed),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn pad(\n        &mut self,\n        input: NodeId,\n        constant: f32,\n        padding: Vec<(usize, usize)>,\n        shape: Vec<usize>,\n    ) -> NodeId {\n        self.add_node(Node {\n            op: Op::Pad(constant, padding),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    // --- Reduce node builders ---\n\n    pub fn sum(\n        &mut self,\n        input: NodeId,\n        dimensions: Vec<usize>,\n        keepdims: bool,\n        shape: Vec<usize>,\n    ) -> NodeId {\n        self.add_node(Node {\n            op: Op::Sum(dimensions, keepdims),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn prod(\n        &mut self,\n        input: NodeId,\n        dimensions: Vec<usize>,\n        keepdims: bool,\n        shape: Vec<usize>,\n    ) -> NodeId {\n        self.add_node(Node {\n            op: Op::Prod(dimensions, keepdims),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn max(\n        &mut self,\n        input: NodeId,\n        dimensions: Vec<usize>,\n        keepdims: bool,\n        shape: Vec<usize>,\n    ) -> NodeId {\n        self.add_node(Node {\n            op: Op::Max(dimensions, keepdims),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    pub fn min(\n        &mut self,\n        input: NodeId,\n        dimensions: Vec<usize>,\n        keepdims: bool,\n        shape: Vec<usize>,\n    ) -> NodeId {\n        self.add_node(Node {\n            op: Op::Min(dimensions, keepdims),\n            inputs: vec![input],\n            shape,\n            dtype: self.node(input).dtype,\n            buffer: None,\n        })\n    }\n\n    /// Count how many nodes reference this node as an input.\n    pub fn consumer_count(&self, id: NodeId) -> usize {\n        self.nodes.iter().filter(|n| n.inputs.contains(&id)).count()\n    }\n}\n"
}
```

**Tool Call: Read file `src/core/lazy/mod.rs` (lines 1-200)**
Status: Completed

```json
{
  "Text": "mod context;\nmod dtype;\nmod graph;\nmod jit;\nmod optimize;\nmod render;\nmod schedule;\nmod tensor;\nmod tests;\n\npub use context::Context;\npub use tensor::{RealizedTensor, Tensor};\n\n// Temporary compatibility alias while migrating call sites.\npub type LTensor = Tensor;\n"
}
```

**Tool Call: Read file `src/core/lazy/context.rs` (lines 1-200)**
Status: Completed

```json
{
  "Text": "use std::sync::{Arc, Mutex};\n\nuse super::graph::Graph;\n\n/// Shared lazy execution context.\n///\n/// A `Context` owns a single computation graph so tensors created within the\n/// same context naturally share nodes and avoid cross-graph imports.\n#[derive(Clone)]\npub struct Context {\n    graph: Arc<Mutex<Graph>>,\n}\n\nimpl Default for Context {\n    fn default() -> Self {\n        Self::new()\n    }\n}\n\nimpl Context {\n    /// Create a new lazy context with an empty graph.\n    pub fn new() -> Self {\n        Self {\n            graph: Arc::new(Mutex::new(Graph::new())),\n        }\n    }\n\n    /// Get a clone of the shared graph handle.\n    pub(crate) fn graph(&self) -> Arc<Mutex<Graph>> {\n        Arc::clone(&self.graph)\n    }\n}\n"
}
```

**Tool Call: Check diagnostics for `venum/src/core/lazy/optimize.rs`**
Status: Completed

warning at line 1: unused imports: `HashMap` and `HashSet`
`#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default
warning at line 1: unused imports: `HashMap` and `HashSet`
`#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default
error at line 96: non-exhaustive patterns: `&Op::Sum(_, _)`, `&Op::Prod(_, _)`, `&Op::Max(_, _)` and 1 more not covered
the matched value is of type `&Op`


**Tool Call: Read file `src/core/lazy/tensor.rs` (lines 401-900)**
Status: Completed

```json
{
  "Text": "        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn max_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {\n        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.max(self.id, dimensions, keepdims, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn min_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {\n        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;\n        let mut graph = self.graph.lock().unwrap();\n        let id = graph.min(self.id, dimensions, keepdims, shape.clone());\n        drop(graph);\n\n        Ok(Tensor {\n            graph: Arc::clone(&self.graph),\n            id,\n            shape,\n        })\n    }\n\n    pub fn sum(&self) -> Result<Tensor> {\n        self.sum_dims((0..self.shape.len()).collect(), true)\n    }\n\n    pub fn product(&self) -> Result<Tensor> {\n        self.product_dims((0..self.shape.len()).collect(), true)\n    }\n\n    pub fn max(&self) -> Result<Tensor> {\n        self.max_dims((0..self.shape.len()).collect(), true)\n    }\n\n    pub fn min(&self) -> Result<Tensor> {\n        self.min_dims((0..self.shape.len()).collect(), true)\n    }\n\n    // -------- visualization --------\n\n    pub fn render_dag(&self) -> String {\n        let graph = self.graph.lock().unwrap();\n        render::render_dag(&graph, self.id)\n    }\n\n    pub fn render_fused_dag(&self) -> String {\n        let graph = self.graph.lock().unwrap();\n        render::render_fused_dag(&graph, self.id)\n    }\n\n    pub fn render_optimized_dag(&self) -> String {\n        let graph = self.graph.lock().unwrap();\n        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);\n        render::render_dag(&opt_graph, opt_root)\n    }\n\n    pub fn render_optimized_fused_dag(&self) -> String {\n        let graph = self.graph.lock().unwrap();\n        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);\n        render::render_fused_dag(&opt_graph, opt_root)\n    }\n\n    // -------- realize --------\n\n    pub fn realize(&self) -> Result<RealizedTensor> {\n        let graph = self.graph.lock().unwrap();\n\n        // Fast path: already-backed leaf.\n        let root_node = graph.node(self.id);\n        if let Some(ref buffer) = root_node.buffer {\n            let data = buffer.as_f32().to_vec();\n            let shape = root_node.shape.clone();\n            drop(graph);\n            return Ok(RealizedTensor::new(data, shape));\n        }\n\n        // Only optimize pure elementwise roots for now.\n        let optimize_safe = is_optimize_safe(&graph, self.id);\n        let (exec_graph, exec_root) = if optimize_safe {\n            optimize::optimize(&graph, self.id)\n        } else {\n            (\n                clone_reachable_subgraph(&graph, self.id),\n                map_root_to_zero(&graph, self.id),\n            )\n        };\n        drop(graph);\n\n        let graph = &exec_graph;\n        let root = exec_root;\n\n        let root_node = graph.node(root);\n        if let Some(ref buffer) = root_node.buffer {\n            return Ok(RealizedTensor::new(\n                buffer.as_f32().to_vec(),\n                root_node.shape.clone(),\n            ));\n        }\n        if let Op::Const(val) = root_node.op {\n            let numel: usize = self.shape.iter().product();\n            return Ok(RealizedTensor::new(vec![val; numel], self.shape.clone()));\n        }\n\n        let schedule = build_schedule(graph, root);\n        let mut realized: std::collections::HashMap<NodeId, Vec<f32>> =\n            std::collections::HashMap::new();\n\n        for item in &schedule {\n            match item {\n                ScheduleItem::Fused(kernel) => {\n                    let compiled = compile_kernel(graph, kernel)?;\n\n                    let input_ptrs: Vec<*const f32> = kernel\n                        .input_buffers\n                        .iter()\n                        .map(|&buf_id| {\n                            let node = graph.node(buf_id);\n                            if let Some(ref buffer) = node.buffer {\n                                buffer.as_f32_ptr()\n                            } else if let Some(data) = realized.get(&buf_id) {\n                                data.as_ptr()\n                            } else {\n                                panic!(\"Input buffer {:?} not realized and has no data\", buf_id);\n                            }\n                        })\n                        .collect();\n\n                    let mut output = vec![0.0f32; kernel.numel];\n                    unsafe {\n                        compiled.execute(&input_ptrs, output.as_mut_ptr(), kernel.numel);\n                    }\n                    realized.insert(kernel.root, output);\n                }\n                ScheduleItem::Shape(shape_item) => {\n                    let input_node = graph.node(shape_item.input);\n                    let input_shape = input_node.shape.clone();\n\n                    let input_data = if let Some(data) = realized.get(&shape_item.input) {\n                        data.clone()\n                    } else if let Some(ref buf) = input_node.buffer {\n                        buf.as_f32().to_vec()\n                    } else {\n                        bail!(\"Shape op input {:?} is not realized\", shape_item.input);\n                    };\n\n                    let output = execute_shape_op(\n                        &shape_item.op,\n                        &input_data,\n                        &input_shape,\n                        &shape_item.shape,\n                    )?;\n                    realized.insert(shape_item.root, output);\n                }\n                ScheduleItem::Reduce(reduce_item) => {\n                    let input_node = graph.node(reduce_item.input);\n                    let input_shape = input_node.shape.clone();\n\n                    let input_data = if let Some(data) = realized.get(&reduce_item.input) {\n                        data.clone()\n                    } else if let Some(ref buf) = input_node.buffer {\n                        buf.as_f32().to_vec()\n                    } else {\n                        bail!(\"Reduce op input {:?} is not realized\", reduce_item.input);\n                    };\n\n                    let output = execute_reduce_op(\n                        &reduce_item.op,\n                        &input_data,\n                        &input_shape,\n                        &reduce_item.shape,\n                    )?;\n                    realized.insert(reduce_item.root, output);\n                }\n            }\n        }\n\n        let data = realized\n            .remove(&root)\n            .ok_or_else(|| anyhow!(\"Root node was not realized\"))?;\n\n        Ok(RealizedTensor::new(data, graph.node(root).shape.clone()))\n    }\n\n    pub fn realize_eager(&self) -> Result<EagerTensor<f32>> {\n        self.realize()?.to_eager()\n    }\n}\n\n// -------- shape-op execution helpers --------\n\nfn execute_shape_op(\n    op: &Op,\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n) -> Result<Vec<f32>> {\n    match op {\n        Op::Reshape(_) => execute_reshape(input_data, input_shape, output_shape),\n        Op::Permute(perm) => execute_permute(input_data, input_shape, output_shape, perm),\n        Op::Transpose(d1, d2) => execute_transpose(input_data, input_shape, output_shape, *d1, *d2),\n        Op::Expand(_) => execute_expand(input_data, input_shape, output_shape),\n        Op::Slice(ranges) => execute_slice(input_data, input_shape, output_shape, ranges),\n        Op::Flip(flips) => execute_flip(input_data, input_shape, output_shape, flips),\n        Op::Squeeze => execute_reshape(input_data, input_shape, output_shape),\n        Op::Unsqueeze(_) => execute_reshape(input_data, input_shape, output_shape),\n        Op::Pad(constant, padding) => {\n            execute_pad(input_data, input_shape, output_shape, *constant, padding)\n        }\n        _ => bail!(\"execute_shape_op called with non-shape op: {:?}\", op),\n    }\n}\n\nfn execute_reduce_op(\n    op: &Op,\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n) -> Result<Vec<f32>> {\n    match op {\n        Op::Sum(dimensions, keepdims) => execute_reduce(\n            input_data,\n            input_shape,\n            output_shape,\n            dimensions,\n            *keepdims,\n            |slice| slice.iter().copied().sum(),\n        ),\n        Op::Prod(dimensions, keepdims) => execute_reduce(\n            input_data,\n            input_shape,\n            output_shape,\n            dimensions,\n            *keepdims,\n            |slice| slice.iter().copied().product(),\n        ),\n        Op::Max(dimensions, keepdims) => execute_reduce(\n            input_data,\n            input_shape,\n            output_shape,\n            dimensions,\n            *keepdims,\n            |slice| {\n                slice\n                    .iter()\n                    .copied()\n                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))\n                    .unwrap_or(0.0)\n            },\n        ),\n        Op::Min(dimensions, keepdims) => execute_reduce(\n            input_data,\n            input_shape,\n            output_shape,\n            dimensions,\n            *keepdims,\n            |slice| {\n                slice\n                    .iter()\n                    .copied()\n                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))\n                    .unwrap_or(0.0)\n            },\n        ),\n        _ => bail!(\"execute_reduce_op called with non-reduce op: {:?}\", op),\n    }\n}\n\nfn execute_reduce(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n    dimensions: &[usize],\n    keepdims: bool,\n    reducer: impl Fn(&[f32]) -> f32,\n) -> Result<Vec<f32>> {\n    validate_reduce_dims(input_shape, dimensions)?;\n\n    let expected_shape = reduced_shape(input_shape, dimensions, keepdims)?;\n    if expected_shape != output_shape {\n        bail!(\n            \"reduce output shape mismatch: expected {:?}, got {:?}\",\n            expected_shape,\n            output_shape\n        );\n    }\n\n    let mut out = Vec::with_capacity(output_shape.iter().product());\n\n    for out_idx in Indexer::new(output_shape) {\n        let mut fixed = vec![None; input_shape.len()];\n\n        if keepdims {\n            for d in 0..input_shape.len() {\n                if !dimensions.contains(&d) {\n                    fixed[d] = Some(out_idx[d]);\n                }\n            }\n        } else {\n            let mut out_pos = 0usize;\n            for d in 0..input_shape.len() {\n                if !dimensions.contains(&d) {\n                    fixed[d] = Some(out_idx[out_pos]);\n                    out_pos += 1;\n                }\n            }\n        }\n\n        let mut reduced_values = Vec::new();\n        collect_reduce_values(\n            input_data,\n            input_shape,\n            dimensions,\n            &fixed,\n            0,\n            &mut vec![0; input_shape.len()],\n            &mut reduced_values,\n        );\n\n        out.push(reducer(&reduced_values));\n    }\n\n    Ok(out)\n}\n\nfn collect_reduce_values(\n    input_data: &[f32],\n    input_shape: &[usize],\n    dimensions: &[usize],\n    fixed: &[Option<usize>],\n    dim: usize,\n    current: &mut Vec<usize>,\n    values: &mut Vec<f32>,\n) {\n    if dim == input_shape.len() {\n        let off = idx_to_offset(current, input_shape);\n        values.push(input_data[off]);\n        return;\n    }\n\n    if dimensions.contains(&dim) {\n        for i in 0..input_shape[dim] {\n            current[dim] = i;\n            collect_reduce_values(\n                input_data,\n                input_shape,\n                dimensions,\n                fixed,\n                dim + 1,\n                current,\n                values,\n            );\n        }\n    } else if let Some(idx) = fixed[dim] {\n        current[dim] = idx;\n        collect_reduce_values(\n            input_data,\n            input_shape,\n            dimensions,\n            fixed,\n            dim + 1,\n            current,\n            values,\n        );\n    }\n}\n\nfn reduced_shape(shape: &[usize], dimensions: &[usize], keepdims: bool) -> Result<Vec<usize>> {\n    validate_reduce_dims(shape, dimensions)?;\n\n    let mut out = Vec::new();\n    for (d, &size) in shape.iter().enumerate() {\n        if dimensions.contains(&d) {\n            if keepdims {\n                out.push(1);\n            }\n        } else {\n            out.push(size);\n        }\n    }\n\n    if out.is_empty() {\n        out.push(1);\n    }\n\n    Ok(out)\n}\n\nfn validate_reduce_dims(shape: &[usize], dimensions: &[usize]) -> Result<()> {\n    let mut seen = vec![false; shape.len()];\n    for &d in dimensions {\n        if d >= shape.len() {\n            bail!(\n                \"reduce dimension {} out of bounds for rank {}\",\n                d,\n                shape.len()\n            );\n        }\n        if seen[d] {\n            bail!(\"duplicate reduce dimension {}\", d);\n        }\n        seen[d] = true;\n    }\n    Ok(())\n}\n\nfn execute_reshape(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n) -> Result<Vec<f32>> {\n    let in_numel: usize = input_shape.iter().product();\n    let out_numel: usize = output_shape.iter().product();\n    if in_numel != out_numel || input_data.len() != in_numel {\n        bail!(\n            \"invalid reshape numel: input {:?}, output {:?}\",\n            input_shape,\n            output_shape\n        );\n    }\n    Ok(input_data.to_vec())\n}\n\nfn execute_permute(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n    perm: &[usize],\n) -> Result<Vec<f32>> {\n    if perm.len() != input_shape.len() || output_shape.len() != input_shape.len() {\n        bail!(\"invalid permute rank\");\n    }\n    let mut out = vec![0.0f32; output_shape.iter().product()];\n\n    for out_idx in Indexer::new(output_shape) {\n        let mut in_idx = vec![0usize; input_shape.len()];\n        for (out_dim, &in_dim) in perm.iter().enumerate() {\n            in_idx[in_dim] = out_idx[out_dim];\n        }\n        let in_off = idx_to_offset(&in_idx, input_shape);\n        let out_off = idx_to_offset(&out_idx, output_shape);\n        out[out_off] = input_data[in_off];\n    }\n\n    Ok(out)\n}\n\nfn execute_transpose(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n    dim_1: usize,\n    dim_2: usize,\n) -> Result<Vec<f32>> {\n    if dim_1 >= input_shape.len()\n        || dim_2 >= input_shape.len()\n        || input_shape.len() != output_shape.len()\n    {\n        bail!(\"invalid transpose dims\");\n    }\n\n    let mut perm: Vec<usize> = (0..input_shape.len()).collect();\n    perm.swap(dim_1, dim_2);\n    execute_permute(input_data, input_shape, output_shape, &perm)\n}\n\nfn execute_expand(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n) -> Result<Vec<f32>> {\n    if input_shape.len() != output_shape.len() {\n        bail!(\"expand rank mismatch\");\n    }\n\n    let mut out = vec![0.0f32; output_shape.iter().product()];\n    for out_idx in Indexer::new(output_shape) {\n        let mut in_idx = vec![0usize; input_shape.len()];\n        for d in 0..input_shape.len() {\n            if input_shape[d] == output_shape[d] {\n                in_idx[d] = out_idx[d];\n            } else if input_shape[d] == 1 {\n                in_idx[d] = 0;\n            } else {\n                bail!(\n                    \"cannot expand dim {} from {} to {}\",\n                    d,\n                    input_shape[d],\n                    output_shape[d]\n                );\n"
}
```

**Tool Call: Read file `src/core/lazy/render.rs` (lines 1-320)**
Status: Completed

```json
{
  "Text": "use std::collections::HashSet;\n\nuse super::graph::{Graph, NodeId, Op};\nuse super::schedule::{build_schedule, ScheduleItem};\n\nfn op_label(op: &Op) -> &'static str {\n    match op {\n        Op::Const(_) => \"Const\",\n        Op::Load => \"Load\",\n        Op::Add => \"Add\",\n        Op::Sub => \"Sub\",\n        Op::Mul => \"Mul\",\n        Op::Div => \"Div\",\n        Op::Exp => \"Exp\",\n        Op::Ln => \"Ln\",\n        Op::Sqrt => \"Sqrt\",\n        Op::Neg => \"Neg\",\n        Op::Reshape(_) => \"Reshape\",\n        Op::Permute(_) => \"Permute\",\n        Op::Transpose(_, _) => \"Transpose\",\n        Op::Expand(_) => \"Expand\",\n        Op::Slice(_) => \"Slice\",\n        Op::Flip(_) => \"Flip\",\n        Op::Squeeze => \"Squeeze\",\n        Op::Unsqueeze(_) => \"Unsqueeze\",\n        Op::Pad(_, _) => \"Pad\",\n        Op::Sum(_, _) => \"Sum\",\n        Op::Prod(_, _) => \"Prod\",\n        Op::Max(_, _) => \"Max\",\n        Op::Min(_, _) => \"Min\",\n    }\n}\n\nfn node_label(graph: &Graph, id: NodeId) -> String {\n    let node = graph.node(id);\n    let shape_str = format!(\"{:?}\", node.shape);\n    match &node.op {\n        Op::Const(v) => format!(\"Const({v})\\\\n{shape_str}\"),\n        Op::Load => {\n            let preview = node\n                .buffer\n                .as_ref()\n                .map(|b| {\n                    let data = b.as_f32();\n                    if data.len() <= 4 {\n                        format!(\"{:?}\", data)\n                    } else {\n                        format!(\"[{}, {}, ... {}]\", data[0], data[1], data[data.len() - 1])\n                    }\n                })\n                .unwrap_or_default();\n            format!(\"Load\\\\n{preview}\\\\n{shape_str}\")\n        }\n        other => format!(\"{}\\\\n{shape_str}\", op_label(other)),\n    }\n}\n\n/// Render the raw DAG (before fusion) rooted at `root` as Mermaid flowchart code.\npub fn render_dag(graph: &Graph, root: NodeId) -> String {\n    let mut visited = HashSet::new();\n    let mut lines = vec![\"flowchart BT\".to_string()];\n    render_dag_node(graph, root, &mut visited, &mut lines);\n    lines.join(\"\\n\")\n}\n\nfn render_dag_node(\n    graph: &Graph,\n    id: NodeId,\n    visited: &mut HashSet<usize>,\n    lines: &mut Vec<String>,\n) {\n    if !visited.insert(id.0) {\n        return;\n    }\n    let node = graph.node(id);\n    let label = node_label(graph, id);\n    lines.push(format!(\"    N{}[\\\"{}\\\"]\", id.0, label));\n\n    for &input_id in &node.inputs {\n        render_dag_node(graph, input_id, visited, lines);\n        lines.push(format!(\"    N{} --> N{}\", input_id.0, id.0));\n    }\n}\n\n/// Render the DAG after scheduling, with fused kernels and shape-op barriers grouped.\npub fn render_fused_dag(graph: &Graph, root: NodeId) -> String {\n    let schedule = build_schedule(graph, root);\n\n    // Collect which nodes belong to which fused kernel.\n    let mut node_to_kernel: std::collections::HashMap<usize, usize> =\n        std::collections::HashMap::new();\n    // Collect shape/reduce barrier roots to render explicitly.\n    let mut shape_barriers: Vec<(usize, usize)> = Vec::new(); // (node_id, schedule_index)\n    let mut reduce_barriers: Vec<(usize, usize)> = Vec::new(); // (node_id, schedule_index)\n\n    for (si, item) in schedule.iter().enumerate() {\n        match item {\n            ScheduleItem::Fused(kernel) => {\n                node_to_kernel.insert(kernel.root.0, si);\n                collect_fused_nodes(\n                    graph,\n                    kernel.root,\n                    &kernel.input_buffers,\n                    &mut node_to_kernel,\n                    si,\n                );\n            }\n            ScheduleItem::Shape(shape_item) => {\n                shape_barriers.push((shape_item.root.0, si));\n            }\n            ScheduleItem::Reduce(reduce_item) => {\n                reduce_barriers.push((reduce_item.root.0, si));\n            }\n        }\n    }\n\n    let mut lines = vec![\"flowchart BT\".to_string()];\n\n    // Render scheduled subgraphs in schedule order.\n    let mut fused_idx = 0usize;\n    let mut shape_idx = 0usize;\n    let mut reduce_idx = 0usize;\n\n    for (si, item) in schedule.iter().enumerate() {\n        match item {\n            ScheduleItem::Fused(_) => {\n                let members: Vec<usize> = node_to_kernel\n                    .iter()\n                    .filter(|(_, &k)| k == si)\n                    .map(|(&n, _)| n)\n                    .collect();\n\n                if members.is_empty() {\n                    continue;\n                }\n\n                lines.push(format!(\n                    \"    subgraph Kernel_{} [\\\"Fused Kernel {}\\\"]\",\n                    fused_idx, fused_idx\n                ));\n                fused_idx += 1;\n\n                for &nid in &members {\n                    let label = node_label(graph, NodeId(nid));\n                    lines.push(format!(\"        N{}[\\\"{}\\\"]\", nid, label));\n                }\n                lines.push(\"    end\".to_string());\n            }\n            ScheduleItem::Shape(shape_item) => {\n                lines.push(format!(\n                    \"    subgraph Shape_{} [\\\"Shape Op {}\\\"]\",\n                    shape_idx,\n                    op_label(&shape_item.op)\n                ));\n                shape_idx += 1;\n\n                let nid = shape_item.root.0;\n                let label = node_label(graph, NodeId(nid));\n                lines.push(format!(\"        N{}[\\\"{}\\\"]\", nid, label));\n                lines.push(\"    end\".to_string());\n            }\n            ScheduleItem::Reduce(reduce_item) => {\n                lines.push(format!(\n                    \"    subgraph Reduce_{} [\\\"Reduce Op {}\\\"]\",\n                    reduce_idx,\n                    op_label(&reduce_item.op)\n                ));\n                reduce_idx += 1;\n\n                let nid = reduce_item.root.0;\n                let label = node_label(graph, NodeId(nid));\n                lines.push(format!(\"        N{}[\\\"{}\\\"]\", nid, label));\n                lines.push(\"    end\".to_string());\n            }\n        }\n    }\n\n    // Render non-scheduled leaf nodes.\n    let mut scheduled_nodes: HashSet<usize> = node_to_kernel.keys().copied().collect();\n    for (nid, _) in &shape_barriers {\n        scheduled_nodes.insert(*nid);\n    }\n    for (nid, _) in &reduce_barriers {\n        scheduled_nodes.insert(*nid);\n    }\n\n    let mut visited = HashSet::new();\n    render_leaf_nodes(graph, root, &scheduled_nodes, &mut visited, &mut lines);\n\n    // Render edges.\n    let mut edge_visited = HashSet::new();\n    render_edges(graph, root, &mut edge_visited, &mut lines);\n\n    lines.join(\"\\n\")\n}\n\nfn collect_fused_nodes(\n    graph: &Graph,\n    id: NodeId,\n    input_buffers: &[NodeId],\n    node_to_kernel: &mut std::collections::HashMap<usize, usize>,\n    ki: usize,\n) {\n    let node = graph.node(id);\n    for &input_id in &node.inputs {\n        if input_buffers.contains(&input_id) {\n            continue;\n        }\n        // This node is inlined into the kernel.\n        node_to_kernel.insert(input_id.0, ki);\n        collect_fused_nodes(graph, input_id, input_buffers, node_to_kernel, ki);\n    }\n}\n\nfn render_leaf_nodes(\n    graph: &Graph,\n    id: NodeId,\n    scheduled_nodes: &HashSet<usize>,\n    visited: &mut HashSet<usize>,\n    lines: &mut Vec<String>,\n) {\n    if !visited.insert(id.0) {\n        return;\n    }\n    if !scheduled_nodes.contains(&id.0) {\n        let label = node_label(graph, id);\n        lines.push(format!(\"    N{}[\\\"{}\\\"]\", id.0, label));\n    }\n    let node = graph.node(id);\n    for &input_id in &node.inputs {\n        render_leaf_nodes(graph, input_id, scheduled_nodes, visited, lines);\n    }\n}\n\nfn render_edges(graph: &Graph, id: NodeId, visited: &mut HashSet<usize>, lines: &mut Vec<String>) {\n    if !visited.insert(id.0) {\n        return;\n    }\n    let node = graph.node(id);\n    for &input_id in &node.inputs {\n        render_edges(graph, input_id, visited, lines);\n        lines.push(format!(\"    N{} --> N{}\", input_id.0, id.0));\n    }\n}\n"
}
```

**Tool Call: Read file `src/core/lazy/tests.rs` (lines 1-420)**
Status: Completed

```json
{
  "Text": "#[cfg(test)]\nmod lazy_tests {\n    use crate::{Context, EagerTensor, Tensor};\n    use anyhow::Result;\n\n    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {\n        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)\n    }\n\n    #[test]\n    fn realize_leaf() -> Result<()> {\n        let cx = Context::new();\n\n        let data = vec![1.0, 2.0, 3.0, 4.0];\n        let dt = Tensor::from_slice(&cx, &data, vec![4]);\n        let result = dt.realize()?;\n\n        assert_eq!(result.data(), &data);\n        assert_eq!(result.shape(), &[4]);\n        Ok(())\n    }\n\n    #[test]\n    fn add_two_tensors() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);\n        let c = &a + &b;\n        let result = c.realize()?;\n\n        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn mul_two_tensors() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[10.0, 10.0, 10.0, 10.0], vec![4]);\n        let c = &a * &b;\n        let result = c.realize()?;\n\n        assert_eq!(result.data(), &[20.0, 30.0, 40.0, 50.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn sub_and_div() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);\n\n        let sub_result = (&a - &b).realize()?;\n        assert_eq!(sub_result.data(), &[9.0, 18.0, 27.0, 36.0]);\n\n        let div_result = (&a / &b).realize()?;\n        assert_eq!(div_result.data(), &[10.0, 10.0, 10.0, 10.0]);\n\n        Ok(())\n    }\n\n    #[test]\n    fn fused_add_mul() -> Result<()> {\n        let cx = Context::new();\n\n        // (a + b) * a should produce ONE fused kernel.\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);\n        let c = &(&a + &b) * &a;\n        let result = c.realize()?;\n\n        // (1+10)*1=11, (2+20)*2=44, (3+30)*3=99, (4+40)*4=176\n        assert_eq!(result.data(), &[11.0, 44.0, 99.0, 176.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn unary_neg() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0, -4.0], vec![4]);\n        let result = a.neg().realize()?;\n\n        assert_eq!(result.data(), &[-1.0, 2.0, -3.0, 4.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn unary_exp() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[0.0, 1.0, 2.0], vec![3]);\n        let result = a.exp().realize()?;\n\n        let expected: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();\n        assert!(approx_eq(result.data(), &expected, 1e-5));\n        Ok(())\n    }\n\n    #[test]\n    fn unary_ln() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.718_281_7, 7.389056], vec![3]);\n        let result = a.ln().realize()?;\n\n        let expected: Vec<f32> = [1.0f32, 2.718_281_7, 7.389056]\n            .iter()\n            .map(|x| x.ln())\n            .collect();\n        assert!(approx_eq(result.data(), &expected, 1e-4));\n        Ok(())\n    }\n\n    #[test]\n    fn unary_sqrt() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 4.0, 9.0, 16.0], vec![4]);\n        let result = a.sqrt().realize()?;\n\n        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn chain_fused_unary_binary() -> Result<()> {\n        let cx = Context::new();\n\n        // exp(a) + b — should fuse into one kernel.\n        let a = Tensor::from_slice(&cx, &[0.0, 0.0, 0.0], vec![3]);\n        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        let c = &a.exp() + &b;\n        let result = c.realize()?;\n\n        // exp(0) + 1 = 2, exp(0) + 2 = 3, exp(0) + 3 = 4\n        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn from_eager_tensor() -> Result<()> {\n        let cx = Context::new();\n\n        let eager = EagerTensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;\n        let dt = Tensor::from_tensor(&cx, &eager);\n        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);\n        let c = &dt + &b;\n        let result = c.realize()?;\n\n        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);\n        assert_eq!(result.shape(), &[2, 2]);\n        Ok(())\n    }\n\n    #[test]\n    fn larger_tensor() -> Result<()> {\n        let cx = Context::new();\n\n        let n = 1024;\n        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();\n        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();\n\n        let a = Tensor::from_slice(&cx, &a_data, vec![n]);\n        let b = Tensor::from_slice(&cx, &b_data, vec![n]);\n        let c = &a + &b;\n        let result = c.realize()?;\n\n        let expected: Vec<f32> = (0..n).map(|_| n as f32).collect();\n        assert_eq!(result.data(), expected.as_slice());\n        Ok(())\n    }\n\n    #[test]\n    fn deep_fusion_chain() -> Result<()> {\n        let cx = Context::new();\n\n        // a * b + a - b should all fuse into one kernel.\n        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);\n\n        let c = &(&(&a * &b) + &a) - &b;\n        let result = c.realize()?;\n\n        // (2*1 + 2 - 1) = 3, (3*1 + 3 - 1) = 5, (4*1 + 4 - 1) = 7, (5*1 + 5 - 1) = 9\n        assert_eq!(result.data(), &[3.0, 5.0, 7.0, 9.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn constant_tensor() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        let c = Tensor::constant(&cx, 10.0, vec![3]);\n        let result = (&a + &c).realize()?;\n\n        assert_eq!(result.data(), &[11.0, 12.0, 13.0]);\n        Ok(())\n    }\n\n    // --- egglog optimization tests ---\n\n    #[test]\n    fn egglog_add_zero_identity() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        let zero = Tensor::constant(&cx, 0.0, vec![3]);\n        // a + 0 should be optimized to just a.\n        let result = (&a + &zero).realize()?;\n        assert_eq!(result.data(), &[1.0, 2.0, 3.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_mul_one_identity() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);\n        let one = Tensor::constant(&cx, 1.0, vec![3]);\n        // a * 1 should be optimized to just a.\n        let result = (&a * &one).realize()?;\n        assert_eq!(result.data(), &[5.0, 10.0, 15.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_mul_zero() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);\n        let zero = Tensor::constant(&cx, 0.0, vec![3]);\n        // a * 0 should be optimized to 0.\n        let result = (&a * &zero).realize()?;\n        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_double_neg() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0], vec![3]);\n        // neg(neg(a)) should be optimized to just a.\n        let result = a.neg().neg().realize()?;\n        assert_eq!(result.data(), &[1.0, -2.0, 3.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_exp_ln_inverse() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        // exp(ln(a)) should be optimized to just a.\n        let result = a.ln().exp().realize()?;\n        assert!(approx_eq(result.data(), &[1.0, 2.0, 3.0], 1e-5));\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_sub_self() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);\n        // a - a should be optimized to 0.\n        let result = (&a - &a).realize()?;\n        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn realized_to_eager() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.realize()?;\n        let e = r.to_eager()?;\n\n        assert_eq!(e.data().as_ref(), &[1.0, 2.0, 3.0, 4.0]);\n        assert_eq!(e.sizes(), &[2, 2]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_sum_dims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.sum_dims(vec![1], false)?.realize()?;\n\n        assert_eq!(r.data(), &[3.0, 7.0]);\n        assert_eq!(r.shape(), &[2]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_product_dims_keepdims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.product_dims(vec![0], true)?.realize()?;\n\n        assert_eq!(r.data(), &[3.0, 8.0]);\n        assert_eq!(r.shape(), &[1, 2]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_max_dims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);\n        let r = a.max_dims(vec![0], false)?.realize()?;\n\n        assert_eq!(r.data(), &[4.0, 9.0, 6.0]);\n        assert_eq!(r.shape(), &[3]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_min_dims_keepdims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);\n        let r = a.min_dims(vec![1], true)?.realize()?;\n\n        assert_eq!(r.data(), &[1.0, 4.0]);\n        assert_eq!(r.shape(), &[2, 1]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_sum_all() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.sum()?.realize()?;\n\n        assert_eq!(r.data(), &[10.0]);\n        assert_eq!(r.shape(), &[1, 1]);\n        Ok(())\n    }\n}\n"
}
```

**Tool Call: Read file `src/lib.rs` (lines 1-200)**
Status: Completed

```json
{
  "Text": "/*!\n```console\n               __\n__   _____  /\\ \\ \\_   _ _ __ ___\n\\ \\ / / _ \\/  \\/ / | | | '_ ` _ \\\n \\ V /  __/ /\\  /| |_| | | | | | |\n  \\_/ \\___\\_\\ \\/  \\__,_|_| |_| |_|\n```\n\nVectorized _N_-dimensional numerical arrays.\n*/\n\nmod core;\n\npub use core::eager::ops::conv;\npub use core::eager::EagerTensor;\npub use core::lazy::{Context, RealizedTensor, Tensor};\n\n// Temporary compatibility re-exports while migrating call sites.\npub use core::eager::ETensor;\npub use core::lazy::LTensor;\n"
}
```

**Tool Call: Read file `examples/dag_render.rs` (lines 1-220)**
Status: Completed

```json
{
  "Text": "use venum::{Context, Tensor};\n\nfn main() {\n    let cx = Context::new();\n\n    let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);\n    let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);\n\n    // (a * b + a) - b\n    let c = &(&(&a * &b) + &a) - &b;\n\n    println!(\"=== RAW DAG ===\");\n    println!(\"{}\", c.render_dag());\n    println!();\n    println!(\"=== FUSED DAG ===\");\n    println!(\"{}\", c.render_fused_dag());\n}\n"
}
```

**Tool Call: Read file `src/core/lazy/tensor.rs` (lines 901-1400)**
Status: Completed

```json
{
  "Text": "            }\n        }\n        let in_off = idx_to_offset(&in_idx, input_shape);\n        let out_off = idx_to_offset(&out_idx, output_shape);\n        out[out_off] = input_data[in_off];\n    }\n\n    Ok(out)\n}\n\nfn execute_slice(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n    ranges: &[(usize, usize)],\n) -> Result<Vec<f32>> {\n    if ranges.len() != input_shape.len() || output_shape.len() != input_shape.len() {\n        bail!(\"slice rank mismatch\");\n    }\n\n    let mut out = vec![0.0f32; output_shape.iter().product()];\n    for out_idx in Indexer::new(output_shape) {\n        let mut in_idx = vec![0usize; input_shape.len()];\n        for d in 0..input_shape.len() {\n            let (start, end_raw) = ranges[d];\n            let end = if end_raw == 0 {\n                input_shape[d]\n            } else {\n                end_raw\n            };\n            if start > end || end > input_shape[d] {\n                bail!(\"invalid slice range {:?} for dim {}\", ranges[d], d);\n            }\n            in_idx[d] = start + out_idx[d];\n        }\n        let in_off = idx_to_offset(&in_idx, input_shape);\n        let out_off = idx_to_offset(&out_idx, output_shape);\n        out[out_off] = input_data[in_off];\n    }\n\n    Ok(out)\n}\n\nfn execute_flip(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n    flips: &[usize],\n) -> Result<Vec<f32>> {\n    if input_shape != output_shape {\n        bail!(\"flip must preserve shape\");\n    }\n\n    let mut flip_mask = vec![false; input_shape.len()];\n    for &d in flips {\n        if d >= input_shape.len() {\n            bail!(\"flip dim {} out of range\", d);\n        }\n        flip_mask[d] = true;\n    }\n\n    let mut out = vec![0.0f32; output_shape.iter().product()];\n    for out_idx in Indexer::new(output_shape) {\n        let mut in_idx = out_idx.clone();\n        for d in 0..in_idx.len() {\n            if flip_mask[d] {\n                in_idx[d] = input_shape[d] - 1 - in_idx[d];\n            }\n        }\n\n        let in_off = idx_to_offset(&in_idx, input_shape);\n        let out_off = idx_to_offset(&out_idx, output_shape);\n        out[out_off] = input_data[in_off];\n    }\n\n    Ok(out)\n}\n\nfn execute_pad(\n    input_data: &[f32],\n    input_shape: &[usize],\n    output_shape: &[usize],\n    constant: f32,\n    padding: &[(usize, usize)],\n) -> Result<Vec<f32>> {\n    if input_shape.len() != output_shape.len() {\n        bail!(\"pad rank mismatch\");\n    }\n\n    let mut pad = padding.to_vec();\n    pad.resize(input_shape.len(), (0, 0));\n\n    let mut expected = Vec::with_capacity(input_shape.len());\n    for d in 0..input_shape.len() {\n        expected.push(pad[d].0 + input_shape[d] + pad[d].1);\n    }\n    if expected != output_shape {\n        bail!(\n            \"pad output shape mismatch: expected {:?}, got {:?}\",\n            expected,\n            output_shape\n        );\n    }\n\n    let mut out = vec![constant; output_shape.iter().product()];\n    for in_idx in Indexer::new(input_shape) {\n        let mut out_idx = vec![0usize; input_shape.len()];\n        for d in 0..input_shape.len() {\n            out_idx[d] = pad[d].0 + in_idx[d];\n        }\n\n        let in_off = idx_to_offset(&in_idx, input_shape);\n        let out_off = idx_to_offset(&out_idx, output_shape);\n        out[out_off] = input_data[in_off];\n    }\n\n    Ok(out)\n}\n\nfn idx_to_offset(index: &[usize], shape: &[usize]) -> usize {\n    let mut stride = 1usize;\n    let mut off = 0usize;\n    for d in (0..shape.len()).rev() {\n        off += index[d] * stride;\n        stride *= shape[d];\n    }\n    off\n}\n\n// -------- graph import helpers --------\n\nfn import_subgraph(src_graph: &Graph, src_id: NodeId, dst_graph: &mut Graph) -> NodeId {\n    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();\n\n    let mut buffer_map: std::collections::HashMap<*const Vec<f32>, NodeId> =\n        std::collections::HashMap::new();\n    for (i, node) in dst_graph.nodes.iter().enumerate() {\n        if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {\n            buffer_map.insert(Arc::as_ptr(arc), NodeId(i));\n        }\n    }\n\n    import_node(src_graph, src_id, dst_graph, &mut id_map, &buffer_map)\n}\n\nfn import_node(\n    src_graph: &Graph,\n    src_id: NodeId,\n    dst_graph: &mut Graph,\n    id_map: &mut std::collections::HashMap<NodeId, NodeId>,\n    buffer_map: &std::collections::HashMap<*const Vec<f32>, NodeId>,\n) -> NodeId {\n    if let Some(&mapped) = id_map.get(&src_id) {\n        return mapped;\n    }\n\n    let node = src_graph.node(src_id);\n\n    if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {\n        if let Some(&existing_id) = buffer_map.get(&Arc::as_ptr(arc)) {\n            id_map.insert(src_id, existing_id);\n            return existing_id;\n        }\n    }\n\n    let new_inputs: Vec<NodeId> = node\n        .inputs\n        .iter()\n        .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map, buffer_map))\n        .collect();\n\n    let new_id = dst_graph.add_node(super::graph::Node {\n        op: node.op.clone(),\n        inputs: new_inputs,\n        shape: node.shape.clone(),\n        dtype: node.dtype,\n        buffer: node.buffer.clone(),\n    });\n\n    id_map.insert(src_id, new_id);\n    new_id\n}\n\n// -------- optimization helpers --------\n\nfn is_optimize_safe(graph: &Graph, root: NodeId) -> bool {\n    fn dfs(graph: &Graph, id: NodeId, seen: &mut std::collections::HashSet<NodeId>) -> bool {\n        if !seen.insert(id) {\n            return true;\n        }\n        let node = graph.node(id);\n        let here_ok = matches!(\n            node.op,\n            Op::Load\n                | Op::Const(_)\n                | Op::Add\n                | Op::Sub\n                | Op::Mul\n                | Op::Div\n                | Op::Exp\n                | Op::Ln\n                | Op::Sqrt\n                | Op::Neg\n        );\n        if !here_ok {\n            return false;\n        }\n        node.inputs.iter().all(|&inp| dfs(graph, inp, seen))\n    }\n\n    let mut seen = std::collections::HashSet::new();\n    dfs(graph, root, &mut seen)\n}\n\nfn clone_reachable_subgraph(src: &Graph, root: NodeId) -> Graph {\n    let mut dst = Graph::new();\n    let mut id_map = std::collections::HashMap::new();\n    let buffer_map = std::collections::HashMap::new();\n    let _ = import_node(src, root, &mut dst, &mut id_map, &buffer_map);\n    dst\n}\n\nfn map_root_to_zero(_src: &Graph, _root: NodeId) -> NodeId {\n    // `clone_reachable_subgraph` always inserts root first due to recursion memoization\n    // when called directly on root.\n    NodeId(0)\n}\n\n// -------- operator overloads --------\n\nimpl Add for &Tensor {\n    type Output = Tensor;\n    fn add(self, rhs: &Tensor) -> Tensor {\n        self.binary_op(rhs, Op::Add)\n    }\n}\n\nimpl Sub for &Tensor {\n    type Output = Tensor;\n    fn sub(self, rhs: &Tensor) -> Tensor {\n        self.binary_op(rhs, Op::Sub)\n    }\n}\n\nimpl Mul for &Tensor {\n    type Output = Tensor;\n    fn mul(self, rhs: &Tensor) -> Tensor {\n        self.binary_op(rhs, Op::Mul)\n    }\n}\n\nimpl Div for &Tensor {\n    type Output = Tensor;\n    fn div(self, rhs: &Tensor) -> Tensor {\n        self.binary_op(rhs, Op::Div)\n    }\n}\n\nimpl Add for Tensor {\n    type Output = Tensor;\n    fn add(self, rhs: Tensor) -> Tensor {\n        self.binary_op(&rhs, Op::Add)\n    }\n}\n\nimpl Sub for Tensor {\n    type Output = Tensor;\n    fn sub(self, rhs: Tensor) -> Tensor {\n        self.binary_op(&rhs, Op::Sub)\n    }\n}\n\nimpl Mul for Tensor {\n    type Output = Tensor;\n    fn mul(self, rhs: Tensor) -> Tensor {\n        self.binary_op(&rhs, Op::Mul)\n    }\n}\n\nimpl Div for Tensor {\n    type Output = Tensor;\n    fn div(self, rhs: Tensor) -> Tensor {\n        self.binary_op(&rhs, Op::Div)\n    }\n}\n"
}
```

**Tool Call: Search files for regex `is_optimize_safe|clone_reachable_subgraph|map_root_to_zero|import_subgraph` (case-sensitive)**
Status: Completed

Found 7 matches:

## Matches in venum/src/core/lazy/tensor.rs

### impl Tensor › fn binary_op › L109-112
```
        } else {
            let rhs_graph = rhs.graph.lock().unwrap();
            import_subgraph(&rhs_graph, rhs.id, &mut graph)
        };
```

### impl Tensor › pub fn realize › L487-491
```

        // Only optimize pure elementwise roots for now.
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)
```

### impl Tensor › pub fn realize › L493-496
```
            (
                clone_reachable_subgraph(&graph, self.id),
                map_root_to_zero(&graph, self.id),
            )
```

### fn import_subgraph › L1032-1042
```
fn import_subgraph(src_graph: &Graph, src_id: NodeId, dst_graph: &mut Graph) -> NodeId {
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();

    let mut buffer_map: std::collections::HashMap<*const Vec<f32>, NodeId> =
        std::collections::HashMap::new();
    for (i, node) in dst_graph.nodes.iter().enumerate() {
        if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
            buffer_map.insert(Arc::as_ptr(arc), NodeId(i));
        }
    }

```

2 lines remaining in ancestor node. Read the file to see all.

### fn is_optimize_safe › L1086-1096
```
fn is_optimize_safe(graph: &Graph, root: NodeId) -> bool {
    fn dfs(graph: &Graph, id: NodeId, seen: &mut std::collections::HashSet<NodeId>) -> bool {
        if !seen.insert(id) {
            return true;
        }
        let node = graph.node(id);
        let here_ok = matches!(
            node.op,
            Op::Load
                | Op::Const(_)
                | Op::Add
```

17 lines remaining in ancestor node. Read the file to see all.

### fn clone_reachable_subgraph › L1115-1121
```
fn clone_reachable_subgraph(src: &Graph, root: NodeId) -> Graph {
    let mut dst = Graph::new();
    let mut id_map = std::collections::HashMap::new();
    let buffer_map = std::collections::HashMap::new();
    let _ = import_node(src, root, &mut dst, &mut id_map, &buffer_map);
    dst
}
```

### fn map_root_to_zero › L1123-1127
```
fn map_root_to_zero(_src: &Graph, _root: NodeId) -> NodeId {
    // `clone_reachable_subgraph` always inserts root first due to recursion memoization
    // when called directly on root.
    NodeId(0)
}
```


**Tool Call: Read file `src/core/lazy/jit.rs` (lines 1-320)**
Status: Completed

```json
{
  "Text": "use anyhow::Result;\nuse cranelift::prelude::*;\nuse cranelift_codegen::ir::Function;\nuse cranelift_jit::{JITBuilder, JITModule};\nuse cranelift_module::{Linkage, Module};\nuse std::collections::HashMap;\n\nuse super::graph::{Graph, NodeId, Op};\nuse super::schedule::FusedKernel;\n\n/// A compiled kernel ready to execute.\npub struct CompiledKernel {\n    /// Number of input buffer pointers.\n    pub num_inputs: usize,\n    /// The JIT module that owns the compiled code.\n    _module: JITModule,\n    /// Raw function pointer to the compiled kernel.\n    fn_ptr: *const u8,\n}\n\n// Safety: The compiled code is immutable once created and the function pointer\n// is valid for the lifetime of _module.\nunsafe impl Send for CompiledKernel {}\nunsafe impl Sync for CompiledKernel {}\n\n/// Extern \"C\" functions we link into the JIT for math ops.\nextern \"C\" fn jit_expf(x: f32) -> f32 {\n    x.exp()\n}\nextern \"C\" fn jit_logf(x: f32) -> f32 {\n    x.ln()\n}\nextern \"C\" fn jit_sqrtf(x: f32) -> f32 {\n    x.sqrt()\n}\n\nimpl CompiledKernel {\n    /// Execute the kernel with the given input buffer pointers and output pointer.\n    ///\n    /// # Safety\n    /// - All pointers must be valid and point to buffers of at least `numel` f32 elements.\n    /// - `inputs` must have exactly `self.num_inputs` elements.\n    pub unsafe fn execute(&self, inputs: &[*const f32], output: *mut f32, numel: usize) {\n        // ABI: fn(in0: *const f32, in1: *const f32, ..., out: *mut f32, n: u64)\n        match self.num_inputs {\n            0 => {\n                let f: extern \"C\" fn(*mut f32, u64) = std::mem::transmute(self.fn_ptr);\n                f(output, numel as u64);\n            }\n            1 => {\n                let f: extern \"C\" fn(*const f32, *mut f32, u64) = std::mem::transmute(self.fn_ptr);\n                f(inputs[0], output, numel as u64);\n            }\n            2 => {\n                let f: extern \"C\" fn(*const f32, *const f32, *mut f32, u64) =\n                    std::mem::transmute(self.fn_ptr);\n                f(inputs[0], inputs[1], output, numel as u64);\n            }\n            3 => {\n                let f: extern \"C\" fn(*const f32, *const f32, *const f32, *mut f32, u64) =\n                    std::mem::transmute(self.fn_ptr);\n                f(inputs[0], inputs[1], inputs[2], output, numel as u64);\n            }\n            4 => {\n                let f: extern \"C\" fn(\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *mut f32,\n                    u64,\n                ) = std::mem::transmute(self.fn_ptr);\n                f(\n                    inputs[0],\n                    inputs[1],\n                    inputs[2],\n                    inputs[3],\n                    output,\n                    numel as u64,\n                );\n            }\n            5 => {\n                let f: extern \"C\" fn(\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *mut f32,\n                    u64,\n                ) = std::mem::transmute(self.fn_ptr);\n                f(\n                    inputs[0],\n                    inputs[1],\n                    inputs[2],\n                    inputs[3],\n                    inputs[4],\n                    output,\n                    numel as u64,\n                );\n            }\n            6 => {\n                let f: extern \"C\" fn(\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *const f32,\n                    *mut f32,\n                    u64,\n                ) = std::mem::transmute(self.fn_ptr);\n                f(\n                    inputs[0],\n                    inputs[1],\n                    inputs[2],\n                    inputs[3],\n                    inputs[4],\n                    inputs[5],\n                    output,\n                    numel as u64,\n                );\n            }\n            _ => {\n                panic!(\n                    \"CompiledKernel: too many inputs ({}), max 6 supported in dispatch\",\n                    self.num_inputs\n                );\n            }\n        }\n    }\n}\n\n/// Compile a fused elementwise kernel into native code via Cranelift.\npub fn compile_kernel(graph: &Graph, kernel: &FusedKernel) -> Result<CompiledKernel> {\n    let num_inputs = kernel.input_buffers.len();\n\n    // Map each input buffer NodeId to its parameter index.\n    let mut input_index: HashMap<NodeId, usize> = HashMap::new();\n    for (i, &buf_id) in kernel.input_buffers.iter().enumerate() {\n        input_index.insert(buf_id, i);\n    }\n\n    // --- Cranelift setup ---\n    let mut flag_builder = settings::builder();\n    flag_builder.set(\"opt_level\", \"speed\").unwrap();\n    flag_builder.set(\"is_pic\", \"false\").unwrap();\n    let isa_builder = cranelift_native::builder().map_err(|e| anyhow::anyhow!(\"{}\", e))?;\n    let isa = isa_builder\n        .finish(settings::Flags::new(flag_builder))\n        .map_err(|e| anyhow::anyhow!(\"{}\", e))?;\n\n    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());\n\n    // Register math symbols.\n    builder.symbol(\"jit_expf\", jit_expf as *const u8);\n    builder.symbol(\"jit_logf\", jit_logf as *const u8);\n    builder.symbol(\"jit_sqrtf\", jit_sqrtf as *const u8);\n\n    let mut module = JITModule::new(builder);\n    let ptr_type = module.target_config().pointer_type();\n\n    // --- Declare math function signatures ---\n    let mut math_sig = module.make_signature();\n    math_sig.params.push(AbiParam::new(types::F32));\n    math_sig.returns.push(AbiParam::new(types::F32));\n\n    let expf_id = module.declare_function(\"jit_expf\", Linkage::Import, &math_sig)?;\n    let logf_id = module.declare_function(\"jit_logf\", Linkage::Import, &math_sig)?;\n    let sqrtf_id = module.declare_function(\"jit_sqrtf\", Linkage::Import, &math_sig)?;\n\n    // --- Build kernel function signature ---\n    // fn(in0: *f32, in1: *f32, ..., out: *mut f32, n: u64)\n    let mut sig = module.make_signature();\n    for _ in 0..num_inputs {\n        sig.params.push(AbiParam::new(ptr_type));\n    }\n    sig.params.push(AbiParam::new(ptr_type)); // output pointer\n    sig.params.push(AbiParam::new(types::I64)); // n\n\n    let func_id = module.declare_function(\"kernel\", Linkage::Local, &sig)?;\n\n    // --- Build function body ---\n    let mut func =\n        Function::with_name_signature(cranelift_codegen::ir::UserFuncName::user(0, 0), sig.clone());\n\n    let mut func_ctx = FunctionBuilderContext::new();\n    let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);\n\n    // Declare references to math functions.\n    let expf_ref = module.declare_func_in_func(expf_id, builder.func);\n    let logf_ref = module.declare_func_in_func(logf_id, builder.func);\n    let sqrtf_ref = module.declare_func_in_func(sqrtf_id, builder.func);\n\n    let entry_block = builder.create_block();\n    builder.append_block_params_for_function_params(entry_block);\n    builder.switch_to_block(entry_block);\n    builder.seal_block(entry_block);\n\n    // Extract parameters.\n    let block_params: Vec<Value> = builder.block_params(entry_block).to_vec();\n    let input_ptrs: Vec<Value> = block_params[..num_inputs].to_vec();\n    let out_ptr = block_params[num_inputs];\n    let n_param = block_params[num_inputs + 1];\n\n    // Loop: for i in 0..n\n    let loop_header = builder.create_block();\n    builder.append_block_param(loop_header, types::I64);\n    let loop_body = builder.create_block();\n    let loop_exit = builder.create_block();\n\n    // i = 0\n    let zero = builder.ins().iconst(types::I64, 0);\n    builder.ins().jump(loop_header, &[zero]);\n\n    // Loop header: check i < n\n    builder.switch_to_block(loop_header);\n    let i = builder.block_params(loop_header)[0];\n    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, i, n_param);\n    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);\n    // Do NOT seal loop_header yet — loop_body back-edge not created yet.\n\n    // Loop body\n    builder.switch_to_block(loop_body);\n    builder.seal_block(loop_body); // Only predecessor is loop_header (already defined).\n\n    // Byte offset = i * 4 (sizeof f32)\n    let four = builder.ins().iconst(types::I64, 4);\n    let byte_offset = builder.ins().imul(i, four);\n\n    let math_refs = MathFuncRefs {\n        expf: expf_ref,\n        logf: logf_ref,\n        sqrtf: sqrtf_ref,\n    };\n\n    let result = build_expression(\n        graph,\n        kernel.root,\n        &mut builder,\n        &input_ptrs,\n        &input_index,\n        byte_offset,\n        &math_refs,\n    )?;\n\n    // Store result to out[i]\n    let out_addr = builder.ins().iadd(out_ptr, byte_offset);\n    builder.ins().store(MemFlags::new(), result, out_addr, 0);\n\n    // i += 1, jump back to header\n    let one = builder.ins().iconst(types::I64, 1);\n    let i_next = builder.ins().iadd(i, one);\n    builder.ins().jump(loop_header, &[i_next]);\n\n    // Now seal loop_header — both predecessors (entry, loop_body) are complete.\n    builder.seal_block(loop_header);\n\n    // Exit\n    builder.switch_to_block(loop_exit);\n    builder.seal_block(loop_exit); // Only predecessor is loop_header.\n    builder.ins().return_(&[]);\n\n    builder.finalize();\n\n    // --- Compile ---\n    let mut ctx = cranelift_codegen::Context::for_function(func);\n    module.define_function(func_id, &mut ctx)?;\n    module.finalize_definitions()?;\n\n    let fn_ptr = module.get_finalized_function(func_id);\n\n    Ok(CompiledKernel {\n        num_inputs,\n        _module: module,\n        fn_ptr,\n    })\n}\n\nstruct MathFuncRefs {\n    expf: cranelift_codegen::ir::FuncRef,\n    logf: cranelift_codegen::ir::FuncRef,\n    sqrtf: cranelift_codegen::ir::FuncRef,\n}\n\n/// Recursively build the Cranelift IR for the expression tree rooted at `id`.\nfn build_expression(\n    graph: &Graph,\n    id: NodeId,\n    builder: &mut FunctionBuilder,\n    input_ptrs: &[Value],\n    input_index: &HashMap<NodeId, usize>,\n    byte_offset: Value,\n    math: &MathFuncRefs,\n) -> Result<Value> {\n    let node = graph.node(id);\n\n    match &node.op {\n        Op::Load => {\n            let idx = input_index\n                .get(&id)\n                .ok_or_else(|| anyhow::anyhow!(\"Load node {:?} not found in input_index\", id))?;\n            let ptr = input_ptrs[*idx];\n            let addr = builder.ins().iadd(ptr, byte_offset);\n            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))\n        }\n        Op::Const(val) => Ok(builder.ins().f32const(*val)),\n        Op::Add | Op::Sub | Op::Mul | Op::Div => {\n            let lhs = build_expression(\n                graph,\n                node.inputs[0],\n                builder,\n                input_ptrs,\n                input_index,\n                byte_offset,\n                math,\n            )?;\n            let rhs = build_expression(\n                graph,\n                node.inputs[1],\n"
}
```

**Tool Call: Read file `src/core/eager/mod.rs` (lines 1-200)**
Status: Completed

```json
{
  "Text": "pub mod ops;\nmod tensor;\npub use ops::conv;\npub use tensor::ETensor as EagerTensor;\n\n// Temporary compatibility alias while migrating call sites.\npub type ETensor<T> = EagerTensor<T>;\n"
}
```

**Tool Call: Check diagnostics for `venum/src/core/lazy/tensor.rs`**
Status: Completed

warning at line 5: unused import: `Product`
warning at line 5: unused import: `Sum`


**Tool Call: cargo test -q**
Status: Completed

Command "cargo test -q" failed with exit code 101.

```
warning: unused imports: `HashMap` and `HashSet`
 --> src/core/lazy/optimize.rs:1:24
  |
1 | use std::collections::{HashMap, HashSet};
  |                        ^^^^^^^  ^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

warning: unused import: `Product`
 --> src/core/lazy/tensor.rs:5:12
  |
5 |     iter::{Product, Sum},
  |            ^^^^^^^

warning: unused import: `Sum`
 --> src/core/lazy/tensor.rs:5:21
  |
5 |     iter::{Product, Sum},
  |                     ^^^

error[E0004]: non-exhaustive patterns: `&Op::Sum(_, _)`, `&Op::Prod(_, _)`, `&Op::Max(_, _)` and 1 more not covered
   --> src/core/lazy/optimize.rs:96:11
    |
 96 |     match &node.op {
    |           ^^^^^^^^ patterns `&Op::Sum(_, _)`, `&Op::Prod(_, _)`, `&Op::Max(_, _)` and 1 more not covered
    |
note: `Op` defined here
   --> src/core/lazy/graph.rs:9:10
    |
  9 | pub enum Op {
    |          ^^
...
 38 |     Sum(Vec<usize>, bool),
    |     --- not covered
 39 |     Prod(Vec<usize>, bool),
    |     ---- not covered
 40 |     Max(Vec<usize>, bool),
    |     --- not covered
 41 |     Min(Vec<usize>, bool),
    |     --- not covered
    = note: the matched value is of type `&Op`
help: ensure that all possible cases are being handled by adding a match arm with a wildcard pattern as shown, or multiple match arms
    |
134 ~         | Op::Pad(_, _) => format!("(tLoad {})", id.0),
135 ~         _ => todo!(),
    |

For more information about this error, try `rustc --explain E0004`.
error: could not compile `venum` (lib) due to 1 previous error; 3 warnings emitted
error: could not compile `venum` (lib test) due to 1 previous error; 3 warnings emitted
```

**Tool Call: src/core/lazy/tensor.rs**
Status: Completed

Diff: venum/src/core/lazy/tensor.rs
```
use anyhow::{anyhow, bail, Result};
use std::sync::Arc;
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
};

use crate::core::{
    eager::ETensor as EagerTensor,
    errors::{ExpansionError, ReshapeError, TransposeError, UnsqueezeError},
    iters::Indexer,
};

use super::context::Context;
use super::graph::{Graph, NodeId, Op};
use super::jit::compile_kernel;
use super::optimize;
use super::render;
use super::schedule::{build_schedule, ScheduleItem};

#[derive(Clone, Debug, PartialEq)]
pub struct RealizedTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl RealizedTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_eager(&self) -> Result<EagerTensor<f32>> {
        EagerTensor::new(&self.data, &self.shape)
    }
}

#[derive(Clone)]
pub struct Tensor {
    graph: Arc<std::sync::Mutex<Graph>>,
    id: NodeId,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn from_tensor(cx: &Context, tensor: &EagerTensor<f32>) -> Self {
        let data = Arc::new(tensor.data().to_vec());
        let shape = tensor.sizes().to_vec();

        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.load(data, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {
        let data = Arc::new(data.to_vec());

        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.load(data, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.constant(value, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // -------- elementwise lazy ops --------

    fn binary_op(&self, rhs: &Tensor, op: Op) -> Tensor {
        let same_graph = Arc::ptr_eq(&self.graph, &rhs.graph);
        let mut graph = self.graph.lock().unwrap();

        let rhs_id = if same_graph {
            rhs.id
        } else {
            let rhs_graph = rhs.graph.lock().unwrap();
            import_subgraph(&rhs_graph, rhs.id, &mut graph)
        };

        let id = graph.binary(op, self.id, rhs_id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    fn unary_op(&self, op: Op) -> Tensor {
        let mut graph = self.graph.lock().unwrap();
        let id = graph.unary(op, self.id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(Op::Exp)
    }

    pub fn ln(&self) -> Tensor {
        self.unary_op(Op::Ln)
    }

    pub fn sqrt(&self) -> Tensor {
        self.unary_op(Op::Sqrt)
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }

    // -------- shape lazy ops --------

    pub fn reshape(&self, sizes: Vec<usize>) -> Result<Tensor> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = sizes.iter().product();
        if old_numel != new_numel {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: sizes
            });
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.reshape(self.id, sizes.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: sizes,
        })
    }

    pub fn permute(&self, permutation: Vec<usize>) -> Result<Tensor> {
        if permutation.len() != self.shape.len() {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: self.shape.clone()
            });
        }

        let mut seen = vec![false; permutation.len()];
        for &p in &permutation {
            if p >= permutation.len() || seen[p] {
                bail!(ReshapeError {
                    current_shape: self.shape.clone(),
                    new_shape: self.shape.clone()
                });
            }
            seen[p] = true;
        }

        let shape = permutation
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();

        let mut graph = self.graph.lock().unwrap();
        let id = graph.permute(self.id, permutation, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if rank < 2 || dim_1 >= rank || dim_2 >= rank {
            bail!(TransposeError);
        }

        let mut shape = self.shape.clone();
        shape.swap(dim_1, dim_2);

        let mut graph = self.graph.lock().unwrap();
        let id = graph.transpose(self.id, dim_1, dim_2, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn expand(&self, expansions: Vec<usize>) -> Result<Tensor> {
        if expansions.len() != self.shape.len() {
            bail!(ExpansionError {
                size: self.shape.len(),
                expansion: expansions.len()
            });
        }

        for (&size, &exp) in self.shape.iter().zip(expansions.iter()) {
            if !(exp == size || size == 1) {
                bail!(ExpansionError {
                    size,
                    expansion: exp
                });
            }
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.expand(self.id, expansions.clone(), expansions.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: expansions,
        })
    }

    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor> {
        if ranges.len() != self.shape.len() {
            bail!(anyhow!(
                "slice ranges rank mismatch: got {}, expected {}",
                ranges.len(),
                self.shape.len()
            ));
        }

        let mut out = Vec::with_capacity(self.shape.len());
        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {
            let size = self.shape[dim];
            let end = if end_raw == 0 { size } else { end_raw };
            if start > end || end > size {
                bail!(anyhow!(
                    "slice range {:?} out of bounds for dim {} with size {}",
                    (start, end),
                    dim,
                    size
                ));
            }
            out.push(end - start);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.slice(self.id, ranges, out.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: out,
        })
    }

    pub fn flip(&self, flips: Vec<usize>) -> Result<Tensor> {
        for &d in &flips {
            if d >= self.shape.len() {
                bail!(anyhow!(
                    "flip dimension {} out of bounds for rank {}",
                    d,
                    self.shape.len()
                ));
            }
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.flip(self.id, flips, self.shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: self.shape.clone(),
        })
    }

    pub fn squeeze(&self) -> Result<Tensor> {
        let mut shape: Vec<usize> = self.shape.iter().copied().filter(|&s| s != 1).collect();
        if shape.is_empty() {
            shape.push(1);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.squeeze(self.id, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn unsqueeze(&self, new_rank: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if new_rank < rank {
            bail!(UnsqueezeError {
                current: rank,
                new_rank
            });
        }
        if new_rank == rank {
            return Ok(self.clone());
        }

        let mut shape = vec![1; new_rank - rank];
        shape.extend_from_slice(&self.shape);

        let mut graph = self.graph.lock().unwrap();
        let id = graph.unsqueeze(self.id, new_rank, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn pad(&self, constant: f32, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut pad = padding;
        pad.resize(self.shape.len(), (0, 0));

        let mut shape = Vec::with_capacity(self.shape.len());
        for (s, (l, r)) in self.shape.iter().copied().zip(pad.iter().copied()) {
            shape.push(l + s + r);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.pad(self.id, constant, pad, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    // -------- reduce lazy ops --------

    pub fn sum_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.sum(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn product_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.prod(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn max_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.max(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn min_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.min(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn sum(&self) -> Result<Tensor> {
        self.sum_dims((0..self.shape.len()).collect(), true)
    }

    pub fn product(&self) -> Result<Tensor> {
        self.product_dims((0..self.shape.len()).collect(), true)
    }

    pub fn max(&self) -> Result<Tensor> {
        self.max_dims((0..self.shape.len()).collect(), true)
    }

    pub fn min(&self) -> Result<Tensor> {
        self.min_dims((0..self.shape.len()).collect(), true)
    }

    // -------- visualization --------

    pub fn render_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_dag(&graph, self.id)
    }

    pub fn render_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_fused_dag(&graph, self.id)
    }

    pub fn render_optimized_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_dag(&opt_graph, opt_root)
    }

    pub fn render_optimized_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_fused_dag(&opt_graph, opt_root)
    }

    // -------- realize --------

    pub fn realize(&self) -> Result<RealizedTensor> {
        let graph = self.graph.lock().unwrap();

        // Fast path: already-backed leaf.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            let data = buffer.as_f32().to_vec();
            let shape = root_node.shape.clone();
            drop(graph);
            return Ok(RealizedTensor::new(data, shape));
        }

        // Only optimize pure elementwise roots for now.
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)
        } else {
            clone_reachable_subgraph(&graph, self.id)
        };
        drop(graph);

        let graph = &exec_graph;
        let root = exec_root;

        let root_node = graph.node(root);
        if let Some(ref buffer) = root_node.buffer {
            return Ok(RealizedTensor::new(
                buffer.as_f32().to_vec(),
                root_node.shape.clone(),
            ));
        }
        if let Op::Const(val) = root_node.op {
            let numel: usize = self.shape.iter().product();
            return Ok(RealizedTensor::new(vec![val; numel], self.shape.clone()));
        }

        let schedule = build_schedule(graph, root);
        let mut realized: std::collections::HashMap<NodeId, Vec<f32>> =
            std::collections::HashMap::new();

        for item in &schedule {
            match item {
                ScheduleItem::Fused(kernel) => {
                    let compiled = compile_kernel(graph, kernel)?;

                    let input_ptrs: Vec<*const f32> = kernel
                        .input_buffers
                        .iter()
                        .map(|&buf_id| {
                            let node = graph.node(buf_id);
                            if let Some(ref buffer) = node.buffer {
                                buffer.as_f32_ptr()
                            } else if let Some(data) = realized.get(&buf_id) {
                                data.as_ptr()
                            } else {
                                panic!("Input buffer {:?} not realized and has no data", buf_id);
                            }
                        })
                        .collect();

                    let mut output = vec![0.0f32; kernel.numel];
                    unsafe {
                        compiled.execute(&input_ptrs, output.as_mut_ptr(), kernel.numel);
                    }
                    realized.insert(kernel.root, output);
                }
                ScheduleItem::Shape(shape_item) => {
                    let input_node = graph.node(shape_item.input);
                    let input_shape = input_node.shape.clone();

                    let input_data = if let Some(data) = realized.get(&shape_item.input) {
                        data.clone()
                    } else if let Some(ref buf) = input_node.buffer {
                        buf.as_f32().to_vec()
                    } else {
                        bail!("Shape op input {:?} is not realized", shape_item.input);
                    };

                    let output = execute_shape_op(
                        &shape_item.op,
                        &input_data,
                        &input_shape,
                        &shape_item.shape,
                    )?;
                    realized.insert(shape_item.root, output);
                }
                ScheduleItem::Reduce(reduce_item) => {
                    let input_node = graph.node(reduce_item.input);
                    let input_shape = input_node.shape.clone();

                    let input_data = if let Some(data) = realized.get(&reduce_item.input) {
                        data.clone()
                    } else if let Some(ref buf) = input_node.buffer {
                        buf.as_f32().to_vec()
                    } else {
                        bail!("Reduce op input {:?} is not realized", reduce_item.input);
                    };

                    let output = execute_reduce_op(
                        &reduce_item.op,
                        &input_data,
                        &input_shape,
                        &reduce_item.shape,
                    )?;
                    realized.insert(reduce_item.root, output);
                }
            }
        }

        let data = realized
            .remove(&root)
            .ok_or_else(|| anyhow!("Root node was not realized"))?;

        Ok(RealizedTensor::new(data, graph.node(root).shape.clone()))
    }

    pub fn realize_eager(&self) -> Result<EagerTensor<f32>> {
        self.realize()?.to_eager()
    }
}

// -------- shape-op execution helpers --------

fn execute_shape_op(
    op: &Op,
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    match op {
        Op::Reshape(_) => execute_reshape(input_data, input_shape, output_shape),
        Op::Permute(perm) => execute_permute(input_data, input_shape, output_shape, perm),
        Op::Transpose(d1, d2) => execute_transpose(input_data, input_shape, output_shape, *d1, *d2),
        Op::Expand(_) => execute_expand(input_data, input_shape, output_shape),
        Op::Slice(ranges) => execute_slice(input_data, input_shape, output_shape, ranges),
        Op::Flip(flips) => execute_flip(input_data, input_shape, output_shape, flips),
        Op::Squeeze => execute_reshape(input_data, input_shape, output_shape),
        Op::Unsqueeze(_) => execute_reshape(input_data, input_shape, output_shape),
        Op::Pad(constant, padding) => {
            execute_pad(input_data, input_shape, output_shape, *constant, padding)
        }
        _ => bail!("execute_shape_op called with non-shape op: {:?}", op),
    }
}

fn execute_reduce_op(
    op: &Op,
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    match op {
        Op::Sum(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().sum(),
        ),
        Op::Prod(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().product(),
        ),
        Op::Max(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(0.0)
            },
        ),
        Op::Min(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(0.0)
            },
        ),
        _ => bail!("execute_reduce_op called with non-reduce op: {:?}", op),
    }
}

fn execute_reduce(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    dimensions: &[usize],
    keepdims: bool,
    reducer: impl Fn(&[f32]) -> f32,
) -> Result<Vec<f32>> {
    validate_reduce_dims(input_shape, dimensions)?;

    let expected_shape = reduced_shape(input_shape, dimensions, keepdims)?;
    if expected_shape != output_shape {
        bail!(
            "reduce output shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            output_shape
        );
    }

    let mut out = Vec::with_capacity(output_shape.iter().product());

    for out_idx in Indexer::new(output_shape) {
        let mut fixed = vec![None; input_shape.len()];

        if keepdims {
            for d in 0..input_shape.len() {
                if !dimensions.contains(&d) {
                    fixed[d] = Some(out_idx[d]);
                }
            }
        } else {
            let mut out_pos = 0usize;
            for d in 0..input_shape.len() {
                if !dimensions.contains(&d) {
                    fixed[d] = Some(out_idx[out_pos]);
                    out_pos += 1;
                }
            }
        }

        let mut reduced_values = Vec::new();
        collect_reduce_values(
            input_data,
            input_shape,
            dimensions,
            &fixed,
            0,
            &mut vec![0; input_shape.len()],
            &mut reduced_values,
        );

        out.push(reducer(&reduced_values));
    }

    Ok(out)
}

fn collect_reduce_values(
    input_data: &[f32],
    input_shape: &[usize],
    dimensions: &[usize],
    fixed: &[Option<usize>],
    dim: usize,
    current: &mut Vec<usize>,
    values: &mut Vec<f32>,
) {
    if dim == input_shape.len() {
        let off = idx_to_offset(current, input_shape);
        values.push(input_data[off]);
        return;
    }

    if dimensions.contains(&dim) {
        for i in 0..input_shape[dim] {
            current[dim] = i;
            collect_reduce_values(
                input_data,
                input_shape,
                dimensions,
                fixed,
                dim + 1,
                current,
                values,
            );
        }
    } else if let Some(idx) = fixed[dim] {
        current[dim] = idx;
        collect_reduce_values(
            input_data,
            input_shape,
            dimensions,
            fixed,
            dim + 1,
            current,
            values,
        );
    }
}

fn reduced_shape(shape: &[usize], dimensions: &[usize], keepdims: bool) -> Result<Vec<usize>> {
    validate_reduce_dims(shape, dimensions)?;

    let mut out = Vec::new();
    for (d, &size) in shape.iter().enumerate() {
        if dimensions.contains(&d) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(size);
        }
    }

    if out.is_empty() {
        out.push(1);
    }

    Ok(out)
}

fn validate_reduce_dims(shape: &[usize], dimensions: &[usize]) -> Result<()> {
    let mut seen = vec![false; shape.len()];
    for &d in dimensions {
        if d >= shape.len() {
            bail!(
                "reduce dimension {} out of bounds for rank {}",
                d,
                shape.len()
            );
        }
        if seen[d] {
            bail!("duplicate reduce dimension {}", d);
        }
        seen[d] = true;
    }
    Ok(())
}

fn execute_reshape(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    let in_numel: usize = input_shape.iter().product();
    let out_numel: usize = output_shape.iter().product();
    if in_numel != out_numel || input_data.len() != in_numel {
        bail!(
            "invalid reshape numel: input {:?}, output {:?}",
            input_shape,
            output_shape
        );
    }
    Ok(input_data.to_vec())
}

fn execute_permute(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    perm: &[usize],
) -> Result<Vec<f32>> {
    if perm.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("invalid permute rank");
    }
    let mut out = vec![0.0f32; output_shape.iter().product()];

    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for (out_dim, &in_dim) in perm.iter().enumerate() {
            in_idx[in_dim] = out_idx[out_dim];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_transpose(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    dim_1: usize,
    dim_2: usize,
) -> Result<Vec<f32>> {
    if dim_1 >= input_shape.len()
        || dim_2 >= input_shape.len()
        || input_shape.len() != output_shape.len()
    {
        bail!("invalid transpose dims");
    }

    let mut perm: Vec<usize> = (0..input_shape.len()).collect();
    perm.swap(dim_1, dim_2);
    execute_permute(input_data, input_shape, output_shape, &perm)
}

fn execute_expand(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    if input_shape.len() != output_shape.len() {
        bail!("expand rank mismatch");
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            if input_shape[d] == output_shape[d] {
                in_idx[d] = out_idx[d];
            } else if input_shape[d] == 1 {
                in_idx[d] = 0;
            } else {
                bail!(
                    "cannot expand dim {} from {} to {}",
                    d,
                    input_shape[d],
                    output_shape[d]
                );
            }
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_slice(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<f32>> {
    if ranges.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("slice rank mismatch");
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            let (start, end_raw) = ranges[d];
            let end = if end_raw == 0 {
                input_shape[d]
            } else {
                end_raw
            };
            if start > end || end > input_shape[d] {
                bail!("invalid slice range {:?} for dim {}", ranges[d], d);
            }
            in_idx[d] = start + out_idx[d];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_flip(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    flips: &[usize],
) -> Result<Vec<f32>> {
    if input_shape != output_shape {
        bail!("flip must preserve shape");
    }

    let mut flip_mask = vec![false; input_shape.len()];
    for &d in flips {
        if d >= input_shape.len() {
            bail!("flip dim {} out of range", d);
        }
        flip_mask[d] = true;
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = out_idx.clone();
        for d in 0..in_idx.len() {
            if flip_mask[d] {
                in_idx[d] = input_shape[d] - 1 - in_idx[d];
            }
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_pad(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    constant: f32,
    padding: &[(usize, usize)],
) -> Result<Vec<f32>> {
    if input_shape.len() != output_shape.len() {
        bail!("pad rank mismatch");
    }

    let mut pad = padding.to_vec();
    pad.resize(input_shape.len(), (0, 0));

    let mut expected = Vec::with_capacity(input_shape.len());
    for d in 0..input_shape.len() {
        expected.push(pad[d].0 + input_shape[d] + pad[d].1);
    }
    if expected != output_shape {
        bail!(
            "pad output shape mismatch: expected {:?}, got {:?}",
            expected,
            output_shape
        );
    }

    let mut out = vec![constant; output_shape.iter().product()];
    for in_idx in Indexer::new(input_shape) {
        let mut out_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            out_idx[d] = pad[d].0 + in_idx[d];
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn idx_to_offset(index: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut off = 0usize;
    for d in (0..shape.len()).rev() {
        off += index[d] * stride;
        stride *= shape[d];
    }
    off
}

// -------- graph import helpers --------

fn import_subgraph(src_graph: &Graph, src_id: NodeId, dst_graph: &mut Graph) -> NodeId {
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();

    let mut buffer_map: std::collections::HashMap<*const Vec<f32>, NodeId> =
        std::collections::HashMap::new();
    for (i, node) in dst_graph.nodes.iter().enumerate() {
        if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
            buffer_map.insert(Arc::as_ptr(arc), NodeId(i));
        }
    }

    import_node(src_graph, src_id, dst_graph, &mut id_map, &buffer_map)
}

fn import_node(
    src_graph: &Graph,
    src_id: NodeId,
    dst_graph: &mut Graph,
    id_map: &mut std::collections::HashMap<NodeId, NodeId>,
    buffer_map: &std::collections::HashMap<*const Vec<f32>, NodeId>,
) -> NodeId {
    if let Some(&mapped) = id_map.get(&src_id) {
        return mapped;
    }

    let node = src_graph.node(src_id);

    if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
        if let Some(&existing_id) = buffer_map.get(&Arc::as_ptr(arc)) {
            id_map.insert(src_id, existing_id);
            return existing_id;
        }
    }

    let new_inputs: Vec<NodeId> = node
        .inputs
        .iter()
        .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map, buffer_map))
        .collect();

    let new_id = dst_graph.add_node(super::graph::Node {
        op: node.op.clone(),
        inputs: new_inputs,
        shape: node.shape.clone(),
        dtype: node.dtype,
        buffer: node.buffer.clone(),
    });

    id_map.insert(src_id, new_id);
    new_id
}

// -------- optimization helpers --------

fn is_optimize_safe(graph: &Graph, root: NodeId) -> bool {
    fn dfs(graph: &Graph, id: NodeId, seen: &mut std::collections::HashSet<NodeId>) -> bool {
        if !seen.insert(id) {
            return true;
        }
        let node = graph.node(id);
        let here_ok = matches!(
            node.op,
            Op::Load
                | Op::Const(_)
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Exp
                | Op::Ln
                | Op::Sqrt
                | Op::Neg
        );
        if !here_ok {
            return false;
        }
        node.inputs.iter().all(|&inp| dfs(graph, inp, seen))
    }

    let mut seen = std::collections::HashSet::new();
    dfs(graph, root, &mut seen)
}

fn clone_reachable_subgraph(src: &Graph, root: NodeId) -> (Graph, NodeId) {
    let mut dst = Graph::new();
    let mut id_map = std::collections::HashMap::new();
    let buffer_map = std::collections::HashMap::new();
    let new_root = import_node(src, root, &mut dst, &mut id_map, &buffer_map);
    (dst, new_root)
}

// -------- operator overloads --------

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Add)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Sub)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Mul)
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Div)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Add)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Sub)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Mul)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Div)
    }
}

```


**Tool Call: src/core/lazy/optimize.rs**
Status: Completed

Diff: venum/src/core/lazy/optimize.rs
```
use super::dtype::DType;
use super::graph::{Graph, Node, NodeId, Op};

/// Run the egglog optimizer on the graph rooted at `root`.
/// Returns a new, optimized Graph and the new root NodeId.
pub fn optimize(graph: &Graph, root: NodeId) -> (Graph, NodeId) {
    // 1. Convert our Graph into egglog terms.
    let mut egraph = egglog::EGraph::default();

    // Define the tensor expression sort.
    egraph
        .parse_and_run_program(
            None,
            r#"
            (datatype TExpr
                (tLoad i64)
                (tConst f64)
                (tAdd TExpr TExpr)
                (tSub TExpr TExpr)
                (tMul TExpr TExpr)
                (tDiv TExpr TExpr)
                (tExp TExpr)
                (tLn  TExpr)
                (tSqrt TExpr)
                (tNeg TExpr)
            )
            "#,
        )
        .unwrap();

    // Algebraic simplification rules.
    egraph
        .parse_and_run_program(
            None,
            r#"
            ;; --- Identity rules ---
            (rewrite (tAdd ?a (tConst 0.0)) ?a)
            (rewrite (tAdd (tConst 0.0) ?a) ?a)
            (rewrite (tSub ?a (tConst 0.0)) ?a)
            (rewrite (tMul ?a (tConst 1.0)) ?a)
            (rewrite (tMul (tConst 1.0) ?a) ?a)
            (rewrite (tDiv ?a (tConst 1.0)) ?a)

            ;; --- Zero rules ---
            (rewrite (tMul ?a (tConst 0.0)) (tConst 0.0))
            (rewrite (tMul (tConst 0.0) ?a) (tConst 0.0))

            ;; --- Double negation ---
            (rewrite (tNeg (tNeg ?a)) ?a)

            ;; --- Inverse ops ---
            (rewrite (tExp (tLn ?a)) ?a)
            (rewrite (tLn (tExp ?a)) ?a)
            (rewrite (tSqrt (tMul ?a ?a)) ?a)

            ;; --- Self-cancellation ---
            (rewrite (tSub ?a ?a) (tConst 0.0))
            (rewrite (tDiv ?a ?a) (tConst 1.0))

            ;; --- Commutativity ---
            (rewrite (tAdd ?a ?b) (tAdd ?b ?a))
            (rewrite (tMul ?a ?b) (tMul ?b ?a))

            ;; --- Strength reduction ---
            (rewrite (tAdd ?a ?a) (tMul (tConst 2.0) ?a))
            "#,
        )
        .unwrap();

    // 2. Insert our DAG as egglog terms.
    let term_str = node_to_egglog(graph, root);
    let insert_cmd = format!("(let root {})", term_str);
    egraph.parse_and_run_program(None, &insert_cmd).unwrap();

    // 3. Run equality saturation.
    egraph.parse_and_run_program(None, "(run 10)").unwrap();

    // 4. Extract the best term.
    let outputs = egraph
        .parse_and_run_program(None, "(extract root)")
        .unwrap();

    let extracted = outputs[0].to_string();

    // 5. Parse the extracted term back into a new Graph.
    let (new_graph, new_root) = parse_egglog_term(graph, &extracted);

    (new_graph, new_root)
}

/// Convert a graph node to an egglog s-expression string.
fn node_to_egglog(graph: &Graph, id: NodeId) -> String {
    let node = graph.node(id);
    match &node.op {
        Op::Load => format!("(tLoad {})", id.0),
        Op::Const(v) => format!("(tConst {:.1})", *v as f64),
        Op::Add => format!(
            "(tAdd {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Sub => format!(
            "(tSub {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Mul => format!(
            "(tMul {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Div => format!(
            "(tDiv {} {})",
            node_to_egglog(graph, node.inputs[0]),
            node_to_egglog(graph, node.inputs[1])
        ),
        Op::Exp => format!("(tExp {})", node_to_egglog(graph, node.inputs[0])),
        Op::Ln => format!("(tLn {})", node_to_egglog(graph, node.inputs[0])),
        Op::Sqrt => format!("(tSqrt {})", node_to_egglog(graph, node.inputs[0])),
        Op::Neg => format!("(tNeg {})", node_to_egglog(graph, node.inputs[0])),

        // Non-elementwise / shape ops are currently not modeled in egglog.
        // Keep optimizer safe by treating them as opaque leaves.
        Op::Reshape(_)
        | Op::Permute(_)
        | Op::Transpose(_, _)
        | Op::Expand(_)
        | Op::Slice(_)
        | Op::Flip(_)
        | Op::Squeeze
        | Op::Unsqueeze(_)
        | Op::Pad(_, _)
        | Op::Sum(_, _)
        | Op::Prod(_, _)
        | Op::Max(_, _)
        | Op::Min(_, _) => format!("(tLoad {})", id.0),
    }
}

/// Parse an egglog extracted term back into a Graph.
/// Reuses Load buffers from the original graph.
fn parse_egglog_term(original: &Graph, term: &str) -> (Graph, NodeId) {
    let term = term.trim();
    let mut graph = Graph::new();
    let root = parse_sexpr(original, term, &mut graph);
    (graph, root)
}

/// Recursive s-expression parser for egglog output.
fn parse_sexpr(original: &Graph, s: &str, graph: &mut Graph) -> NodeId {
    let s = s.trim();

    // Not an s-expression — should be a bare token (shouldn't happen at top level).
    if !s.starts_with('(') {
        panic!("Expected s-expression, got: {}", s);
    }

    // Strip outer parens.
    let inner = &s[1..s.len() - 1];

    // Split into head and args.
    let (head, rest) = split_head(inner);

    match head {
        "tLoad" => {
            let id: usize = rest.trim().parse().unwrap();
            let orig_node = original.node(NodeId(id));
            graph.add_node(Node {
                op: Op::Load,
                inputs: vec![],
                shape: orig_node.shape.clone(),
                dtype: orig_node.dtype,
                buffer: orig_node.buffer.clone(),
            })
        }
        "tConst" => {
            let val: f64 = rest.trim().parse().unwrap();
            // Infer shape from context — for now use [1] for scalar constants.
            // The JIT handles Const nodes inline (f32const), so shape doesn't matter for codegen.
            graph.add_node(Node {
                op: Op::Const(val as f32),
                inputs: vec![],
                shape: vec![1],
                dtype: DType::F32,
                buffer: None,
            })
        }
        "tAdd" | "tSub" | "tMul" | "tDiv" => {
            let (arg1_str, arg2_str) = split_two_args(rest);
            let lhs = parse_sexpr(original, arg1_str, graph);
            let rhs = parse_sexpr(original, arg2_str, graph);
            let op = match head {
                "tAdd" => Op::Add,
                "tSub" => Op::Sub,
                "tMul" => Op::Mul,
                "tDiv" => Op::Div,
                _ => unreachable!(),
            };
            let shape = graph.node(lhs).shape.clone();
            graph.add_node(Node {
                op,
                inputs: vec![lhs, rhs],
                shape,
                dtype: DType::F32,
                buffer: None,
            })
        }
        "tExp" | "tLn" | "tSqrt" | "tNeg" => {
            let arg = parse_sexpr(original, rest.trim(), graph);
            let op = match head {
                "tExp" => Op::Exp,
                "tLn" => Op::Ln,
                "tSqrt" => Op::Sqrt,
                "tNeg" => Op::Neg,
                _ => unreachable!(),
            };
            let shape = graph.node(arg).shape.clone();
            graph.add_node(Node {
                op,
                inputs: vec![arg],
                shape,
                dtype: DType::F32,
                buffer: None,
            })
        }
        _ => {
            // Be conservative for any unknown extracted term:
            // materialize it as an opaque load by reusing original root buffer.
            let orig_node = original.node(NodeId(0));
            graph.add_node(Node {
                op: Op::Load,
                inputs: vec![],
                shape: orig_node.shape.clone(),
                dtype: orig_node.dtype,
                buffer: orig_node.buffer.clone(),
            })
        }
    }
}

/// Split "head rest..." from an s-expression body.
fn split_head(s: &str) -> (&str, &str) {
    let s = s.trim();
    if let Some(idx) = s.find(|c: char| c.is_whitespace()) {
        (&s[..idx], &s[idx..])
    } else {
        (s, "")
    }
}

/// Split two s-expression arguments from a string like " (tAdd ...) (tMul ...)".
fn split_two_args(s: &str) -> (&str, &str) {
    let s = s.trim();
    let end_of_first = find_sexpr_end(s);
    let first = &s[..end_of_first];
    let second = s[end_of_first..].trim();
    (first, second)
}

/// Find the end index of the first s-expression in the string.
fn find_sexpr_end(s: &str) -> usize {
    let s = s.trim();
    if s.starts_with('(') {
        let mut depth = 0;
        for (i, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return i + 1;
                    }
                }
                _ => {}
            }
        }
        s.len()
    } else {
        // Bare token — find next whitespace or end.
        s.find(|c: char| c.is_whitespace()).unwrap_or(s.len())
    }
}

```


**Tool Call: cargo test -q**
Status: Completed

```
warning: method `size_bytes` is never used
 --> src/core/lazy/dtype.rs:9:12
  |
8 | impl DType {
  | ---------- method in this implementation
9 |     pub fn size_bytes(self) -> usize {
  |            ^^^^^^^^^^
  |
  = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: methods `dtype` and `len` are never used
  --> src/core/lazy/dtype.rs:22:12
   |
21 | impl Buffer {
   | ----------- methods in this implementation
22 |     pub fn dtype(&self) -> DType {
   |            ^^^^^
...
28 |     pub fn len(&self) -> usize {
   |            ^^^

warning: field `0` is never read
  --> src/core/lazy/graph.rs:27:13
   |
27 |     Reshape(Vec<usize>),
   |     ------- ^^^^^^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
27 -     Reshape(Vec<usize>),
27 +     Reshape(()),
   |

warning: field `0` is never read
  --> src/core/lazy/graph.rs:30:12
   |
30 |     Expand(Vec<usize>),
   |     ------ ^^^^^^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
30 -     Expand(Vec<usize>),
30 +     Expand(()),
   |

warning: field `0` is never read
  --> src/core/lazy/graph.rs:34:15
   |
34 |     Unsqueeze(usize),
   |     --------- ^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
34 -     Unsqueeze(usize),
34 +     Unsqueeze(()),
   |

warning: methods `is_binary` and `is_unary` are never used
  --> src/core/lazy/graph.rs:52:12
   |
44 | impl Op {
   | ------- methods in this implementation
...
52 |     pub fn is_binary(&self) -> bool {
   |            ^^^^^^^^^
...
56 |     pub fn is_unary(&self) -> bool {
   |            ^^^^^^^^

warning: methods `node_mut` and `consumer_count` are never used
   --> src/core/lazy/graph.rs:117:12
    |
102 | impl Graph {
    | ---------- methods in this implementation
...
117 |     pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
    |            ^^^^^^^^
...
340 |     pub fn consumer_count(&self, id: NodeId) -> usize {
    |            ^^^^^^^^^^^^^^

warning: field `sqrtf` is never read
   --> src/core/lazy/jit.rs:283:5
    |
280 | struct MathFuncRefs {
    |        ------------ field in this struct
...
283 |     sqrtf: cranelift_codegen::ir::FuncRef,
    |     ^^^^^


running 31 tests
...............................
test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**Tool Call: Check diagnostics for `venum/src/core/lazy/optimize.rs`**
Status: Completed

File doesn't have errors or warnings!

**Tool Call: Check diagnostics for `venum/src/core/lazy/tensor.rs`**
Status: Completed

File doesn't have errors or warnings!

## Assistant

I continued from the refactor state and got the lazy branch back to a clean passing test run.

## What I fixed

### 1. `optimize.rs` now handles reduce ops conservatively
The build was failing because `node_to_egglog()` didn’t cover the new reduce variants.

I updated it so these are treated the same way as shape ops for now:

- `Op::Sum`
- `Op::Prod`
- `Op::Max`
- `Op::Min`

That keeps egglog safely limited to pure elementwise subgraphs, which matches the current strategy.

### 2. Fixed a latent root-mapping bug in `tensor.rs`
There was a fallback path in `realize()` for non-optimizable graphs that cloned the reachable subgraph and then assumed the cloned root was always `NodeId(0)`.

That assumption wasn’t actually guaranteed by the import order.

I changed `clone_reachable_subgraph()` to return:

- the cloned `Graph`
- the actual mapped `NodeId` for the root

So the non-egglog execution path is now structurally correct for shape/reduce graphs too.

### 3. Cleaned the immediate warnings introduced by those files
I removed the unused imports in:

- `src/core/lazy/optimize.rs`
- `src/core/lazy/tensor.rs`

## Validation

I ran:

- `cargo test -q`

Current result:

- **31 tests passed**
- **0 failures**

## Current state

So at this point:

- lazy-first API is still in place
- shared `Context` is present
- shape ops are scheduled and realized
- reduce ops are scheduled and realized
- egglog remains gated to elementwise-safe graphs
- test suite is green again

## Remaining warnings

There are still some non-fatal warnings, mostly dead-code / temporarily-unused API surface in:

- `src/core/lazy/dtype.rs`
- `src/core/lazy/graph.rs`
- `src/core/lazy/jit.rs`

These look like normal refactor residue, not blockers.

## Best next step

The most sensible continuation is now **Phase 4 hardening**, specifically:

1. tighten reduce execution coverage
2. add tests for mixed pipelines like:
   - elementwise -> shape -> reduce
   - reduce -> elementwise
   - chained reduce + keepdims cases
3. decide whether `render_optimized_*` should explicitly refuse non-elementwise roots instead of relying on current safety assumptions

If you want, I can continue directly with that and push the reduce path from “working” to “solid and test-backed.”

## User

Yeah go ahead

**Tool Call: Search files for regex `reshape\(|permute\(|transpose\(|expand\(|slice\(|flip\(|squeeze\(|unsqueeze\(|pad\(`**
Status: Completed

Showing matches 1-20 (there were more matches found; use offset: 20 to see next page):

## Matches in venum/src/core/lazy/tests.rs

### mod lazy_tests › fn realize_leaf › L11-21
```
    fn realize_leaf() -> Result<()> {
        let cx = Context::new();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let dt = Tensor::from_slice(&cx, &data, vec![4]);
        let result = dt.realize()?;

        assert_eq!(result.data(), &data);
        assert_eq!(result.shape(), &[4]);
        Ok(())
    }
```

### mod lazy_tests › fn add_two_tensors › L24-34
```
    fn add_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &a + &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }
```

### mod lazy_tests › fn mul_two_tensors › L37-47
```
    fn mul_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 10.0, 10.0, 10.0], vec![4]);
        let c = &a * &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[20.0, 30.0, 40.0, 50.0]);
        Ok(())
    }
```

### mod lazy_tests › fn sub_and_div › L50-60
```
    fn sub_and_div() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);

        let sub_result = (&a - &b).realize()?;
        assert_eq!(sub_result.data(), &[9.0, 18.0, 27.0, 36.0]);

        let div_result = (&a / &b).realize()?;
        assert_eq!(div_result.data(), &[10.0, 10.0, 10.0, 10.0]);
```

3 lines remaining in ancestor node. Read the file to see all.

### mod lazy_tests › fn fused_add_mul › L66-76
```
    fn fused_add_mul() -> Result<()> {
        let cx = Context::new();

        // (a + b) * a should produce ONE fused kernel.
        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &(&a + &b) * &a;
        let result = c.realize()?;

        // (1+10)*1=11, (2+20)*2=44, (3+30)*3=99, (4+40)*4=176
        assert_eq!(result.data(), &[11.0, 44.0, 99.0, 176.0]);
```

2 lines remaining in ancestor node. Read the file to see all.

### mod lazy_tests › fn unary_neg › L81-89
```
    fn unary_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0, -4.0], vec![4]);
        let result = a.neg().realize()?;

        assert_eq!(result.data(), &[-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }
```

### mod lazy_tests › fn unary_exp › L92-101
```
    fn unary_exp() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[0.0, 1.0, 2.0], vec![3]);
        let result = a.exp().realize()?;

        let expected: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        assert!(approx_eq(result.data(), &expected, 1e-5));
        Ok(())
    }
```

### mod lazy_tests › fn unary_ln › L104-114
```
    fn unary_ln() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.718_281_7, 7.389056], vec![3]);
        let result = a.ln().realize()?;

        let expected: Vec<f32> = [1.0f32, 2.718_281_7, 7.389056]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert!(approx_eq(result.data(), &expected, 1e-4));
```

2 lines remaining in ancestor node. Read the file to see all.

### mod lazy_tests › fn unary_sqrt › L119-127
```
    fn unary_sqrt() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 4.0, 9.0, 16.0], vec![4]);
        let result = a.sqrt().realize()?;

        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
```

### mod lazy_tests › fn chain_fused_unary_binary › L130-140
```
    fn chain_fused_unary_binary() -> Result<()> {
        let cx = Context::new();

        // exp(a) + b — should fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[0.0, 0.0, 0.0], vec![3]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = &a.exp() + &b;
        let result = c.realize()?;

        // exp(0) + 1 = 2, exp(0) + 2 = 3, exp(0) + 3 = 4
        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);
```

2 lines remaining in ancestor node. Read the file to see all.

### mod lazy_tests › fn from_eager_tensor › L145-155
```
    fn from_eager_tensor() -> Result<()> {
        let cx = Context::new();

        let eager = EagerTensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
        let dt = Tensor::from_tensor(&cx, &eager);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let c = &dt + &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(result.shape(), &[2, 2]);
```

2 lines remaining in ancestor node. Read the file to see all.

### mod lazy_tests › fn larger_tensor › L160-170
```
    fn larger_tensor() -> Result<()> {
        let cx = Context::new();

        let n = 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let a = Tensor::from_slice(&cx, &a_data, vec![n]);
        let b = Tensor::from_slice(&cx, &b_data, vec![n]);
        let c = &a + &b;
        let result = c.realize()?;
```

5 lines remaining in ancestor node. Read the file to see all.

### mod lazy_tests › fn larger_tensor › L171-175
```

        let expected: Vec<f32> = (0..n).map(|_| n as f32).collect();
        assert_eq!(result.data(), expected.as_slice());
        Ok(())
    }
```

### mod lazy_tests › fn deep_fusion_chain › L178-188
```
    fn deep_fusion_chain() -> Result<()> {
        let cx = Context::new();

        // a * b + a - b should all fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);

        let c = &(&(&a * &b) + &a) - &b;
        let result = c.realize()?;

        // (2*1 + 2 - 1) = 3, (3*1 + 3 - 1) = 5, (4*1 + 4 - 1) = 7, (5*1 + 5 - 1) = 9
```

3 lines remaining in ancestor node. Read the file to see all.

### mod lazy_tests › fn constant_tensor › L194-203
```
    fn constant_tensor() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = Tensor::constant(&cx, 10.0, vec![3]);
        let result = (&a + &c).realize()?;

        assert_eq!(result.data(), &[11.0, 12.0, 13.0]);
        Ok(())
    }
```

### mod lazy_tests › fn egglog_add_zero_identity › L208-217
```
    fn egglog_add_zero_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a + 0 should be optimized to just a.
        let result = (&a + &zero).realize()?;
        assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
        Ok(())
    }
```

### mod lazy_tests › fn egglog_mul_one_identity › L220-229
```
    fn egglog_mul_one_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let one = Tensor::constant(&cx, 1.0, vec![3]);
        // a * 1 should be optimized to just a.
        let result = (&a * &one).realize()?;
        assert_eq!(result.data(), &[5.0, 10.0, 15.0]);
        Ok(())
    }
```

### mod lazy_tests › fn egglog_mul_zero › L232-241
```
    fn egglog_mul_zero() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a * 0 should be optimized to 0.
        let result = (&a * &zero).realize()?;
        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);
        Ok(())
    }
```

### mod lazy_tests › fn egglog_double_neg › L244-252
```
    fn egglog_double_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0], vec![3]);
        // neg(neg(a)) should be optimized to just a.
        let result = a.neg().neg().realize()?;
        assert_eq!(result.data(), &[1.0, -2.0, 3.0]);
        Ok(())
    }
```

### mod lazy_tests › fn egglog_exp_ln_inverse › L255-263
```
    fn egglog_exp_ln_inverse() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        // exp(ln(a)) should be optimized to just a.
        let result = a.ln().exp().realize()?;
        assert!(approx_eq(result.data(), &[1.0, 2.0, 3.0], 1e-5));
        Ok(())
    }
```


**Tool Call: Search files for regex `render_optimized_dag|render_optimized_fused_dag` (case-sensitive)**
Status: Completed

Found 2 matches:

## Matches in venum/src/core/lazy/tensor.rs

### impl Tensor › pub fn render_optimized_dag › L459-463
```
    }

    pub fn render_optimized_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
```

### impl Tensor › pub fn render_optimized_fused_dag › L465-469
```
    }

    pub fn render_optimized_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
```


**Tool Call: Read file `src/core/lazy/tests.rs` (lines 1-520)**
Status: Completed

```json
{
  "Text": "#[cfg(test)]\nmod lazy_tests {\n    use crate::{Context, EagerTensor, Tensor};\n    use anyhow::Result;\n\n    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {\n        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)\n    }\n\n    #[test]\n    fn realize_leaf() -> Result<()> {\n        let cx = Context::new();\n\n        let data = vec![1.0, 2.0, 3.0, 4.0];\n        let dt = Tensor::from_slice(&cx, &data, vec![4]);\n        let result = dt.realize()?;\n\n        assert_eq!(result.data(), &data);\n        assert_eq!(result.shape(), &[4]);\n        Ok(())\n    }\n\n    #[test]\n    fn add_two_tensors() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);\n        let c = &a + &b;\n        let result = c.realize()?;\n\n        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn mul_two_tensors() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[10.0, 10.0, 10.0, 10.0], vec![4]);\n        let c = &a * &b;\n        let result = c.realize()?;\n\n        assert_eq!(result.data(), &[20.0, 30.0, 40.0, 50.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn sub_and_div() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);\n\n        let sub_result = (&a - &b).realize()?;\n        assert_eq!(sub_result.data(), &[9.0, 18.0, 27.0, 36.0]);\n\n        let div_result = (&a / &b).realize()?;\n        assert_eq!(div_result.data(), &[10.0, 10.0, 10.0, 10.0]);\n\n        Ok(())\n    }\n\n    #[test]\n    fn fused_add_mul() -> Result<()> {\n        let cx = Context::new();\n\n        // (a + b) * a should produce ONE fused kernel.\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);\n        let c = &(&a + &b) * &a;\n        let result = c.realize()?;\n\n        // (1+10)*1=11, (2+20)*2=44, (3+30)*3=99, (4+40)*4=176\n        assert_eq!(result.data(), &[11.0, 44.0, 99.0, 176.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn unary_neg() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0, -4.0], vec![4]);\n        let result = a.neg().realize()?;\n\n        assert_eq!(result.data(), &[-1.0, 2.0, -3.0, 4.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn unary_exp() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[0.0, 1.0, 2.0], vec![3]);\n        let result = a.exp().realize()?;\n\n        let expected: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();\n        assert!(approx_eq(result.data(), &expected, 1e-5));\n        Ok(())\n    }\n\n    #[test]\n    fn unary_ln() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.718_281_7, 7.389056], vec![3]);\n        let result = a.ln().realize()?;\n\n        let expected: Vec<f32> = [1.0f32, 2.718_281_7, 7.389056]\n            .iter()\n            .map(|x| x.ln())\n            .collect();\n        assert!(approx_eq(result.data(), &expected, 1e-4));\n        Ok(())\n    }\n\n    #[test]\n    fn unary_sqrt() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 4.0, 9.0, 16.0], vec![4]);\n        let result = a.sqrt().realize()?;\n\n        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn chain_fused_unary_binary() -> Result<()> {\n        let cx = Context::new();\n\n        // exp(a) + b — should fuse into one kernel.\n        let a = Tensor::from_slice(&cx, &[0.0, 0.0, 0.0], vec![3]);\n        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        let c = &a.exp() + &b;\n        let result = c.realize()?;\n\n        // exp(0) + 1 = 2, exp(0) + 2 = 3, exp(0) + 3 = 4\n        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn from_eager_tensor() -> Result<()> {\n        let cx = Context::new();\n\n        let eager = EagerTensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;\n        let dt = Tensor::from_tensor(&cx, &eager);\n        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);\n        let c = &dt + &b;\n        let result = c.realize()?;\n\n        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);\n        assert_eq!(result.shape(), &[2, 2]);\n        Ok(())\n    }\n\n    #[test]\n    fn larger_tensor() -> Result<()> {\n        let cx = Context::new();\n\n        let n = 1024;\n        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();\n        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();\n\n        let a = Tensor::from_slice(&cx, &a_data, vec![n]);\n        let b = Tensor::from_slice(&cx, &b_data, vec![n]);\n        let c = &a + &b;\n        let result = c.realize()?;\n\n        let expected: Vec<f32> = (0..n).map(|_| n as f32).collect();\n        assert_eq!(result.data(), expected.as_slice());\n        Ok(())\n    }\n\n    #[test]\n    fn deep_fusion_chain() -> Result<()> {\n        let cx = Context::new();\n\n        // a * b + a - b should all fuse into one kernel.\n        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);\n        let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);\n\n        let c = &(&(&a * &b) + &a) - &b;\n        let result = c.realize()?;\n\n        // (2*1 + 2 - 1) = 3, (3*1 + 3 - 1) = 5, (4*1 + 4 - 1) = 7, (5*1 + 5 - 1) = 9\n        assert_eq!(result.data(), &[3.0, 5.0, 7.0, 9.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn constant_tensor() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        let c = Tensor::constant(&cx, 10.0, vec![3]);\n        let result = (&a + &c).realize()?;\n\n        assert_eq!(result.data(), &[11.0, 12.0, 13.0]);\n        Ok(())\n    }\n\n    // --- egglog optimization tests ---\n\n    #[test]\n    fn egglog_add_zero_identity() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        let zero = Tensor::constant(&cx, 0.0, vec![3]);\n        // a + 0 should be optimized to just a.\n        let result = (&a + &zero).realize()?;\n        assert_eq!(result.data(), &[1.0, 2.0, 3.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_mul_one_identity() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);\n        let one = Tensor::constant(&cx, 1.0, vec![3]);\n        // a * 1 should be optimized to just a.\n        let result = (&a * &one).realize()?;\n        assert_eq!(result.data(), &[5.0, 10.0, 15.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_mul_zero() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);\n        let zero = Tensor::constant(&cx, 0.0, vec![3]);\n        // a * 0 should be optimized to 0.\n        let result = (&a * &zero).realize()?;\n        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_double_neg() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0], vec![3]);\n        // neg(neg(a)) should be optimized to just a.\n        let result = a.neg().neg().realize()?;\n        assert_eq!(result.data(), &[1.0, -2.0, 3.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_exp_ln_inverse() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);\n        // exp(ln(a)) should be optimized to just a.\n        let result = a.ln().exp().realize()?;\n        assert!(approx_eq(result.data(), &[1.0, 2.0, 3.0], 1e-5));\n        Ok(())\n    }\n\n    #[test]\n    fn egglog_sub_self() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);\n        // a - a should be optimized to 0.\n        let result = (&a - &a).realize()?;\n        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);\n        Ok(())\n    }\n\n    #[test]\n    fn realized_to_eager() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.realize()?;\n        let e = r.to_eager()?;\n\n        assert_eq!(e.data().as_ref(), &[1.0, 2.0, 3.0, 4.0]);\n        assert_eq!(e.sizes(), &[2, 2]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_sum_dims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.sum_dims(vec![1], false)?.realize()?;\n\n        assert_eq!(r.data(), &[3.0, 7.0]);\n        assert_eq!(r.shape(), &[2]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_product_dims_keepdims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.product_dims(vec![0], true)?.realize()?;\n\n        assert_eq!(r.data(), &[3.0, 8.0]);\n        assert_eq!(r.shape(), &[1, 2]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_max_dims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);\n        let r = a.max_dims(vec![0], false)?.realize()?;\n\n        assert_eq!(r.data(), &[4.0, 9.0, 6.0]);\n        assert_eq!(r.shape(), &[3]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_min_dims_keepdims() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);\n        let r = a.min_dims(vec![1], true)?.realize()?;\n\n        assert_eq!(r.data(), &[1.0, 4.0]);\n        assert_eq!(r.shape(), &[2, 1]);\n        Ok(())\n    }\n\n    #[test]\n    fn reduce_sum_all() -> Result<()> {\n        let cx = Context::new();\n\n        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);\n        let r = a.sum()?.realize()?;\n\n        assert_eq!(r.data(), &[10.0]);\n        assert_eq!(r.shape(), &[1, 1]);\n        Ok(())\n    }\n}\n"
}
```

**Tool Call: src/core/lazy/tensor.rs**
Status: Completed

Diff: venum/src/core/lazy/tensor.rs
```
use anyhow::{anyhow, bail, Result};
use std::sync::Arc;
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
};

use crate::core::{
    eager::ETensor as EagerTensor,
    errors::{ExpansionError, ReshapeError, TransposeError, UnsqueezeError},
    iters::Indexer,
};

use super::context::Context;
use super::graph::{Graph, NodeId, Op};
use super::jit::compile_kernel;
use super::optimize;
use super::render;
use super::schedule::{build_schedule, ScheduleItem};

#[derive(Clone, Debug, PartialEq)]
pub struct RealizedTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl RealizedTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_eager(&self) -> Result<EagerTensor<f32>> {
        EagerTensor::new(&self.data, &self.shape)
    }
}

#[derive(Clone)]
pub struct Tensor {
    graph: Arc<std::sync::Mutex<Graph>>,
    id: NodeId,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn from_tensor(cx: &Context, tensor: &EagerTensor<f32>) -> Self {
        let data = Arc::new(tensor.data().to_vec());
        let shape = tensor.sizes().to_vec();

        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.load(data, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn from_slice(cx: &Context, data: &[f32], shape: Vec<usize>) -> Self {
        let data = Arc::new(data.to_vec());

        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.load(data, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn constant(cx: &Context, value: f32, shape: Vec<usize>) -> Self {
        let graph = cx.graph();
        let id = {
            let mut g = graph.lock().unwrap();
            g.constant(value, shape.clone())
        };

        Self { graph, id, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // -------- elementwise lazy ops --------

    fn binary_op(&self, rhs: &Tensor, op: Op) -> Tensor {
        let same_graph = Arc::ptr_eq(&self.graph, &rhs.graph);
        let mut graph = self.graph.lock().unwrap();

        let rhs_id = if same_graph {
            rhs.id
        } else {
            let rhs_graph = rhs.graph.lock().unwrap();
            import_subgraph(&rhs_graph, rhs.id, &mut graph)
        };

        let id = graph.binary(op, self.id, rhs_id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    fn unary_op(&self, op: Op) -> Tensor {
        let mut graph = self.graph.lock().unwrap();
        let id = graph.unary(op, self.id);
        let shape = graph.node(id).shape.clone();
        drop(graph);

        Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        }
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(Op::Exp)
    }

    pub fn ln(&self) -> Tensor {
        self.unary_op(Op::Ln)
    }

    pub fn sqrt(&self) -> Tensor {
        self.unary_op(Op::Sqrt)
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }

    // -------- shape lazy ops --------

    pub fn reshape(&self, sizes: Vec<usize>) -> Result<Tensor> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = sizes.iter().product();
        if old_numel != new_numel {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: sizes
            });
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.reshape(self.id, sizes.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: sizes,
        })
    }

    pub fn permute(&self, permutation: Vec<usize>) -> Result<Tensor> {
        if permutation.len() != self.shape.len() {
            bail!(ReshapeError {
                current_shape: self.shape.clone(),
                new_shape: self.shape.clone()
            });
        }

        let mut seen = vec![false; permutation.len()];
        for &p in &permutation {
            if p >= permutation.len() || seen[p] {
                bail!(ReshapeError {
                    current_shape: self.shape.clone(),
                    new_shape: self.shape.clone()
                });
            }
            seen[p] = true;
        }

        let shape = permutation
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();

        let mut graph = self.graph.lock().unwrap();
        let id = graph.permute(self.id, permutation, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn transpose(&self, dim_1: usize, dim_2: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if rank < 2 || dim_1 >= rank || dim_2 >= rank {
            bail!(TransposeError);
        }

        let mut shape = self.shape.clone();
        shape.swap(dim_1, dim_2);

        let mut graph = self.graph.lock().unwrap();
        let id = graph.transpose(self.id, dim_1, dim_2, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn expand(&self, expansions: Vec<usize>) -> Result<Tensor> {
        if expansions.len() != self.shape.len() {
            bail!(ExpansionError {
                size: self.shape.len(),
                expansion: expansions.len()
            });
        }

        for (&size, &exp) in self.shape.iter().zip(expansions.iter()) {
            if !(exp == size || size == 1) {
                bail!(ExpansionError {
                    size,
                    expansion: exp
                });
            }
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.expand(self.id, expansions.clone(), expansions.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: expansions,
        })
    }

    pub fn slice(&self, ranges: Vec<(usize, usize)>) -> Result<Tensor> {
        if ranges.len() != self.shape.len() {
            bail!(anyhow!(
                "slice ranges rank mismatch: got {}, expected {}",
                ranges.len(),
                self.shape.len()
            ));
        }

        let mut out = Vec::with_capacity(self.shape.len());
        for (dim, &(start, end_raw)) in ranges.iter().enumerate() {
            let size = self.shape[dim];
            let end = if end_raw == 0 { size } else { end_raw };
            if start > end || end > size {
                bail!(anyhow!(
                    "slice range {:?} out of bounds for dim {} with size {}",
                    (start, end),
                    dim,
                    size
                ));
            }
            out.push(end - start);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.slice(self.id, ranges, out.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: out,
        })
    }

    pub fn flip(&self, flips: Vec<usize>) -> Result<Tensor> {
        for &d in &flips {
            if d >= self.shape.len() {
                bail!(anyhow!(
                    "flip dimension {} out of bounds for rank {}",
                    d,
                    self.shape.len()
                ));
            }
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.flip(self.id, flips, self.shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape: self.shape.clone(),
        })
    }

    pub fn squeeze(&self) -> Result<Tensor> {
        let mut shape: Vec<usize> = self.shape.iter().copied().filter(|&s| s != 1).collect();
        if shape.is_empty() {
            shape.push(1);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.squeeze(self.id, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn unsqueeze(&self, new_rank: usize) -> Result<Tensor> {
        let rank = self.shape.len();
        if new_rank < rank {
            bail!(UnsqueezeError {
                current: rank,
                new_rank
            });
        }
        if new_rank == rank {
            return Ok(self.clone());
        }

        let mut shape = vec![1; new_rank - rank];
        shape.extend_from_slice(&self.shape);

        let mut graph = self.graph.lock().unwrap();
        let id = graph.unsqueeze(self.id, new_rank, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn pad(&self, constant: f32, padding: Vec<(usize, usize)>) -> Result<Tensor> {
        let mut pad = padding;
        pad.resize(self.shape.len(), (0, 0));

        let mut shape = Vec::with_capacity(self.shape.len());
        for (s, (l, r)) in self.shape.iter().copied().zip(pad.iter().copied()) {
            shape.push(l + s + r);
        }

        let mut graph = self.graph.lock().unwrap();
        let id = graph.pad(self.id, constant, pad, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    // -------- reduce lazy ops --------

    pub fn sum_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.sum(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn product_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.prod(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn max_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.max(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn min_dims(&self, dimensions: Vec<usize>, keepdims: bool) -> Result<Tensor> {
        let shape = reduced_shape(&self.shape, &dimensions, keepdims)?;
        let mut graph = self.graph.lock().unwrap();
        let id = graph.min(self.id, dimensions, keepdims, shape.clone());
        drop(graph);

        Ok(Tensor {
            graph: Arc::clone(&self.graph),
            id,
            shape,
        })
    }

    pub fn sum(&self) -> Result<Tensor> {
        self.sum_dims((0..self.shape.len()).collect(), true)
    }

    pub fn product(&self) -> Result<Tensor> {
        self.product_dims((0..self.shape.len()).collect(), true)
    }

    pub fn max(&self) -> Result<Tensor> {
        self.max_dims((0..self.shape.len()).collect(), true)
    }

    pub fn min(&self) -> Result<Tensor> {
        self.min_dims((0..self.shape.len()).collect(), true)
    }

    // -------- visualization --------

    pub fn render_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_dag(&graph, self.id)
    }

    pub fn render_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        render::render_fused_dag(&graph, self.id)
    }

    pub fn render_optimized_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        if !is_optimize_safe(&graph, self.id) {
            return render::render_dag(&graph, self.id);
        }
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_dag(&opt_graph, opt_root)
    }

    pub fn render_optimized_fused_dag(&self) -> String {
        let graph = self.graph.lock().unwrap();
        if !is_optimize_safe(&graph, self.id) {
            return render::render_fused_dag(&graph, self.id);
        }
        let (opt_graph, opt_root) = optimize::optimize(&graph, self.id);
        render::render_fused_dag(&opt_graph, opt_root)
    }

    // -------- realize --------

    pub fn realize(&self) -> Result<RealizedTensor> {
        let graph = self.graph.lock().unwrap();

        // Fast path: already-backed leaf.
        let root_node = graph.node(self.id);
        if let Some(ref buffer) = root_node.buffer {
            let data = buffer.as_f32().to_vec();
            let shape = root_node.shape.clone();
            drop(graph);
            return Ok(RealizedTensor::new(data, shape));
        }

        // Only optimize pure elementwise roots for now.
        let optimize_safe = is_optimize_safe(&graph, self.id);
        let (exec_graph, exec_root) = if optimize_safe {
            optimize::optimize(&graph, self.id)
        } else {
            clone_reachable_subgraph(&graph, self.id)
        };
        drop(graph);

        let graph = &exec_graph;
        let root = exec_root;

        let root_node = graph.node(root);
        if let Some(ref buffer) = root_node.buffer {
            return Ok(RealizedTensor::new(
                buffer.as_f32().to_vec(),
                root_node.shape.clone(),
            ));
        }
        if let Op::Const(val) = root_node.op {
            let numel: usize = self.shape.iter().product();
            return Ok(RealizedTensor::new(vec![val; numel], self.shape.clone()));
        }

        let schedule = build_schedule(graph, root);
        let mut realized: std::collections::HashMap<NodeId, Vec<f32>> =
            std::collections::HashMap::new();

        for item in &schedule {
            match item {
                ScheduleItem::Fused(kernel) => {
                    let compiled = compile_kernel(graph, kernel)?;

                    let input_ptrs: Vec<*const f32> = kernel
                        .input_buffers
                        .iter()
                        .map(|&buf_id| {
                            let node = graph.node(buf_id);
                            if let Some(ref buffer) = node.buffer {
                                buffer.as_f32_ptr()
                            } else if let Some(data) = realized.get(&buf_id) {
                                data.as_ptr()
                            } else {
                                panic!("Input buffer {:?} not realized and has no data", buf_id);
                            }
                        })
                        .collect();

                    let mut output = vec![0.0f32; kernel.numel];
                    unsafe {
                        compiled.execute(&input_ptrs, output.as_mut_ptr(), kernel.numel);
                    }
                    realized.insert(kernel.root, output);
                }
                ScheduleItem::Shape(shape_item) => {
                    let input_node = graph.node(shape_item.input);
                    let input_shape = input_node.shape.clone();

                    let input_data = if let Some(data) = realized.get(&shape_item.input) {
                        data.clone()
                    } else if let Some(ref buf) = input_node.buffer {
                        buf.as_f32().to_vec()
                    } else {
                        bail!("Shape op input {:?} is not realized", shape_item.input);
                    };

                    let output = execute_shape_op(
                        &shape_item.op,
                        &input_data,
                        &input_shape,
                        &shape_item.shape,
                    )?;
                    realized.insert(shape_item.root, output);
                }
                ScheduleItem::Reduce(reduce_item) => {
                    let input_node = graph.node(reduce_item.input);
                    let input_shape = input_node.shape.clone();

                    let input_data = if let Some(data) = realized.get(&reduce_item.input) {
                        data.clone()
                    } else if let Some(ref buf) = input_node.buffer {
                        buf.as_f32().to_vec()
                    } else {
                        bail!("Reduce op input {:?} is not realized", reduce_item.input);
                    };

                    let output = execute_reduce_op(
                        &reduce_item.op,
                        &input_data,
                        &input_shape,
                        &reduce_item.shape,
                    )?;
                    realized.insert(reduce_item.root, output);
                }
            }
        }

        let data = realized
            .remove(&root)
            .ok_or_else(|| anyhow!("Root node was not realized"))?;

        Ok(RealizedTensor::new(data, graph.node(root).shape.clone()))
    }

    pub fn realize_eager(&self) -> Result<EagerTensor<f32>> {
        self.realize()?.to_eager()
    }
}

// -------- shape-op execution helpers --------

fn execute_shape_op(
    op: &Op,
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    match op {
        Op::Reshape(_) => execute_reshape(input_data, input_shape, output_shape),
        Op::Permute(perm) => execute_permute(input_data, input_shape, output_shape, perm),
        Op::Transpose(d1, d2) => execute_transpose(input_data, input_shape, output_shape, *d1, *d2),
        Op::Expand(_) => execute_expand(input_data, input_shape, output_shape),
        Op::Slice(ranges) => execute_slice(input_data, input_shape, output_shape, ranges),
        Op::Flip(flips) => execute_flip(input_data, input_shape, output_shape, flips),
        Op::Squeeze => execute_reshape(input_data, input_shape, output_shape),
        Op::Unsqueeze(_) => execute_reshape(input_data, input_shape, output_shape),
        Op::Pad(constant, padding) => {
            execute_pad(input_data, input_shape, output_shape, *constant, padding)
        }
        _ => bail!("execute_shape_op called with non-shape op: {:?}", op),
    }
}

fn execute_reduce_op(
    op: &Op,
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    match op {
        Op::Sum(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().sum(),
        ),
        Op::Prod(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| slice.iter().copied().product(),
        ),
        Op::Max(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(0.0)
            },
        ),
        Op::Min(dimensions, keepdims) => execute_reduce(
            input_data,
            input_shape,
            output_shape,
            dimensions,
            *keepdims,
            |slice| {
                slice
                    .iter()
                    .copied()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap_or(0.0)
            },
        ),
        _ => bail!("execute_reduce_op called with non-reduce op: {:?}", op),
    }
}

fn execute_reduce(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    dimensions: &[usize],
    keepdims: bool,
    reducer: impl Fn(&[f32]) -> f32,
) -> Result<Vec<f32>> {
    validate_reduce_dims(input_shape, dimensions)?;

    let expected_shape = reduced_shape(input_shape, dimensions, keepdims)?;
    if expected_shape != output_shape {
        bail!(
            "reduce output shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            output_shape
        );
    }

    let mut out = Vec::with_capacity(output_shape.iter().product());

    for out_idx in Indexer::new(output_shape) {
        let mut fixed = vec![None; input_shape.len()];

        if keepdims {
            for d in 0..input_shape.len() {
                if !dimensions.contains(&d) {
                    fixed[d] = Some(out_idx[d]);
                }
            }
        } else {
            let mut out_pos = 0usize;
            for d in 0..input_shape.len() {
                if !dimensions.contains(&d) {
                    fixed[d] = Some(out_idx[out_pos]);
                    out_pos += 1;
                }
            }
        }

        let mut reduced_values = Vec::new();
        collect_reduce_values(
            input_data,
            input_shape,
            dimensions,
            &fixed,
            0,
            &mut vec![0; input_shape.len()],
            &mut reduced_values,
        );

        out.push(reducer(&reduced_values));
    }

    Ok(out)
}

fn collect_reduce_values(
    input_data: &[f32],
    input_shape: &[usize],
    dimensions: &[usize],
    fixed: &[Option<usize>],
    dim: usize,
    current: &mut Vec<usize>,
    values: &mut Vec<f32>,
) {
    if dim == input_shape.len() {
        let off = idx_to_offset(current, input_shape);
        values.push(input_data[off]);
        return;
    }

    if dimensions.contains(&dim) {
        for i in 0..input_shape[dim] {
            current[dim] = i;
            collect_reduce_values(
                input_data,
                input_shape,
                dimensions,
                fixed,
                dim + 1,
                current,
                values,
            );
        }
    } else if let Some(idx) = fixed[dim] {
        current[dim] = idx;
        collect_reduce_values(
            input_data,
            input_shape,
            dimensions,
            fixed,
            dim + 1,
            current,
            values,
        );
    }
}

fn reduced_shape(shape: &[usize], dimensions: &[usize], keepdims: bool) -> Result<Vec<usize>> {
    validate_reduce_dims(shape, dimensions)?;

    let mut out = Vec::new();
    for (d, &size) in shape.iter().enumerate() {
        if dimensions.contains(&d) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(size);
        }
    }

    if out.is_empty() {
        out.push(1);
    }

    Ok(out)
}

fn validate_reduce_dims(shape: &[usize], dimensions: &[usize]) -> Result<()> {
    let mut seen = vec![false; shape.len()];
    for &d in dimensions {
        if d >= shape.len() {
            bail!(
                "reduce dimension {} out of bounds for rank {}",
                d,
                shape.len()
            );
        }
        if seen[d] {
            bail!("duplicate reduce dimension {}", d);
        }
        seen[d] = true;
    }
    Ok(())
}

fn execute_reshape(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    let in_numel: usize = input_shape.iter().product();
    let out_numel: usize = output_shape.iter().product();
    if in_numel != out_numel || input_data.len() != in_numel {
        bail!(
            "invalid reshape numel: input {:?}, output {:?}",
            input_shape,
            output_shape
        );
    }
    Ok(input_data.to_vec())
}

fn execute_permute(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    perm: &[usize],
) -> Result<Vec<f32>> {
    if perm.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("invalid permute rank");
    }
    let mut out = vec![0.0f32; output_shape.iter().product()];

    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for (out_dim, &in_dim) in perm.iter().enumerate() {
            in_idx[in_dim] = out_idx[out_dim];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_transpose(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    dim_1: usize,
    dim_2: usize,
) -> Result<Vec<f32>> {
    if dim_1 >= input_shape.len()
        || dim_2 >= input_shape.len()
        || input_shape.len() != output_shape.len()
    {
        bail!("invalid transpose dims");
    }

    let mut perm: Vec<usize> = (0..input_shape.len()).collect();
    perm.swap(dim_1, dim_2);
    execute_permute(input_data, input_shape, output_shape, &perm)
}

fn execute_expand(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    if input_shape.len() != output_shape.len() {
        bail!("expand rank mismatch");
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            if input_shape[d] == output_shape[d] {
                in_idx[d] = out_idx[d];
            } else if input_shape[d] == 1 {
                in_idx[d] = 0;
            } else {
                bail!(
                    "cannot expand dim {} from {} to {}",
                    d,
                    input_shape[d],
                    output_shape[d]
                );
            }
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_slice(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<f32>> {
    if ranges.len() != input_shape.len() || output_shape.len() != input_shape.len() {
        bail!("slice rank mismatch");
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            let (start, end_raw) = ranges[d];
            let end = if end_raw == 0 {
                input_shape[d]
            } else {
                end_raw
            };
            if start > end || end > input_shape[d] {
                bail!("invalid slice range {:?} for dim {}", ranges[d], d);
            }
            in_idx[d] = start + out_idx[d];
        }
        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_flip(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    flips: &[usize],
) -> Result<Vec<f32>> {
    if input_shape != output_shape {
        bail!("flip must preserve shape");
    }

    let mut flip_mask = vec![false; input_shape.len()];
    for &d in flips {
        if d >= input_shape.len() {
            bail!("flip dim {} out of range", d);
        }
        flip_mask[d] = true;
    }

    let mut out = vec![0.0f32; output_shape.iter().product()];
    for out_idx in Indexer::new(output_shape) {
        let mut in_idx = out_idx.clone();
        for d in 0..in_idx.len() {
            if flip_mask[d] {
                in_idx[d] = input_shape[d] - 1 - in_idx[d];
            }
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn execute_pad(
    input_data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    constant: f32,
    padding: &[(usize, usize)],
) -> Result<Vec<f32>> {
    if input_shape.len() != output_shape.len() {
        bail!("pad rank mismatch");
    }

    let mut pad = padding.to_vec();
    pad.resize(input_shape.len(), (0, 0));

    let mut expected = Vec::with_capacity(input_shape.len());
    for d in 0..input_shape.len() {
        expected.push(pad[d].0 + input_shape[d] + pad[d].1);
    }
    if expected != output_shape {
        bail!(
            "pad output shape mismatch: expected {:?}, got {:?}",
            expected,
            output_shape
        );
    }

    let mut out = vec![constant; output_shape.iter().product()];
    for in_idx in Indexer::new(input_shape) {
        let mut out_idx = vec![0usize; input_shape.len()];
        for d in 0..input_shape.len() {
            out_idx[d] = pad[d].0 + in_idx[d];
        }

        let in_off = idx_to_offset(&in_idx, input_shape);
        let out_off = idx_to_offset(&out_idx, output_shape);
        out[out_off] = input_data[in_off];
    }

    Ok(out)
}

fn idx_to_offset(index: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut off = 0usize;
    for d in (0..shape.len()).rev() {
        off += index[d] * stride;
        stride *= shape[d];
    }
    off
}

// -------- graph import helpers --------

fn import_subgraph(src_graph: &Graph, src_id: NodeId, dst_graph: &mut Graph) -> NodeId {
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();

    let mut buffer_map: std::collections::HashMap<*const Vec<f32>, NodeId> =
        std::collections::HashMap::new();
    for (i, node) in dst_graph.nodes.iter().enumerate() {
        if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
            buffer_map.insert(Arc::as_ptr(arc), NodeId(i));
        }
    }

    import_node(src_graph, src_id, dst_graph, &mut id_map, &buffer_map)
}

fn import_node(
    src_graph: &Graph,
    src_id: NodeId,
    dst_graph: &mut Graph,
    id_map: &mut std::collections::HashMap<NodeId, NodeId>,
    buffer_map: &std::collections::HashMap<*const Vec<f32>, NodeId>,
) -> NodeId {
    if let Some(&mapped) = id_map.get(&src_id) {
        return mapped;
    }

    let node = src_graph.node(src_id);

    if let (Op::Load, Some(super::dtype::Buffer::F32(ref arc))) = (&node.op, &node.buffer) {
        if let Some(&existing_id) = buffer_map.get(&Arc::as_ptr(arc)) {
            id_map.insert(src_id, existing_id);
            return existing_id;
        }
    }

    let new_inputs: Vec<NodeId> = node
        .inputs
        .iter()
        .map(|&input_id| import_node(src_graph, input_id, dst_graph, id_map, buffer_map))
        .collect();

    let new_id = dst_graph.add_node(super::graph::Node {
        op: node.op.clone(),
        inputs: new_inputs,
        shape: node.shape.clone(),
        dtype: node.dtype,
        buffer: node.buffer.clone(),
    });

    id_map.insert(src_id, new_id);
    new_id
}

// -------- optimization helpers --------

fn is_optimize_safe(graph: &Graph, root: NodeId) -> bool {
    fn dfs(graph: &Graph, id: NodeId, seen: &mut std::collections::HashSet<NodeId>) -> bool {
        if !seen.insert(id) {
            return true;
        }
        let node = graph.node(id);
        let here_ok = matches!(
            node.op,
            Op::Load
                | Op::Const(_)
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Exp
                | Op::Ln
                | Op::Sqrt
                | Op::Neg
        );
        if !here_ok {
            return false;
        }
        node.inputs.iter().all(|&inp| dfs(graph, inp, seen))
    }

    let mut seen = std::collections::HashSet::new();
    dfs(graph, root, &mut seen)
}

fn clone_reachable_subgraph(src: &Graph, root: NodeId) -> (Graph, NodeId) {
    let mut dst = Graph::new();
    let mut id_map = std::collections::HashMap::new();
    let buffer_map = std::collections::HashMap::new();
    let new_root = import_node(src, root, &mut dst, &mut id_map, &buffer_map);
    (dst, new_root)
}

// -------- operator overloads --------

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Add)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Sub)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Mul)
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.binary_op(rhs, Op::Div)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Add)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Sub)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Mul)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        self.binary_op(&rhs, Op::Div)
    }
}

```


**Tool Call: src/core/lazy/tests.rs**
Status: Completed

Diff: venum/src/core/lazy/tests.rs
```
#[cfg(test)]
mod lazy_tests {
    use crate::{Context, EagerTensor, Tensor};
    use anyhow::Result;

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn realize_leaf() -> Result<()> {
        let cx = Context::new();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let dt = Tensor::from_slice(&cx, &data, vec![4]);
        let result = dt.realize()?;

        assert_eq!(result.data(), &data);
        assert_eq!(result.shape(), &[4]);
        Ok(())
    }

    #[test]
    fn add_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &a + &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn mul_two_tensors() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 10.0, 10.0, 10.0], vec![4]);
        let c = &a * &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[20.0, 30.0, 40.0, 50.0]);
        Ok(())
    }

    #[test]
    fn sub_and_div() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);

        let sub_result = (&a - &b).realize()?;
        assert_eq!(sub_result.data(), &[9.0, 18.0, 27.0, 36.0]);

        let div_result = (&a / &b).realize()?;
        assert_eq!(div_result.data(), &[10.0, 10.0, 10.0, 10.0]);

        Ok(())
    }

    #[test]
    fn fused_add_mul() -> Result<()> {
        let cx = Context::new();

        // (a + b) * a should produce ONE fused kernel.
        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![4]);
        let c = &(&a + &b) * &a;
        let result = c.realize()?;

        // (1+10)*1=11, (2+20)*2=44, (3+30)*3=99, (4+40)*4=176
        assert_eq!(result.data(), &[11.0, 44.0, 99.0, 176.0]);
        Ok(())
    }

    #[test]
    fn unary_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0, -4.0], vec![4]);
        let result = a.neg().realize()?;

        assert_eq!(result.data(), &[-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn unary_exp() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[0.0, 1.0, 2.0], vec![3]);
        let result = a.exp().realize()?;

        let expected: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|x| x.exp()).collect();
        assert!(approx_eq(result.data(), &expected, 1e-5));
        Ok(())
    }

    #[test]
    fn unary_ln() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.718_281_7, 7.389056], vec![3]);
        let result = a.ln().realize()?;

        let expected: Vec<f32> = [1.0f32, 2.718_281_7, 7.389056]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert!(approx_eq(result.data(), &expected, 1e-4));
        Ok(())
    }

    #[test]
    fn unary_sqrt() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 4.0, 9.0, 16.0], vec![4]);
        let result = a.sqrt().realize()?;

        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn chain_fused_unary_binary() -> Result<()> {
        let cx = Context::new();

        // exp(a) + b — should fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[0.0, 0.0, 0.0], vec![3]);
        let b = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = &a.exp() + &b;
        let result = c.realize()?;

        // exp(0) + 1 = 2, exp(0) + 2 = 3, exp(0) + 3 = 4
        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn from_eager_tensor() -> Result<()> {
        let cx = Context::new();

        let eager = EagerTensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
        let dt = Tensor::from_tensor(&cx, &eager);
        let b = Tensor::from_slice(&cx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let c = &dt + &b;
        let result = c.realize()?;

        assert_eq!(result.data(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(result.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn larger_tensor() -> Result<()> {
        let cx = Context::new();

        let n = 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let a = Tensor::from_slice(&cx, &a_data, vec![n]);
        let b = Tensor::from_slice(&cx, &b_data, vec![n]);
        let c = &a + &b;
        let result = c.realize()?;

        let expected: Vec<f32> = (0..n).map(|_| n as f32).collect();
        assert_eq!(result.data(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn deep_fusion_chain() -> Result<()> {
        let cx = Context::new();

        // a * b + a - b should all fuse into one kernel.
        let a = Tensor::from_slice(&cx, &[2.0, 3.0, 4.0, 5.0], vec![4]);
        let b = Tensor::from_slice(&cx, &[1.0, 1.0, 1.0, 1.0], vec![4]);

        let c = &(&(&a * &b) + &a) - &b;
        let result = c.realize()?;

        // (2*1 + 2 - 1) = 3, (3*1 + 3 - 1) = 5, (4*1 + 4 - 1) = 7, (5*1 + 5 - 1) = 9
        assert_eq!(result.data(), &[3.0, 5.0, 7.0, 9.0]);
        Ok(())
    }

    #[test]
    fn constant_tensor() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let c = Tensor::constant(&cx, 10.0, vec![3]);
        let result = (&a + &c).realize()?;

        assert_eq!(result.data(), &[11.0, 12.0, 13.0]);
        Ok(())
    }

    // --- egglog optimization tests ---

    #[test]
    fn egglog_add_zero_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a + 0 should be optimized to just a.
        let result = (&a + &zero).realize()?;
        assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn egglog_mul_one_identity() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let one = Tensor::constant(&cx, 1.0, vec![3]);
        // a * 1 should be optimized to just a.
        let result = (&a * &one).realize()?;
        assert_eq!(result.data(), &[5.0, 10.0, 15.0]);
        Ok(())
    }

    #[test]
    fn egglog_mul_zero() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        let zero = Tensor::constant(&cx, 0.0, vec![3]);
        // a * 0 should be optimized to 0.
        let result = (&a * &zero).realize()?;
        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn egglog_double_neg() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, -2.0, 3.0], vec![3]);
        // neg(neg(a)) should be optimized to just a.
        let result = a.neg().neg().realize()?;
        assert_eq!(result.data(), &[1.0, -2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn egglog_exp_ln_inverse() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0], vec![3]);
        // exp(ln(a)) should be optimized to just a.
        let result = a.ln().exp().realize()?;
        assert!(approx_eq(result.data(), &[1.0, 2.0, 3.0], 1e-5));
        Ok(())
    }

    #[test]
    fn egglog_sub_self() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[5.0, 10.0, 15.0], vec![3]);
        // a - a should be optimized to 0.
        let result = (&a - &a).realize()?;
        assert_eq!(result.data(), &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn realized_to_eager() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.realize()?;
        let e = r.to_eager()?;

        assert_eq!(e.data().as_ref(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(e.sizes(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn reduce_sum_dims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.sum_dims(vec![1], false)?.realize()?;

        assert_eq!(r.data(), &[3.0, 7.0]);
        assert_eq!(r.shape(), &[2]);
        Ok(())
    }

    #[test]
    fn reduce_product_dims_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.product_dims(vec![0], true)?.realize()?;

        assert_eq!(r.data(), &[3.0, 8.0]);
        assert_eq!(r.shape(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn reduce_max_dims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);
        let r = a.max_dims(vec![0], false)?.realize()?;

        assert_eq!(r.data(), &[4.0, 9.0, 6.0]);
        assert_eq!(r.shape(), &[3]);
        Ok(())
    }

    #[test]
    fn reduce_min_dims_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 9.0, 3.0, 4.0, 7.0, 6.0], vec![2, 3]);
        let r = a.min_dims(vec![1], true)?.realize()?;

        assert_eq!(r.data(), &[1.0, 4.0]);
        assert_eq!(r.shape(), &[2, 1]);
        Ok(())
    }

    #[test]
    fn reduce_sum_all() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.sum()?.realize()?;

        assert_eq!(r.data(), &[10.0]);
        assert_eq!(r.shape(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn shape_then_reduce_sum() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = a.reshape(vec![3, 2])?.sum_dims(vec![1], false)?.realize()?;

        assert_eq!(r.data(), &[3.0, 7.0, 11.0]);
        assert_eq!(r.shape(), &[3]);
        Ok(())
    }

    #[test]
    fn reduce_then_elementwise_add() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let summed = a.sum_dims(vec![1], true)?;
        let bias = Tensor::constant(&cx, 10.0, vec![2, 1]);
        let r = (&summed + &bias).realize()?;

        assert_eq!(r.data(), &[13.0, 17.0]);
        assert_eq!(r.shape(), &[2, 1]);
        Ok(())
    }

    #[test]
    fn permute_then_reduce_keepdims() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = a.permute(vec![1, 0])?.sum_dims(vec![1], true)?.realize()?;

        assert_eq!(r.data(), &[5.0, 7.0, 9.0]);
        assert_eq!(r.shape(), &[3, 1]);
        Ok(())
    }

    #[test]
    fn chained_reduce_keepdims_then_reduce() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a
            .sum_dims(vec![1], true)?
            .sum_dims(vec![0], true)?
            .realize()?;

        assert_eq!(r.data(), &[10.0]);
        assert_eq!(r.shape(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn shape_reduce_elementwise_chain() -> Result<()> {
        let cx = Context::new();

        let a = Tensor::from_slice(&cx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a
            .transpose(0, 1)?
            .sum_dims(vec![1], false)?
            .unsqueeze(2)?
            .realize()?;

        assert_eq!(r.data(), &[4.0, 6.0]);
        assert_eq!(r.shape(), &[1, 2]);
        Ok(())
    }
}

```


**Tool Call: cargo test -q**
Status: Completed

Command "cargo test -q" failed with exit code 101.

```
warning: method `size_bytes` is never used
 --> src/core/lazy/dtype.rs:9:12
  |
8 | impl DType {
  | ---------- method in this implementation
9 |     pub fn size_bytes(self) -> usize {
  |            ^^^^^^^^^^
  |
  = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: methods `dtype` and `len` are never used
  --> src/core/lazy/dtype.rs:22:12
   |
21 | impl Buffer {
   | ----------- methods in this implementation
22 |     pub fn dtype(&self) -> DType {
   |            ^^^^^
...
28 |     pub fn len(&self) -> usize {
   |            ^^^

warning: field `0` is never read
  --> src/core/lazy/graph.rs:27:13
   |
27 |     Reshape(Vec<usize>),
   |     ------- ^^^^^^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
27 -     Reshape(Vec<usize>),
27 +     Reshape(()),
   |

warning: field `0` is never read
  --> src/core/lazy/graph.rs:30:12
   |
30 |     Expand(Vec<usize>),
   |     ------ ^^^^^^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
30 -     Expand(Vec<usize>),
30 +     Expand(()),
   |

warning: field `0` is never read
  --> src/core/lazy/graph.rs:34:15
   |
34 |     Unsqueeze(usize),
   |     --------- ^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
34 -     Unsqueeze(usize),
34 +     Unsqueeze(()),
   |

warning: methods `is_binary` and `is_unary` are never used
  --> src/core/lazy/graph.rs:52:12
   |
44 | impl Op {
   | ------- methods in this implementation
...
52 |     pub fn is_binary(&self) -> bool {
   |            ^^^^^^^^^
...
56 |     pub fn is_unary(&self) -> bool {
   |            ^^^^^^^^

warning: methods `node_mut` and `consumer_count` are never used
   --> src/core/lazy/graph.rs:117:12
    |
102 | impl Graph {
    | ---------- methods in this implementation
...
117 |     pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
    |            ^^^^^^^^
...
340 |     pub fn consumer_count(&self, id: NodeId) -> usize {
    |            ^^^^^^^^^^^^^^

warning: field `sqrtf` is never read
   --> src/core/lazy/jit.rs:283:5
    |
280 | struct MathFuncRefs {
    |        ------------ field in this struct
...
283 |     sqrtf: cranelift_codegen::ir::FuncRef,
    |     ^^^^^


running 36 tests
................ 16/36
core::lazy::tests::lazy_tests::reduce_then_elementwise_add --- FAILED
...................
failures:

---- core::lazy::tests::lazy_tests::reduce_then_elementwise_add stdout ----
Error: Unsupported op in JIT expression builder for node NodeId(1): Sum([1], true)


failures:
    core::lazy::tests::lazy_tests::reduce_then_elementwise_add

test result: FAILED. 35 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s

error: test failed, to rerun pass `--lib`
```

**Tool Call: Check diagnostics for `venum/src/core/lazy/tests.rs`**
Status: Completed

File doesn't have errors or warnings!

**Tool Call: Check diagnostics for `venum/src/core/lazy/tensor.rs`**
Status: Completed

File doesn't have errors or warnings!

**Tool Call: Read file `src/core/lazy/jit.rs` (lines 280-420)**
Status: Completed

```json
{
  "Text": "struct MathFuncRefs {\n    expf: cranelift_codegen::ir::FuncRef,\n    logf: cranelift_codegen::ir::FuncRef,\n    sqrtf: cranelift_codegen::ir::FuncRef,\n}\n\n/// Recursively build the Cranelift IR for the expression tree rooted at `id`.\nfn build_expression(\n    graph: &Graph,\n    id: NodeId,\n    builder: &mut FunctionBuilder,\n    input_ptrs: &[Value],\n    input_index: &HashMap<NodeId, usize>,\n    byte_offset: Value,\n    math: &MathFuncRefs,\n) -> Result<Value> {\n    let node = graph.node(id);\n\n    match &node.op {\n        Op::Load => {\n            let idx = input_index\n                .get(&id)\n                .ok_or_else(|| anyhow::anyhow!(\"Load node {:?} not found in input_index\", id))?;\n            let ptr = input_ptrs[*idx];\n            let addr = builder.ins().iadd(ptr, byte_offset);\n            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))\n        }\n        Op::Const(val) => Ok(builder.ins().f32const(*val)),\n        Op::Add | Op::Sub | Op::Mul | Op::Div => {\n            let lhs = build_expression(\n                graph,\n                node.inputs[0],\n                builder,\n                input_ptrs,\n                input_index,\n                byte_offset,\n                math,\n            )?;\n            let rhs = build_expression(\n                graph,\n                node.inputs[1],\n                builder,\n                input_ptrs,\n                input_index,\n                byte_offset,\n                math,\n            )?;\n            Ok(match node.op {\n                Op::Add => builder.ins().fadd(lhs, rhs),\n                Op::Sub => builder.ins().fsub(lhs, rhs),\n                Op::Mul => builder.ins().fmul(lhs, rhs),\n                Op::Div => builder.ins().fdiv(lhs, rhs),\n                _ => unreachable!(),\n            })\n        }\n        Op::Neg => {\n            let val = build_expression(\n                graph,\n                node.inputs[0],\n                builder,\n                input_ptrs,\n                input_index,\n                byte_offset,\n                math,\n            )?;\n            Ok(builder.ins().fneg(val))\n        }\n        Op::Exp => {\n            let val = build_expression(\n                graph,\n                node.inputs[0],\n                builder,\n                input_ptrs,\n                input_index,\n                byte_offset,\n                math,\n            )?;\n            let call = builder.ins().call(math.expf, &[val]);\n            Ok(builder.inst_results(call)[0])\n        }\n        Op::Ln => {\n            let val = build_expression(\n                graph,\n                node.inputs[0],\n                builder,\n                input_ptrs,\n                input_index,\n                byte_offset,\n                math,\n            )?;\n            let call = builder.ins().call(math.logf, &[val]);\n            Ok(builder.inst_results(call)[0])\n        }\n        Op::Sqrt => {\n            let val = build_expression(\n                graph,\n                node.inputs[0],\n                builder,\n                input_ptrs,\n                input_index,\n                byte_offset,\n                math,\n            )?;\n            Ok(builder.ins().sqrt(val))\n        }\n        _ => Err(anyhow::anyhow!(\n            \"Unsupported op in JIT expression builder for node {:?}: {:?}\",\n            id,\n            node.op\n        )),\n    }\n}\n"
}
```

**Tool Call: src/core/lazy/jit.rs**
Status: Completed

Diff: venum/src/core/lazy/jit.rs
```
use anyhow::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::Function;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

use super::graph::{Graph, NodeId, Op};
use super::schedule::FusedKernel;

/// A compiled kernel ready to execute.
pub struct CompiledKernel {
    /// Number of input buffer pointers.
    pub num_inputs: usize,
    /// The JIT module that owns the compiled code.
    _module: JITModule,
    /// Raw function pointer to the compiled kernel.
    fn_ptr: *const u8,
}

// Safety: The compiled code is immutable once created and the function pointer
// is valid for the lifetime of _module.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

/// Extern "C" functions we link into the JIT for math ops.
extern "C" fn jit_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn jit_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn jit_sqrtf(x: f32) -> f32 {
    x.sqrt()
}

impl CompiledKernel {
    /// Execute the kernel with the given input buffer pointers and output pointer.
    ///
    /// # Safety
    /// - All pointers must be valid and point to buffers of at least `numel` f32 elements.
    /// - `inputs` must have exactly `self.num_inputs` elements.
    pub unsafe fn execute(&self, inputs: &[*const f32], output: *mut f32, numel: usize) {
        // ABI: fn(in0: *const f32, in1: *const f32, ..., out: *mut f32, n: u64)
        match self.num_inputs {
            0 => {
                let f: extern "C" fn(*mut f32, u64) = std::mem::transmute(self.fn_ptr);
                f(output, numel as u64);
            }
            1 => {
                let f: extern "C" fn(*const f32, *mut f32, u64) = std::mem::transmute(self.fn_ptr);
                f(inputs[0], output, numel as u64);
            }
            2 => {
                let f: extern "C" fn(*const f32, *const f32, *mut f32, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], output, numel as u64);
            }
            3 => {
                let f: extern "C" fn(*const f32, *const f32, *const f32, *mut f32, u64) =
                    std::mem::transmute(self.fn_ptr);
                f(inputs[0], inputs[1], inputs[2], output, numel as u64);
            }
            4 => {
                let f: extern "C" fn(
                    *const f32,
                    *const f32,
                    *const f32,
                    *const f32,
                    *mut f32,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    output,
                    numel as u64,
                );
            }
            5 => {
                let f: extern "C" fn(
                    *const f32,
                    *const f32,
                    *const f32,
                    *const f32,
                    *const f32,
                    *mut f32,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    output,
                    numel as u64,
                );
            }
            6 => {
                let f: extern "C" fn(
                    *const f32,
                    *const f32,
                    *const f32,
                    *const f32,
                    *const f32,
                    *const f32,
                    *mut f32,
                    u64,
                ) = std::mem::transmute(self.fn_ptr);
                f(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    output,
                    numel as u64,
                );
            }
            _ => {
                panic!(
                    "CompiledKernel: too many inputs ({}), max 6 supported in dispatch",
                    self.num_inputs
                );
            }
        }
    }
}

/// Compile a fused elementwise kernel into native code via Cranelift.
pub fn compile_kernel(graph: &Graph, kernel: &FusedKernel) -> Result<CompiledKernel> {
    let num_inputs = kernel.input_buffers.len();

    // Map each input buffer NodeId to its parameter index.
    let mut input_index: HashMap<NodeId, usize> = HashMap::new();
    for (i, &buf_id) in kernel.input_buffers.iter().enumerate() {
        input_index.insert(buf_id, i);
    }

    // --- Cranelift setup ---
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().map_err(|e| anyhow::anyhow!("{}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Register math symbols.
    builder.symbol("jit_expf", jit_expf as *const u8);
    builder.symbol("jit_logf", jit_logf as *const u8);
    builder.symbol("jit_sqrtf", jit_sqrtf as *const u8);

    let mut module = JITModule::new(builder);
    let ptr_type = module.target_config().pointer_type();

    // --- Declare math function signatures ---
    let mut math_sig = module.make_signature();
    math_sig.params.push(AbiParam::new(types::F32));
    math_sig.returns.push(AbiParam::new(types::F32));

    let expf_id = module.declare_function("jit_expf", Linkage::Import, &math_sig)?;
    let logf_id = module.declare_function("jit_logf", Linkage::Import, &math_sig)?;
    let sqrtf_id = module.declare_function("jit_sqrtf", Linkage::Import, &math_sig)?;

    // --- Build kernel function signature ---
    // fn(in0: *f32, in1: *f32, ..., out: *mut f32, n: u64)
    let mut sig = module.make_signature();
    for _ in 0..num_inputs {
        sig.params.push(AbiParam::new(ptr_type));
    }
    sig.params.push(AbiParam::new(ptr_type)); // output pointer
    sig.params.push(AbiParam::new(types::I64)); // n

    let func_id = module.declare_function("kernel", Linkage::Local, &sig)?;

    // --- Build function body ---
    let mut func =
        Function::with_name_signature(cranelift_codegen::ir::UserFuncName::user(0, 0), sig.clone());

    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Declare references to math functions.
    let expf_ref = module.declare_func_in_func(expf_id, builder.func);
    let logf_ref = module.declare_func_in_func(logf_id, builder.func);
    let sqrtf_ref = module.declare_func_in_func(sqrtf_id, builder.func);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Extract parameters.
    let block_params: Vec<Value> = builder.block_params(entry_block).to_vec();
    let input_ptrs: Vec<Value> = block_params[..num_inputs].to_vec();
    let out_ptr = block_params[num_inputs];
    let n_param = block_params[num_inputs + 1];

    // Loop: for i in 0..n
    let loop_header = builder.create_block();
    builder.append_block_param(loop_header, types::I64);
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    // i = 0
    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().jump(loop_header, &[zero]);

    // Loop header: check i < n
    builder.switch_to_block(loop_header);
    let i = builder.block_params(loop_header)[0];
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, i, n_param);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);
    // Do NOT seal loop_header yet — loop_body back-edge not created yet.

    // Loop body
    builder.switch_to_block(loop_body);
    builder.seal_block(loop_body); // Only predecessor is loop_header (already defined).

    // Byte offset = i * 4 (sizeof f32)
    let four = builder.ins().iconst(types::I64, 4);
    let byte_offset = builder.ins().imul(i, four);

    let math_refs = MathFuncRefs {
        expf: expf_ref,
        logf: logf_ref,
        sqrtf: sqrtf_ref,
    };

    let result = build_expression(
        graph,
        kernel.root,
        &mut builder,
        &input_ptrs,
        &input_index,
        byte_offset,
        &math_refs,
    )?;

    // Store result to out[i]
    let out_addr = builder.ins().iadd(out_ptr, byte_offset);
    builder.ins().store(MemFlags::new(), result, out_addr, 0);

    // i += 1, jump back to header
    let one = builder.ins().iconst(types::I64, 1);
    let i_next = builder.ins().iadd(i, one);
    builder.ins().jump(loop_header, &[i_next]);

    // Now seal loop_header — both predecessors (entry, loop_body) are complete.
    builder.seal_block(loop_header);

    // Exit
    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit); // Only predecessor is loop_header.
    builder.ins().return_(&[]);

    builder.finalize();

    // --- Compile ---
    let mut ctx = cranelift_codegen::Context::for_function(func);
    module.define_function(func_id, &mut ctx)?;
    module.finalize_definitions()?;

    let fn_ptr = module.get_finalized_function(func_id);

    Ok(CompiledKernel {
        num_inputs,
        _module: module,
        fn_ptr,
    })
}

struct MathFuncRefs {
    expf: cranelift_codegen::ir::FuncRef,
    logf: cranelift_codegen::ir::FuncRef,
    sqrtf: cranelift_codegen::ir::FuncRef,
}

/// Recursively build the Cranelift IR for the expression tree rooted at `id`.
fn build_expression(
    graph: &Graph,
    id: NodeId,
    builder: &mut FunctionBuilder,
    input_ptrs: &[Value],
    input_index: &HashMap<NodeId, usize>,
    byte_offset: Value,
    math: &MathFuncRefs,
) -> Result<Value> {
    let node = graph.node(id);

    match &node.op {
        op if !op.is_elementwise() && !matches!(op, Op::Const(_)) => {
            let idx = input_index
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("Barrier node {:?} not found in input_index", id))?;
            let ptr = input_ptrs[*idx];
            let addr = builder.ins().iadd(ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }
        Op::Load => {
            let idx = input_index
                .get(&id)
                .ok_or_else(|| anyhow::anyhow!("Load node {:?} not found in input_index", id))?;
            let ptr = input_ptrs[*idx];
            let addr = builder.ins().iadd(ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }
        Op::Const(val) => Ok(builder.ins().f32const(*val)),
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            let lhs = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
            )?;
            let rhs = build_expression(
                graph,
                node.inputs[1],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
            )?;
            Ok(match node.op {
                Op::Add => builder.ins().fadd(lhs, rhs),
                Op::Sub => builder.ins().fsub(lhs, rhs),
                Op::Mul => builder.ins().fmul(lhs, rhs),
                Op::Div => builder.ins().fdiv(lhs, rhs),
                _ => unreachable!(),
            })
        }
        Op::Neg => {
            let val = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
            )?;
            Ok(builder.ins().fneg(val))
        }
        Op::Exp => {
            let val = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
            )?;
            let call = builder.ins().call(math.expf, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Ln => {
            let val = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
            )?;
            let call = builder.ins().call(math.logf, &[val]);
            Ok(builder.inst_results(call)[0])
        }
        Op::Sqrt => {
            let val = build_expression(
                graph,
                node.inputs[0],
                builder,
                input_ptrs,
                input_index,
                byte_offset,
                math,
            )?;
            Ok(builder.ins().sqrt(val))
        }
        _ => Err(anyhow::anyhow!(
            "Unsupported op in JIT expression builder for node {:?}: {:?}",
            id,
            node.op
        )),
    }
}

```


**Tool Call: cargo test -q**
Status: Completed

```
warning: method `size_bytes` is never used
 --> src/core/lazy/dtype.rs:9:12
  |
8 | impl DType {
  | ---------- method in this implementation
9 |     pub fn size_bytes(self) -> usize {
  |            ^^^^^^^^^^
  |
  = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: methods `dtype` and `len` are never used
  --> src/core/lazy/dtype.rs:22:12
   |
21 | impl Buffer {
   | ----------- methods in this implementation
22 |     pub fn dtype(&self) -> DType {
   |            ^^^^^
...
28 |     pub fn len(&self) -> usize {
   |            ^^^

warning: field `0` is never read
  --> src/core/lazy/graph.rs:27:13
   |
27 |     Reshape(Vec<usize>),
   |     ------- ^^^^^^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
27 -     Reshape(Vec<usize>),
27 +     Reshape(()),
   |

warning: field `0` is never read
  --> src/core/lazy/graph.rs:30:12
   |
30 |     Expand(Vec<usize>),
   |     ------ ^^^^^^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
30 -     Expand(Vec<usize>),
30 +     Expand(()),
   |

warning: field `0` is never read
  --> src/core/lazy/graph.rs:34:15
   |
34 |     Unsqueeze(usize),
   |     --------- ^^^^^
   |     |
   |     field in this variant
   |
   = note: `Op` has derived impls for the traits `Debug` and `Clone`, but these are intentionally ignored during dead code analysis
help: consider changing the field to be of unit type to suppress this warning while preserving the field numbering, or remove the field
   |
34 -     Unsqueeze(usize),
34 +     Unsqueeze(()),
   |

warning: methods `is_binary` and `is_unary` are never used
  --> src/core/lazy/graph.rs:52:12
   |
44 | impl Op {
   | ------- methods in this implementation
...
52 |     pub fn is_binary(&self) -> bool {
   |            ^^^^^^^^^
...
56 |     pub fn is_unary(&self) -> bool {
   |            ^^^^^^^^

warning: methods `node_mut` and `consumer_count` are never used
   --> src/core/lazy/graph.rs:117:12
    |
102 | impl Graph {
    | ---------- methods in this implementation
...
117 |     pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
    |            ^^^^^^^^
...
340 |     pub fn consumer_count(&self, id: NodeId) -> usize {
    |            ^^^^^^^^^^^^^^

warning: field `sqrtf` is never read
   --> src/core/lazy/jit.rs:283:5
    |
280 | struct MathFuncRefs {
    |        ------------ field in this struct
...
283 |     sqrtf: cranelift_codegen::ir::FuncRef,
    |     ^^^^^


running 36 tests
....................................
test result: ok. 36 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

**Tool Call: Check diagnostics for `venum/src/core/lazy/jit.rs`**
Status: Completed

warning at line 283: field `sqrtf` is never read


**Tool Call: Check diagnostics for `venum/src/core/lazy/tests.rs`**
Status: Completed

File doesn't have errors or warnings!

## Assistant

Done — I pushed Phase 4 hardening forward and validated it.

## What I changed

### 1. Hardened optimized DAG rendering
I updated:

- `Tensor::render_optimized_dag()`
- `Tensor::render_optimized_fused_dag()`

So they now **only run egglog when the graph is optimize-safe**.  
If the graph contains shape or reduce ops, they fall back to normal rendering instead of trying to optimize unsupported structure.

That prevents misleading or unsafe optimized visualizations for mixed graphs.

### 2. Fixed JIT execution for mixed pipelines
A real issue showed up with this kind of pipeline:

- `reduce -> elementwise`

The scheduler correctly emitted the reduce as a barrier and then a fused elementwise kernel after it, but the JIT builder still tried to recursively descend into the `Sum` node as if it were elementwise IR.

I fixed `build_expression()` in `src/core/lazy/jit.rs` so that:

- any non-elementwise, non-const node encountered inside a fused kernel
- is treated as an **already-materialized external input**
- and loaded from the kernel input buffer list

That makes the JIT path consistent with the scheduler’s barrier model.

## New regression coverage

I added tests for the exact mixed cases we wanted to harden:

- shape -> reduce
- reduce -> elementwise
- permute -> reduce with `keepdims`
- chained reduces with `keepdims`
- shape -> reduce -> shape chain

## Validation

I ran:

- `cargo test -q`

Result:

- **36 passed**
- **0 failed**

## Current status of Phase 4

At this point, Phase 4 looks meaningfully more solid:

- reduce scheduling works
- reduce realization works
- mixed reduce/shape/elementwise pipelines work
- JIT no longer breaks when a fused kernel depends on a realized barrier node
- optimized rendering now avoids unsupported graphs safely

## Remaining non-blocking warnings

There are still refactor-residue warnings in:

- `src/core/lazy/dtype.rs`
- `src/core/lazy/graph.rs`
- `src/core/lazy/jit.rs`

Notably:
- some helper methods unused
- some `Op` payload fields not yet consumed everywhere
- `sqrtf` ref field unused because `Sqrt` currently lowers directly via Cranelift intrinsic

These are cleanup-level, not correctness-level.

## Best next move

The natural next step is **Phase 5: graph matmul/conv introduction**.

I’d suggest doing it in this order:

1. add `Op::Matmul` and scheduler barrier support
2. realize matmul through a standalone runtime path first
3. migrate any existing examples/tests to `Tensor` + `Context`
4. only then tackle conv with the same pattern

If you want, I can continue directly into Phase 5 and start with `Matmul` first.

## User

I want to ask,
for both graph nodes and the tensors themselves,
would arena allocators be more optimal,
grouping them with lifetimes, for eg - bumpalo ?

## Assistant

<thinking>
The user is asking a good design question about whether arena allocators (like bumpalo) would be more optimal for both graph nodes and tensors in this lazy tensor system. Let me think through this carefully.

**Current architecture:**
- `Graph` holds a `Vec<Node>` — nodes are stored inline by value
- `Tensor` holds `Arc<Mutex<Graph>>` + `NodeId` + `shape: Vec<usize>`
- `Buffer` holds `Arc<Vec<f32>>` for shared ownership of tensor data
- The shared `Context` owns the `Arc<Mutex<Graph>>`

**What would an arena bring?**

Let me think through the actual memory layout and access patterns:


Nodes are already stored contiguously in a vector, so they have good cache locality. Each `Node` contains several heap allocations though — the `Vec<NodeId>` for inputs, the `Vec<usize>` for shape, and potentially a `Buffer` with its own `Arc<Vec<f32>>`. An arena could consolidate these allocations, but I need to consider whether the benefit outweighs the complexity of lifetime management and the fact that nodes have different lifetimes depending on whether they're still referenced by active tensors. -allocation overhead for small slices (typical inputs list is 0-2 elements, shape is 1-4 elements)
- Improve cache locality since these small slices would be packed together in the bump arena
- Eliminate individual `Vec` capacity/len/ptr (24 bytes each) overhead

Now looking at the `Tensor` struct itself — it's already pretty lean with just an Arc pointer, a NodeId, and a shape Vec. Since tensors are typically short-lived user-facing handles that get stack-allocated or cloned, an arena wouldn't provide much benefit here. The shape Vec is small enough that the overhead isn't a real concern.

For the buffer data...

The actual tensor data stored in `Buffer::F32` is where the real memory lives, but an arena doesn't make sense there either. The buffers can be massive, their lifetimes don't align naturally with graph nodes, and the whole design relies on sharing them across multiple tensors via Arc. Trying to manage that with a bump allocator would create more problems than it solves.

Looking at the practical constraints with bumpalo — the lifetime scoping means everything allocated in the arena needs to outlive it, which conflicts with how the lazy graph system works. Tensors can stick around across multiple `realize()` calls, and graphs grow incrementally as operations get added. Bumpalo really shines when you build everything at once and tear it down together, not in this kind of incremental, long-lived scenario.

That said, if the entire graph lived for the duration of the `Context`, bumpalo could actually work well. The real wins would come from replacing the small Vec allocations — `Node.inputs` and `Node.shape` are typically tiny, so converting them to slices would save the Vec overhead. Same with the operation variants that hold vectors like `Reshape` and `Permute`.

But there are places where it wouldn't help: `Tensor` structs are usually short-lived and stack-allocated, `Buffer` data has different lifetime semantics and is too large to benefit, and `Graph::nodes` is already a contiguous vector doing its job well.

A simpler alternative might be small vector optimization — using something like `SmallVec` or `tinyvec` to keep small collections inline. Most of these vectors are just 2-4 elements anyway, so a `SmallVec<[NodeId; 2]>` for inputs or `SmallVec<[usize; 4]>` for shapes would eliminate heap allocations for the common case without the lifetime complexity.

This is often the better practical choice because it avoids threading lifetime parameters throughout the codebase, still gets the allocation wins for typical cases, and plays nicer with incremental graph building. For node metadata specifically, an arena would help, but the lifetime coupling would be pervasive — every struct would need the parameter — and bumpalo's all-or-nothing freeing doesn't fit a multi-session incremental model anyway. At the current scale, the gains probably don't justify the architectural cost.

What matters more is cache locality during traversal and the fact that the real bottleneck will be JIT compilation, not graph construction. Using `Vec<NodeId>` indices for scheduling is already optimal. So my take is that small inline collections for sub-allocations give most of the benefit with minimal complexity, while a full context-scoped arena is probably overkill for this use case.

The practical question comes down to usage patterns: if graphs are built once and discarded as a whole, a bump arena for node metadata is clean and worth it. But if users need tensors to persist across context resets, the lifetime coupling gets messy. Looking at the actual memory footprint—ops are around 32 bytes, inputs vectors are 24 bytes plus heap overhead for a few elements, shapes are another 24 bytes—the gains from arenas are real but not transformative.

Using SmallVec for both inputs and shapes would be more impactful: most ML operations have 2 or fewer inputs and shapes fit in 4 dimensions, so we'd eliminate heap allocations for the typical case and pack nodes into 1-2 cache lines. That's a cleaner, more practical improvement than arena allocation.
</thinking>

Good question — let me give you a thorough, honest answer because the tradeoffs are non-trivial here.

Let me look at the actual memory layout first to make this concrete.

**Tool Call: Read file `src/core/lazy/graph.rs` (lines 1-105)**
Status: Completed

```venum/src/core/lazy/graph.rs
use std::sync::Arc;

use super::dtype::{Buffer, DType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug)]
pub enum Op {
    // Leaf
    Const(f32),
    Load,

    // Binary elementwise
    Add,
    Sub,
    Mul,
    Div,

    // Unary elementwise
    Exp,
    Ln,
    Sqrt,
    Neg,

    // Shape ops (lazy graph nodes)
    Reshape(Vec<usize>),
    Permute(Vec<usize>),
    Transpose(usize, usize),
    Expand(Vec<usize>),
    Slice(Vec<(usize, usize)>),
    Flip(Vec<usize>),
    Squeeze,
    Unsqueeze(usize),
    Pad(f32, Vec<(usize, usize)>),

    // Reduce ops
    Sum(Vec<usize>, bool),
    Prod(Vec<usize>, bool),
    Max(Vec<usize>, bool),
    Min(Vec<usize>, bool),
}

impl Op {
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Exp | Op::Ln | Op::Sqrt | Op::Neg
        )
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Op::Add | Op::Sub | Op::Mul | Op::Div)
    }

    pub fn is_unary(&self) -> bool {
        matches!(self, Op::Exp | Op::Ln | Op::Sqrt | Op::Neg)
    }

    pub fn is_shape_op(&self) -> bool {
        matches!(
            self,
            Op::Reshape(_)
                | Op::Permute(_)
                | Op::Transpose(_, _)
                | Op::Expand(_)
                | Op::Slice(_)
                | Op::Flip(_)
                | Op::Squeeze
                | Op::Unsqueeze(_)
                | Op::Pad(_, _)
        )
    }

    pub fn is_reduce_op(&self) -> bool {
        matches!(
            self,
            Op::Sum(_, _) | Op::Prod(_, _) | Op::Max(_, _) | Op::Min(_, _)
        )
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub buffer: Option<Buffer>,
}

impl Node {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

```


**Tool Call: Read file `src/core/lazy/dtype.rs`**
Status: Completed

```venum/src/core/lazy/dtype.rs
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Buffer {
    F32(Arc<Vec<f32>>),
}

impl Buffer {
    pub fn dtype(&self) -> DType {
        match self {
            Buffer::F32(_) => DType::F32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Buffer::F32(v) => v.len(),
        }
    }

    pub fn as_f32(&self) -> &[f32] {
        match self {
            Buffer::F32(v) => v,
        }
    }

    pub fn as_f32_ptr(&self) -> *const f32 {
        match self {
            Buffer::F32(v) => v.as_ptr(),
        }
    }
}

```


**Tool Call: Read file `Cargo.toml`**
Status: Completed

```venum/Cargo.toml
[package]
name = "venum"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1.0.89" # result/error
num-traits = "0.2.19" # dtype traits
prettytable = "0.10.0" # printing
thiserror = "1.0.64" # error definition
cranelift = "0.116"
cranelift-jit = "0.116"
cranelift-module = "0.116"
cranelift-native = "0.116"
cranelift-codegen = "0.116"
egglog = "1.0.0"

[dev-dependencies]
image = "0.24.6" # examples/kernels.rs 

```


## Assistant

<thinking>
Now I have a clear picture of the exact memory layout. Let me think through the actual bytes and give a comprehensive answer.

Current per-node memory cost (on 64-bit):
- `Op`: let me enumerate the discriminant + payload:
  - Simple variants (Const, Load, Add, Sub, Mul, Div, Exp, Ln, Sqrt, Neg, Squeeze): discriminant (8 bytes wide) + small payload (f32 for Const = 4 bytes)
  - Vec variants: discriminant + Vec<T> = 8 + 24 = 32 bytes
  - The largest variant determines the enum size. `Op::Pad(f32, Vec<(usize, usize)>)` = 8 + 4 + 24 = let's say 40 bytes? Actually enum size is padded to alignment, so:
  - The max payload is `Pad(f32, Vec<(usize, usize)>)` where the Vec takes 24 bytes and the f32 takes 4 bytes padded to 8, giving 32 bytes total payload plus the 8-byte discriminant, so Op ends up around 40-48 bytes depending on how Rust packs the discriminant and applies niche-filling optimizations.
  
  Looking at `Sum(Vec<usize>, bool)` similarly: the Vec is 24 bytes, bool is 1 byte with 7 bytes of padding to reach 8, so again 32 bytes plus discriminant for roughly 40 bytes total. The simpler variants are wasting that full 32-byte payload space.
  
  For `inputs: Vec<NodeId>`, the Vec header itself is 24 bytes, and for typical cases like binary ops with 2 inputs or unary ops with 1 input, I'm looking at 24 bytes for the header plus the actual allocation for the NodeId data—16 bytes for a binary op's two 8-byte pointers.
  
  The `shape: Vec<usize>` field follows the same pattern: 24-byte header plus heap allocation for the shape dimensions, typically 16 bytes for 2D tensors. Both vectors incur allocator overhead on top of their actual data.
  
  For `dtype: DType`, that's just 1 byte with padding, and `buffer: Option<Buffer>` is either 8 bytes for None or 8 bytes for Some with the Arc pointer, plus whatever overhead the Arc itself carries.
  
  So for a typical binary operation node like `Add`, I'm looking at roughly 40 bytes for the enum variant, 24 bytes for the inputs Vec header plus around 32 bytes of heap overhead, another 24 bytes for the shape Vec header with similar heap costs, plus the dtype and buffer fields.
  
  Load nodes are different—the inputs Vec is empty so no heap allocation there, but the buffer field with its Arc<Vec<f32>> becomes the dominant cost factor.
  
  The outer `Vec<Node>` in the Graph holds all these nodes together.
  
  With bumpalo, I could replace those small Vecs with slices backed by the arena—`inputs: &'bump [NodeId]` and `shape: &'bump [usize]`—which would eliminate per-allocation overhead and improve cache locality during graph traversals. The tricky part is threading the lifetime through the entire system.

A cleaner approach might be to make the bump arena owned by `Context` itself, so `Graph` just borrows from it. This works well if tensors don't outlive their context anyway, which seems like the intended design—you create a context, build tensors, realize them, and you're done.

Alternatively, I could use `SmallVec` for both node inputs and shape dimensions, which would eliminate the two most common heap allocations per node without any lifetime complexity. `SmallVec<[NodeId; 2]>` handles unary and binary ops (which cover nearly everything), and `SmallVec<[usize; 4]>` covers most tensor shapes.

For the immediate term, `SmallVec` is the pragmatic choice. Full bumpalo for node metadata becomes worth considering once the API stabilizes around a clean "Context scope = arena scope" model, but I should never arena the actual data buffers since that breaks ownership semantics.

The main friction with bumpalo is that `Bump` doesn't easily support `Send`/`Sync` — it uses `UnsafeCell` internally, and since `Context` is currently `Arc<Mutex<Graph>>` for thread safety, mixing in a bump arena requires careful handling. I'd need to either explicitly implement `Send` (which is actually safe in practice since the bump is single-threaded per context) or find an alternative approach.

Other arena crates like `typed-arena` or `id-arena` have different tradeoffs, but they also struggle with `Send`. The real issue is that if the graph contains bump-allocated slices behind a `Mutex`, the allocator itself becomes a bottleneck — you'd hold the lock during allocation, which defeats some of the performance benefit.

The cleanest path forward is probably index-based storage backed by `Vec` collections, which sidesteps the threading complexity entirely while still giving me the memory efficiency I'm after. Vec<Node> already functions as an arena for node headers, so the real bottleneck is the sub-allocations within each node itself—SmallVec solves that for most cases, while a full slab approach eliminates it entirely, though bumpalo offers true sub-linear allocation at the cost of lifetime complexity.
</thinking>

Good — now I have the full picture. Let me reason through the actual memory layout before answering.

## What each `Node` currently looks like in memory

```venum/src/core/lazy/graph.rs#L87-97
pub struct Node {
    pub op: Op,              // ~40 bytes (enum; Vec-payload variants dominate)
    pub inputs: Vec<NodeId>, // 24-byte header + heap alloc (usually 1-2 elements)
    pub shape: Vec<usize>,   // 24-byte header + heap alloc (usually 2-4 elements)
    pub dtype: DType,        // 1 byte + padding
    pub buffer: Option<Buffer>, // ptr-sized or None
}
```

For a typical binary node like `Add`:
- The `Vec<NodeId>` header is 24 bytes, and then a **separate heap allocation** for 2 × 8 = 16 bytes of data (plus malloc overhead, ~16 extra bytes)
- The `Vec<usize>` shape header is 24 bytes, then **another separate heap allocation** for maybe 16 bytes

So every node creates **2 small heap allocations** for things that are fixed at construction and never mutate. That is the real cost.

`Graph::nodes: Vec<Node>` is already a good linear slab for the node headers themselves — that part is fine.

---

## Would bumpalo help?

**Yes, but only for those sub-allocations**, and it comes with real tradeoffs.

### ✅ Where bumpalo would genuinely help

| Sub-allocation | Typical size | Current cost |
|---|---|---|
| `inputs: Vec<NodeId>` | 1–2 elements | 24B header + heap alloc |
| `shape: Vec<usize>` | 1–4 elements | 24B header + heap alloc |
| `Op::Reshape(Vec<usize>)` and similar payload vecs | 1–4 elements | heap alloc |

If you replace those with `&'bump [NodeId]` and `&'bump [usize]`:
- No per-node malloc overhead
- All node metadata for a whole context packed in one contiguous bump region
- Topo sort / scheduling traversals are then reading from one or two hot cache lines per node

### ⚠️ Where it would not help

`Buffer::F32(Arc<Vec<f32>>)` — the actual tensor data. This has completely different ownership semantics (shared, potentially large, outlives individual graph nodes) and should stay as `Arc<Vec<f32>>`.

The `Tensor` struct handles themselves are typically stack-allocated in user code — no benefit there either.

---

## The hard part: bumpalo + threading

Your current architecture uses `Arc<Mutex<Graph>>` for thread safety. This is where bumpalo creates friction:

`bumpalo::Bump` is `!Send`. You can't put it inside `Arc<Mutex<...>>` without `unsafe`. Wrapping it with `unsafe impl Send` is actually defensible (the bump is only ever accessed while the mutex is held), but it requires explicit justification and makes reviewers nervous.

The lifetime also has to thread through everything:

```venum/src/core/lazy/context.rs#L1-26
// Context<'bump> → Arc<Mutex<Graph<'bump>>> → Node<'bump> → &'bump [NodeId]
//                                                          → &'bump [usize]
// Tensor<'cx> → carries 'cx for graph refs
```

That `'bump` / `'cx` parameter propagates into every public type: `Context<'cx>`, `Tensor<'cx>`, all function signatures. It's manageable but it's not free.

---

## The pragmatic alternative: `SmallVec`

`SmallVec<[NodeId; 2]>` for inputs and `SmallVec<[usize; 4]>` for shape captures **roughly 90% of the benefit** with **zero lifetime complexity**:

- For unary/binary ops (≤2 inputs): zero heap allocation, data stored inline
- For ≤4-dim tensors (covers nearly all ML workloads): zero heap allocation, data stored inline
- Completely `Send` + `Sync`, no lifetime threading, compiles today

The only cost is a slightly larger struct (inline storage instead of a pointer + allocation).

---

## There is a third path: flat slab storage

If you want true arena locality without bumpalo's threading headache, you can roll a simple slab inside `Graph` itself:

```venum/src/core/lazy/graph.rs#L100-110
pub struct Graph {
    pub nodes: Vec<Node>,
    // Flat backing stores — nodes reference into these by (start, len)
    input_slab: Vec<NodeId>,
    shape_slab: Vec<usize>,
}
```

Each node stores `inputs_start: u32, inputs_len: u8, shape_start: u32, shape_len: u8` instead of `Vec`. Traversal then reads contiguous `&input_slab[start..start+len]`. This is trivially `Send`, no lifetime param, and matches arena locality. It's what LLVM's `MachineInstr` pool and many IR designs do.

---

## My honest recommendation

In priority order:

**1. Do `SmallVec` now** — switch `inputs` and `shape` to `SmallVec`. It eliminates the two most common per-node heap allocations immediately, no API changes needed, tests stay green. Add `smallvec = "1"` to `Cargo.toml`.

**2. Consider the flat slab design** when the `Graph` API stabilizes — it gives full arena-style locality for node metadata and is `Send`-safe without unsafe.

**3. Hold on bumpalo** until you decide whether user-facing lifetimes (`Context<'cx>`, `Tensor<'cx>`) are acceptable in the public API. If you're going ML-framework style where a `Context` has a clear build-and-realize session lifecycle, it fits naturally. If you want tensors to be freely `'static`, it gets messy.

**Don't arena the data buffers.** `Arc<Vec<f32>>` is correct there.

---

Want me to go ahead and wire in `SmallVec` for the node fields first? That's an easy, clean win right now. Or if you want to commit to the full bumpalo/slab route instead we can sketch the architecture before touching code.

