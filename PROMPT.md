Phases 1, 2, 3 and 7 have been implemented from @PLAN.md. Also view @specs/04/SPEC.md.
Please go through the repo once. I am not satisfied with the schedule optimization.

Here, as you can see in this example, @examples/multi_subgraph.rs.
All 4 tensors have the same initial shape and the same final reshape.

Thus as they are contiguous, elementwise operations can be done, then a final reshape.
However, this is not happening. I need you to fix this, and every other shape fusion case.

---

venum on  solid [!?] is 󰏗 v0.1.0 via 󱘗 v1.96.0-nightly via  impure (nix-shell-env)
❯ cargo run --example multi_subgraph
   Compiling venum v0.1.0 (/home/sword/Desktop/code/venum)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.38s
     Running `target/debug/examples/multi_subgraph`
multi-subgraph graph nodes: 14

Mermaid graph:
graph TD
  n0["Load#0\nbuffer=0"]
  n1["Reshape#1"]
  n2["Load#2\nbuffer=1"]
  n3["Reshape#3"]
  n4["Load#4\nbuffer=2"]
  n5["Reshape#5"]
  n6["Load#6\nbuffer=3"]
  n7["Reshape#7"]
  n8["Mul#8"]
  n9["Add#9"]
  n10["Mul#10"]
  n11["Neg#11"]
  n12["Add#12"]
  n13["Add#13"]
  n0 --> n1
  n2 --> n3
  n4 --> n5
  n6 --> n7
  n1 --> n8
  n3 --> n8
  n8 --> n9
  n1 --> n9
  n5 --> n10
  n7 --> n10
  n5 --> n11
  n10 --> n12
  n11 --> n12
  n9 --> n13
  n12 --> n13


Optimized Mermaid graph:
graph TD
  n0["Load#0\nbuffer=0"]
  n1["Reshape#1"]
  n2["Load#2\nbuffer=1"]
  n3["Reshape#3"]
  n4["Mul#4"]
  n5["Add#5"]
  n6["Load#6\nbuffer=2"]
  n7["Reshape#7"]
  n8["Load#8\nbuffer=3"]
  n9["Reshape#9"]
  n10["Mul#10"]
  n11["Neg#11"]
  n12["Add#12"]
  n13["Add#13"]
  n0 --> n1
  n2 --> n3
  n1 --> n4
  n3 --> n4
  n4 --> n5
  n1 --> n5
  n6 --> n7
  n8 --> n9
  n7 --> n10
  n9 --> n10
  n7 --> n11
  n10 --> n12
  n11 --> n12
  n5 --> n13
  n12 --> n13

result: [1.0, 3.0, 5.0, 7.0]

