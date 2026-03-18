use anyhow::Result;

pub(super) fn define_datatype(egraph: &mut egglog::EGraph) -> Result<()> {
    egraph.parse_and_run_program(
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
    )?;
    Ok(())
}

pub(super) fn define_rewrites(egraph: &mut egglog::EGraph) -> Result<()> {
    egraph.parse_and_run_program(
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
    )?;
    Ok(())
}
