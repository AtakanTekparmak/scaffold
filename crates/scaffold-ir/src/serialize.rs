//! AST-to-IR lowering for Scaffold v2

use scaffold_syntax::ast::*;

use crate::ir::*;

/// Error during lowering
#[derive(Debug, Clone)]
pub struct LowerError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lower error at {:?}: {}", self.span, self.message)
    }
}

/// Lower a program AST to IR
pub fn lower(program: &Program) -> Result<ScaffoldIR, Vec<LowerError>> {
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(program)?;
    Ok(lowerer.ir)
}

/// Serialize IR to JSON string
pub fn to_json(ir: &ScaffoldIR) -> Result<String, String> {
    serde_json::to_string_pretty(ir).map_err(|e| e.to_string())
}

/// Serialize IR to compact JSON
pub fn to_json_compact(ir: &ScaffoldIR) -> Result<String, String> {
    serde_json::to_string(ir).map_err(|e| e.to_string())
}

/// Deserialize IR from JSON string
pub fn from_json(json: &str) -> Result<ScaffoldIR, String> {
    serde_json::from_str(json).map_err(|e| e.to_string())
}

pub struct Lowerer {
    ir: ScaffoldIR,
    errors: Vec<LowerError>,
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            ir: ScaffoldIR::default(),
            errors: Vec::new(),
        }
    }

    pub fn lower_program(&mut self, program: &Program) -> Result<(), Vec<LowerError>> {
        for decl in &program.declarations {
            match decl {
                Declaration::Type(td) => self.lower_type_decl(td),
                Declaration::Node(nd) => self.lower_node_decl(nd),
                Declaration::Graph(gd) => self.lower_graph_decl(gd),
                Declaration::Objective(od) => self.lower_objective_decl(od),
            }
        }
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    fn lower_type_decl(&mut self, td: &TypeDecl) {
        let ty = lower_type_expr(&td.ty);
        self.ir.types.push(TypeDefIR {
            name: td.name.name.clone(),
            ty,
        });
    }

    fn lower_node_decl(&mut self, nd: &NodeDecl) {
        let kind = match nd.kind.node {
            NodeKind::Prompt => NodeKindIR::Prompt,
            NodeKind::Tool => NodeKindIR::Tool,
            NodeKind::Agent => NodeKindIR::Agent,
            NodeKind::Verify => NodeKindIR::Verify,
        };

        let config = NodeConfigIR {
            template: nd.config.template.as_ref().map(lower_string_or_file),
            system: nd.config.system.as_ref().map(lower_string_or_file),
            model: nd.config.model.clone(),
            temperature: nd.config.temperature,
            max_tokens: nd.config.max_tokens,
            tools: nd.config.tools.iter().map(|i| i.name.clone()).collect(),
            max_turns: nd.config.max_turns,
            timeout: nd.config.timeout,
            on_error: nd.config.on_error.as_ref().map(|e| match e {
                ErrorStrategy::Abort => ErrorStrategyIR::Abort,
                ErrorStrategy::Retry(n) => ErrorStrategyIR::Retry { max: *n },
            }),
            shell: nd.config.shell.clone(),
            json: nd.config.json.as_ref().map(|fields| {
                fields
                    .iter()
                    .map(|f| JsonFieldIR {
                        key: f.key.clone(),
                        value: lower_expr(&f.value),
                    })
                    .collect()
            }),
        };

        self.ir.nodes.push(NodeIR {
            name: nd.name.name.clone(),
            kind,
            input: lower_type_expr(&nd.input),
            output: lower_type_expr(&nd.output),
            config,
        });
    }

    fn lower_graph_decl(&mut self, gd: &GraphDecl) {
        let body = gd.body.iter().map(lower_graph_stmt).collect();
        self.ir.graphs.push(GraphIR {
            name: gd.name.name.clone(),
            input: lower_type_expr(&gd.input),
            output: lower_type_expr(&gd.output),
            body,
        });
    }

    fn lower_objective_decl(&mut self, od: &ObjectiveDecl) {
        let dataset = match &od.dataset {
            DatasetSpec::File(path) => DatasetSpecIR::File {
                path: path.clone(),
            },
            DatasetSpec::Inline { cases } => DatasetSpecIR::Inline {
                cases: cases
                    .iter()
                    .map(|c| InlineCaseIR {
                        input: lower_expr(&c.input),
                        expected: lower_expr(&c.expected),
                        id: c.id.clone(),
                    })
                    .collect(),
            },
        };

        let checkers = od
            .checkers
            .iter()
            .map(|c| CheckerIR {
                name: c.name.name.clone(),
                expr: lower_expr(&c.expr),
            })
            .collect();

        let judges = od
            .judges
            .iter()
            .map(|j| JudgeIR {
                name: j.name.name.clone(),
                model: j.model.clone(),
                template: j.template.as_ref().map(lower_string_or_file),
                rubric: j.rubric.as_ref().map(lower_string_or_file),
            })
            .collect();

        let metrics = od
            .metrics
            .iter()
            .map(|m| MetricIR {
                name: m.name.name.clone(),
                checker: m.checker.name.clone(),
            })
            .collect();

        let split = od.split.as_ref().map(|s| SplitIR {
            train: s.train,
            val: s.val,
            test: s.test,
        });

        let select = od.select.as_ref().map(|s| SelectIR {
            primary: s.primary.name.clone(),
            tie_breakers: s.tie_breakers.iter().map(|i| i.name.clone()).collect(),
        });

        let tunables = od
            .tunables
            .iter()
            .map(|t| TunableIR {
                path: t.path.iter().map(|i| i.name.clone()).collect(),
                domain: t.domain.iter().map(lower_expr).collect(),
            })
            .collect();

        let topology = od.topology.as_ref().map(|t| TopologyIR {
            mutations: t.mutations.clone(),
            max_nodes: t.max_nodes,
            max_depth: t.max_depth,
            preserve: t.preserve.clone(),
        });

        self.ir.objectives.push(ObjectiveIR {
            name: od.name.name.clone(),
            graph: od.graph.name.clone(),
            dataset,
            checkers,
            judges,
            metrics,
            score: lower_expr(&od.score),
            repeats: od.repeats,
            split,
            select,
            tunables,
            topology,
        });
    }
}

fn lower_type_expr(texpr: &Spanned<TypeExpr>) -> TypeIR {
    match &texpr.node {
        TypeExpr::Primitive(p) => match p {
            PrimitiveType::Bool => TypeIR::Bool,
            PrimitiveType::Int => TypeIR::Int,
            PrimitiveType::Float => TypeIR::Float,
            PrimitiveType::String => TypeIR::String,
            PrimitiveType::Bytes => TypeIR::Bytes,
            PrimitiveType::Any => TypeIR::Any,
        },
        TypeExpr::Named(name) => TypeIR::Named { name: name.clone() },
        TypeExpr::List(inner) => TypeIR::List {
            element: Box::new(lower_type_expr(inner)),
        },
        TypeExpr::Map(key, val) => TypeIR::Map {
            key: Box::new(lower_type_expr(key)),
            value: Box::new(lower_type_expr(val)),
        },
        TypeExpr::Option(inner) => TypeIR::Option {
            inner: Box::new(lower_type_expr(inner)),
        },
        TypeExpr::Struct(fields) => TypeIR::Struct {
            fields: fields
                .iter()
                .map(|f| FieldIR {
                    name: f.name.name.clone(),
                    ty: lower_type_expr(&f.ty),
                })
                .collect(),
        },
    }
}

fn lower_string_or_file(sof: &StringOrFile) -> StringOrFileIR {
    match sof {
        StringOrFile::Literal(s) => StringOrFileIR::Literal { value: s.clone() },
        StringOrFile::File(p) => StringOrFileIR::File { path: p.clone() },
    }
}

fn lower_graph_stmt(stmt: &GraphStmt) -> GraphStmtIR {
    match stmt {
        GraphStmt::Step(s) => GraphStmtIR::Step(StepIR {
            name: s.name.name.clone(),
            node: s.node.name.clone(),
            args: s.args.iter().map(lower_step_arg).collect(),
        }),
        GraphStmt::Loop(l) => GraphStmtIR::Loop(LoopIR {
            max: lower_expr(&l.max),
            while_cond: lower_expr(&l.while_cond),
            body: l.body.iter().map(lower_graph_stmt).collect(),
        }),
        GraphStmt::If(i) => GraphStmtIR::If(IfIR {
            cond: lower_expr(&i.cond),
            then_body: i.then_body.iter().map(lower_graph_stmt).collect(),
            else_body: i.else_body.iter().map(lower_graph_stmt).collect(),
        }),
        GraphStmt::Choose(c) => GraphStmtIR::Choose(ChooseIR {
            alternatives: c.alternatives.iter().map(|i| i.name.clone()).collect(),
        }),
        GraphStmt::Parallel(p) => GraphStmtIR::Parallel(ParallelIR {
            var: p.var.name.clone(),
            collection: lower_expr(&p.collection),
            reduce: p.reduce.as_ref().map(|i| i.name.clone()),
            body: p.body.iter().map(lower_graph_stmt).collect(),
        }),
        GraphStmt::Emit(e) => GraphStmtIR::Emit(match e {
            EmitStmt::Direct { value, .. } => EmitIR::Direct {
                value: lower_expr(value),
            },
            EmitStmt::Record { fields, .. } => EmitIR::Record {
                fields: fields
                    .iter()
                    .map(|f| EmitFieldIR {
                        name: f.name.name.clone(),
                        value: lower_expr(&f.value),
                    })
                    .collect(),
            },
        }),
        GraphStmt::Carry(c) => GraphStmtIR::Carry(CarryIR {
            name: c.name.name.clone(),
            value: lower_expr(&c.value),
        }),
    }
}

fn lower_step_arg(arg: &StepArg) -> StepArgIR {
    match arg {
        StepArg::Positional(expr) => StepArgIR::Positional {
            value: lower_expr(expr),
        },
        StepArg::Named { name, value } => StepArgIR::Named {
            name: name.name.clone(),
            value: lower_expr(value),
        },
    }
}

fn lower_expr(expr: &Spanned<Expr>) -> ExprIR {
    match &expr.node {
        Expr::Literal(lit) => match lit {
            Literal::Int(v) => ExprIR::LitInt { value: *v },
            Literal::Float(v) => ExprIR::LitFloat { value: *v },
            Literal::String(s) => ExprIR::LitString { value: s.clone() },
            Literal::Bool(b) => ExprIR::LitBool { value: *b },
            Literal::Null => ExprIR::LitNull,
        },
        Expr::Ident(name) => ExprIR::Ident { name: name.clone() },
        Expr::FieldAccess(base, field) => ExprIR::FieldAccess {
            base: Box::new(lower_expr(base)),
            field: field.name.clone(),
        },
        Expr::Index(base, index) => ExprIR::Index {
            base: Box::new(lower_expr(base)),
            index: Box::new(lower_expr(index)),
        },
        Expr::UnaryNot(operand) => ExprIR::UnaryNot {
            operand: Box::new(lower_expr(operand)),
        },
        Expr::UnaryNeg(operand) => ExprIR::UnaryNeg {
            operand: Box::new(lower_expr(operand)),
        },
        Expr::Binary(left, op, right) => ExprIR::Binary {
            left: Box::new(lower_expr(left)),
            op: op.to_string(),
            right: Box::new(lower_expr(right)),
        },
        Expr::Call(name, args) => ExprIR::Call {
            name: name.clone(),
            args: args.iter().map(lower_expr).collect(),
        },
        Expr::List(elements) => ExprIR::List {
            elements: elements.iter().map(lower_expr).collect(),
        },
        Expr::Record(fields) => ExprIR::Record {
            fields: fields
                .iter()
                .map(|f| ExprFieldIR {
                    key: f.key.name.clone(),
                    value: lower_expr(&f.value),
                })
                .collect(),
        },
        Expr::Paren(inner) => lower_expr(inner),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaffold_syntax::parser::parse;

    #[test]
    fn lower_and_serialize() {
        let src = r#"
            node solver: prompt {
                in: string
                out: string
                model: "gpt-4o"
                template: "Solve: {{ input }}"
            }

            graph solve {
                in: string
                out: string
                step s = solver(input)
                emit s
            }
        "#;
        let program = parse(src).unwrap();
        let ir = lower(&program).unwrap();
        assert_eq!(ir.nodes.len(), 1);
        assert_eq!(ir.graphs.len(), 1);
        assert_eq!(ir.nodes[0].name, "solver");
        assert_eq!(ir.graphs[0].name, "solve");

        // Test JSON round-trip
        let json = to_json(&ir).unwrap();
        let ir2 = from_json(&json).unwrap();
        assert_eq!(ir2.nodes.len(), 1);
        assert_eq!(ir2.graphs.len(), 1);
    }
}
