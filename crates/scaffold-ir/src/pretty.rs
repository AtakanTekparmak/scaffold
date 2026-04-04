//! Pretty-printer: IR → .scaffold v2 source

use crate::ir::*;

/// Pretty-print an IR to scaffold source
pub fn pretty_print(ir: &ScaffoldIR) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_ir(ir);
    pp.output
}

/// Pretty-print a single graph to scaffold source
pub fn pretty_print_graph(graph: &GraphIR) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_graph(graph);
    pp.output
}

/// Pretty-print a single node definition to scaffold source
pub fn pretty_print_node(node: &NodeIR) -> String {
    let mut pp = PrettyPrinter::new();
    pp.print_node(node);
    pp.output
}

struct PrettyPrinter {
    output: String,
    indent: usize,
}

impl PrettyPrinter {
    fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
        }
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(s);
        self.output.push('\n');
    }

    fn print_ir(&mut self, ir: &ScaffoldIR) {
        for td in &ir.types {
            self.print_type_def(td);
            self.output.push('\n');
        }
        for node in &ir.nodes {
            self.print_node(node);
            self.output.push('\n');
        }
        for graph in &ir.graphs {
            self.print_graph(graph);
            self.output.push('\n');
        }
        for obj in &ir.objectives {
            self.print_objective(obj);
            self.output.push('\n');
        }
    }

    fn print_type_def(&mut self, td: &TypeDefIR) {
        self.line(&format!("type {} = {}", td.name, format_type(&td.ty)));
    }

    fn print_node(&mut self, node: &NodeIR) {
        let kind = match node.kind {
            NodeKindIR::Prompt => "prompt",
            NodeKindIR::Tool => "tool",
            NodeKindIR::Agent => "agent",
            NodeKindIR::Verify => "verify",
        };
        self.line(&format!("node {}: {} {{", node.name, kind));
        self.indent += 1;
        self.line(&format!("in: {}", format_type(&node.input)));
        self.line(&format!("out: {}", format_type(&node.output)));

        let c = &node.config;
        if let Some(ref t) = c.template {
            self.line(&format!("template: {}", format_string_or_file(t)));
        }
        if let Some(ref s) = c.system {
            self.line(&format!("system: {}", format_string_or_file(s)));
        }
        if let Some(ref m) = c.model {
            self.line(&format!("model: \"{}\"", m));
        }
        if let Some(temp) = c.temperature {
            self.line(&format!("temperature: {}", temp));
        }
        if let Some(mt) = c.max_tokens {
            self.line(&format!("max_tokens: {}", mt));
        }
        if !c.tools.is_empty() {
            self.line(&format!("tools: [{}]", c.tools.join(", ")));
        }
        if let Some(mt) = c.max_turns {
            self.line(&format!("max_turns: {}", mt));
        }
        if let Some(t) = c.timeout {
            self.line(&format!("timeout: {}", t));
        }
        if let Some(ref oe) = c.on_error {
            match oe {
                ErrorStrategyIR::Abort => self.line("on_error: abort"),
                ErrorStrategyIR::Retry { max } => self.line(&format!("on_error: retry({})", max)),
            }
        }
        if let Some(ref sh) = c.shell {
            self.line(&format!("shell: \"{}\"", sh.replace('"', "\\\"")));
        }
        if let Some(ref json_fields) = c.json {
            let fields: Vec<String> = json_fields
                .iter()
                .map(|f| format!("\"{}\": {}", f.key, format_expr(&f.value)))
                .collect();
            self.line(&format!("json: {{ {} }}", fields.join(", ")));
        }

        self.indent -= 1;
        self.line("}");
    }

    fn print_graph(&mut self, graph: &GraphIR) {
        self.line(&format!("graph {} {{", graph.name));
        self.indent += 1;
        self.line(&format!("in: {}", format_type(&graph.input)));
        self.line(&format!("out: {}", format_type(&graph.output)));
        for stmt in &graph.body {
            self.print_graph_stmt(stmt);
        }
        self.indent -= 1;
        self.line("}");
    }

    fn print_graph_stmt(&mut self, stmt: &GraphStmtIR) {
        match stmt {
            GraphStmtIR::Step(s) => {
                let args: Vec<String> = s.args.iter().map(format_step_arg).collect();
                self.line(&format!(
                    "step {} = {}({})",
                    s.name,
                    s.node,
                    args.join(", ")
                ));
            }
            GraphStmtIR::Loop(l) => {
                self.line(&format!(
                    "loop (max: {}, while: {}) {{",
                    format_expr(&l.max),
                    format_expr(&l.while_cond)
                ));
                self.indent += 1;
                for s in &l.body {
                    self.print_graph_stmt(s);
                }
                self.indent -= 1;
                self.line("}");
            }
            GraphStmtIR::If(i) => {
                self.line(&format!("if {} {{", format_expr(&i.cond)));
                self.indent += 1;
                for s in &i.then_body {
                    self.print_graph_stmt(s);
                }
                self.indent -= 1;
                if !i.else_body.is_empty() {
                    self.line("} else {");
                    self.indent += 1;
                    for s in &i.else_body {
                        self.print_graph_stmt(s);
                    }
                    self.indent -= 1;
                }
                self.line("}");
            }
            GraphStmtIR::Choose(c) => {
                self.line(&format!("choose [{}]", c.alternatives.join(", ")));
            }
            GraphStmtIR::Parallel(p) => {
                let reduce = p
                    .reduce
                    .as_ref()
                    .map(|r| format!(", reduce: {}", r))
                    .unwrap_or_default();
                self.line(&format!(
                    "parallel ({} in {}{}) {{",
                    p.var,
                    format_expr(&p.collection),
                    reduce
                ));
                self.indent += 1;
                for s in &p.body {
                    self.print_graph_stmt(s);
                }
                self.indent -= 1;
                self.line("}");
            }
            GraphStmtIR::Emit(e) => match e {
                EmitIR::Direct { value } => {
                    self.line(&format!("emit {}", format_expr(value)));
                }
                EmitIR::Record { fields } => {
                    let fs: Vec<String> = fields
                        .iter()
                        .map(|f| format!("{}: {}", f.name, format_expr(&f.value)))
                        .collect();
                    self.line(&format!("emit {{ {} }}", fs.join(", ")));
                }
            },
            GraphStmtIR::Carry(c) => {
                self.line(&format!("carry {} = {}", c.name, format_expr(&c.value)));
            }
        }
    }

    fn print_objective(&mut self, obj: &ObjectiveIR) {
        self.line(&format!("objective {} {{", obj.name));
        self.indent += 1;

        self.line(&format!("graph: {}", obj.graph));

        self.print_dataset(&obj.dataset);

        for c in &obj.checkers {
            self.line(&format!("checker {} {{ {} }}", c.name, format_expr(&c.expr)));
        }

        for m in &obj.metrics {
            self.line(&format!("metric {} {{ checker: {} }}", m.name, m.checker));
        }

        self.line(&format!("score: {}", format_expr(&obj.score)));

        if let Some(r) = obj.repeats {
            self.line(&format!("repeats: {}", r));
        }

        if let Some(ref s) = obj.split {
            self.line(&format!(
                "split {{ train: {}, val: {}, test: {} }}",
                s.train, s.val, s.test
            ));
        }

        if let Some(ref s) = obj.select {
            if s.tie_breakers.is_empty() {
                self.line(&format!("select {{ primary: {} }}", s.primary));
            } else {
                self.line(&format!(
                    "select {{ primary: {}, tie_breakers: [{}] }}",
                    s.primary,
                    s.tie_breakers.join(", ")
                ));
            }
        }

        if !obj.tunables.is_empty() {
            self.line("tune {");
            self.indent += 1;
            for t in &obj.tunables {
                let domain: Vec<String> = t.domain.iter().map(format_expr).collect();
                self.line(&format!(
                    "{} in [{}]",
                    t.path.join("."),
                    domain.join(", ")
                ));
            }
            self.indent -= 1;
            self.line("}");
        }

        self.print_topology(&obj.topology);

        for sub in &obj.subs {
            self.print_sub_objective(sub);
        }

        self.indent -= 1;
        self.line("}");
    }

    fn print_sub_objective(&mut self, sub: &SubObjectiveIR) {
        self.line(&format!("sub {} {{", sub.name));
        self.indent += 1;

        self.line(&format!("graph: {}", sub.graph));

        self.print_dataset(&sub.dataset);

        for c in &sub.checkers {
            self.line(&format!("checker {} {{ {} }}", c.name, format_expr(&c.expr)));
        }

        for m in &sub.metrics {
            self.line(&format!("metric {} {{ checker: {} }}", m.name, m.checker));
        }

        self.line(&format!("score: {}", format_expr(&sub.score)));

        if let Some(r) = sub.repeats {
            self.line(&format!("repeats: {}", r));
        }

        if let Some(ref s) = sub.split {
            self.line(&format!(
                "split {{ train: {}, val: {}, test: {} }}",
                s.train, s.val, s.test
            ));
        }

        if let Some(ref s) = sub.select {
            if s.tie_breakers.is_empty() {
                self.line(&format!("select {{ primary: {} }}", s.primary));
            } else {
                self.line(&format!(
                    "select {{ primary: {}, tie_breakers: [{}] }}",
                    s.primary,
                    s.tie_breakers.join(", ")
                ));
            }
        }

        if !sub.tunables.is_empty() {
            self.line("tune {");
            self.indent += 1;
            for t in &sub.tunables {
                let domain: Vec<String> = t.domain.iter().map(format_expr).collect();
                self.line(&format!(
                    "{} in [{}]",
                    t.path.join("."),
                    domain.join(", ")
                ));
            }
            self.indent -= 1;
            self.line("}");
        }

        self.print_topology(&sub.topology);

        self.indent -= 1;
        self.line("}");
    }

    fn print_topology(&mut self, topology: &Option<TopologyIR>) {
        if let Some(ref t) = topology {
            self.line("topology {");
            self.indent += 1;
            if !t.mutations.is_empty() {
                self.line(&format!("mutations: [{}]", t.mutations.join(", ")));
            }
            if let Some(mn) = t.max_nodes {
                self.line(&format!("max_nodes: {}", mn));
            }
            if let Some(md) = t.max_depth {
                self.line(&format!("max_depth: {}", md));
            }
            if !t.preserve.is_empty() {
                self.line(&format!("preserve: [{}]", t.preserve.join(", ")));
            }
            if let Some(ts) = t.target_score {
                self.line(&format!("target_score: {}", ts));
            }
            self.indent -= 1;
            self.line("}");
        }
    }

    fn print_dataset(&mut self, dataset: &DatasetSpecIR) {
        match dataset {
            DatasetSpecIR::File { path } => {
                self.line(&format!("dataset: file(\"{}\")", path));
            }
            DatasetSpecIR::Inline { cases } => {
                self.line("dataset: cases [");
                self.indent += 1;
                for case in cases {
                    let id_str = case
                        .id
                        .as_ref()
                        .map(|id| format!(", id: \"{}\"", id))
                        .unwrap_or_default();
                    self.line(&format!(
                        "{{ input: {}, expected: {}{} }}",
                        format_expr(&case.input),
                        format_expr(&case.expected),
                        id_str
                    ));
                }
                self.indent -= 1;
                self.line("]");
            }
        }
    }
}

pub fn format_type(ty: &TypeIR) -> String {
    match ty {
        TypeIR::Bool => "bool".to_string(),
        TypeIR::Int => "int".to_string(),
        TypeIR::Float => "float".to_string(),
        TypeIR::String => "string".to_string(),
        TypeIR::Bytes => "bytes".to_string(),
        TypeIR::Any => "any".to_string(),
        TypeIR::Named { name } => name.clone(),
        TypeIR::List { element } => format!("list<{}>", format_type(element)),
        TypeIR::Map { key, value } => {
            format!("map<{}, {}>", format_type(key), format_type(value))
        }
        TypeIR::Option { inner } => format!("option<{}>", format_type(inner)),
        TypeIR::Struct { fields } => {
            let fs: Vec<String> = fields
                .iter()
                .map(|f| format!("{}: {}", f.name, format_type(&f.ty)))
                .collect();
            format!("{{ {} }}", fs.join(", "))
        }
    }
}

fn format_string_or_file(sof: &StringOrFileIR) -> String {
    match sof {
        StringOrFileIR::Literal { value } => format!("\"{}\"", value.replace('"', "\\\"")),
        StringOrFileIR::File { path } => format!("file(\"{}\")", path),
    }
}

fn format_step_arg(arg: &StepArgIR) -> String {
    match arg {
        StepArgIR::Positional { value } => format_expr(value),
        StepArgIR::Named { name, value } => format!("{}: {}", name, format_expr(value)),
    }
}

pub fn format_expr(expr: &ExprIR) -> String {
    match expr {
        ExprIR::LitInt { value } => value.to_string(),
        ExprIR::LitFloat { value } => format!("{}", value),
        ExprIR::LitString { value } => format!("\"{}\"", value.replace('"', "\\\"")),
        ExprIR::LitBool { value } => value.to_string(),
        ExprIR::LitNull => "null".to_string(),
        ExprIR::Ident { name } => name.clone(),
        ExprIR::FieldAccess { base, field } => {
            format!("{}.{}", format_expr(base), field)
        }
        ExprIR::Index { base, index } => {
            format!("{}[{}]", format_expr(base), format_expr(index))
        }
        ExprIR::UnaryNot { operand } => format!("!{}", format_expr(operand)),
        ExprIR::UnaryNeg { operand } => format!("-{}", format_expr(operand)),
        ExprIR::Binary { left, op, right } => {
            format!("{} {} {}", format_expr(left), op, format_expr(right))
        }
        ExprIR::Call { name, args } => {
            let a: Vec<String> = args.iter().map(format_expr).collect();
            format!("{}({})", name, a.join(", "))
        }
        ExprIR::List { elements } => {
            let e: Vec<String> = elements.iter().map(format_expr).collect();
            format!("[{}]", e.join(", "))
        }
        ExprIR::Record { fields } => {
            let fs: Vec<String> = fields
                .iter()
                .map(|f| format!("{}: {}", f.key, format_expr(&f.value)))
                .collect();
            format!("{{ {} }}", fs.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::{from_json, lower, to_json};
    use scaffold_syntax::parser::parse;

    #[test]
    fn pretty_print_round_trip_with_subs() {
        let src = r#"
            node solver: prompt {
                in: string
                out: string
                model: "gpt-4o"
                template: "Solve: {{ input }}"
            }
            graph inner {
                in: string
                out: string
                step s = solver(input)
                emit s
            }
            graph outer {
                in: string
                out: string
                step s = inner(input)
                emit s
            }
            objective eval {
                graph: outer
                dataset: cases [
                    { input: "x", expected: "y" }
                ]
                checker exact { output == expected }
                metric acc { checker: exact }
                score: acc

                sub inner_opt {
                    graph: inner
                    dataset: cases [
                        { input: "a", expected: "b" }
                    ]
                    checker sub_exact { output == expected }
                    metric sub_acc { checker: sub_exact }
                    score: sub_acc
                    topology {
                        mutations: [insert_verify]
                        max_nodes: 8
                    }
                }
            }
        "#;
        // Parse → Lower → Pretty print → Re-parse → Lower → Compare
        let program = parse(src).unwrap();
        let ir = lower(&program).unwrap();
        let printed = pretty_print(&ir);

        // Re-parse the printed output
        let program2 = parse(&printed).expect("pretty-printed output should re-parse");
        let ir2 = lower(&program2).unwrap();

        // Check structural equivalence
        assert_eq!(ir.objectives.len(), ir2.objectives.len());
        assert_eq!(ir.objectives[0].subs.len(), ir2.objectives[0].subs.len());
        assert_eq!(ir.objectives[0].subs[0].name, ir2.objectives[0].subs[0].name);
        assert_eq!(ir.objectives[0].subs[0].graph, ir2.objectives[0].subs[0].graph);

        // Also check JSON round-trip
        let json = to_json(&ir).unwrap();
        let ir3 = from_json(&json).unwrap();
        assert_eq!(ir3.objectives[0].subs.len(), 1);
    }

    #[test]
    fn pretty_print_node_round_trip() {
        let src = r#"
            node solver: prompt {
                in: string
                out: string
                model: "gpt-4o"
                temperature: 0.7
                template: "Solve: {{ input }}"
                system: "You are helpful."
            }
        "#;
        let program = parse(src).unwrap();
        let ir = lower(&program).unwrap();
        let node = &ir.nodes[0];

        let printed = pretty_print_node(node);
        assert!(printed.contains("node solver: prompt {"));
        assert!(printed.contains("model: \"gpt-4o\""));
        assert!(printed.contains("temperature: 0.7"));

        // Re-parse to verify it's valid
        let program2 = parse(&format!(
            "{}\ngraph g {{ in: string out: string step s = solver(input) emit s }}",
            printed
        ))
        .expect("pretty-printed node should re-parse");
        let ir2 = lower(&program2).unwrap();
        assert_eq!(ir2.nodes[0].name, "solver");
        assert_eq!(ir2.nodes[0].config.model.as_deref(), Some("gpt-4o"));
    }
}
