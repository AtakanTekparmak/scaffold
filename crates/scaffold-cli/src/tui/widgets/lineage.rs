use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::tui::app::AppState;

/// Build a tree of candidates as styled lines using box-drawing characters.
pub fn render_lineage(state: &AppState) -> Paragraph<'_> {
    if state.candidates.is_empty() {
        return Paragraph::new("  Waiting for candidates...").block(
            Block::default()
                .title(" Lineage Tree ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    }

    // Build parent -> children map
    let mut children: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    let mut roots: Vec<usize> = Vec::new();

    for c in &state.candidates {
        match c.parent_id {
            Some(pid) => {
                children.entry(pid).or_default().push(c.id);
            }
            None => roots.push(c.id),
        }
    }

    let mut lines: Vec<Line<'_>> = Vec::new();

    for (ri, &root_id) in roots.iter().enumerate() {
        let is_last_root = ri == roots.len() - 1;
        render_node(
            root_id,
            "",
            is_last_root,
            true,
            state,
            &children,
            &mut lines,
        );
    }

    Paragraph::new(lines).block(
        Block::default()
            .title(format!(" Lineage Tree  [best: {:.4}] ", state.best_score))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    )
}

fn render_node<'a>(
    id: usize,
    prefix: &str,
    is_last: bool,
    is_root: bool,
    state: &AppState,
    children: &std::collections::HashMap<usize, Vec<usize>>,
    lines: &mut Vec<Line<'a>>,
) {
    let candidate = match state.candidates.iter().find(|c| c.id == id) {
        Some(c) => c,
        None => return,
    };

    let connector = if is_root {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };

    let label = if candidate.mutations.is_empty() {
        "seed".to_string()
    } else {
        candidate.mutations.join(", ")
    };

    let score_str = format!("{:.4}", candidate.score);

    let best_marker = if candidate.is_best { " * BEST" } else { "" };

    let style = if candidate.is_best {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };

    let text = format!(
        "{}{}[#{}] {}  {}{}",
        prefix, connector, id, label, score_str, best_marker
    );
    lines.push(Line::from(Span::styled(text, style)));

    // Recurse into children
    let child_ids = children.get(&id).cloned().unwrap_or_default();
    let child_prefix = if is_root {
        "".to_string()
    } else if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}│   ", prefix)
    };

    for (ci, &child_id) in child_ids.iter().enumerate() {
        let is_last_child = ci == child_ids.len() - 1;
        render_node(
            child_id,
            &child_prefix,
            is_last_child,
            false,
            state,
            children,
            lines,
        );
    }
}
