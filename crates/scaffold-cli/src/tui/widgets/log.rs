use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem};

use crate::tui::app::{AppState, LogKind};

pub fn render_log(state: &AppState, height: usize) -> List<'_> {
    let visible_height = height.saturating_sub(2); // borders
    let total = state.log_entries.len();

    // Determine visible window
    let start = if total <= visible_height {
        0
    } else if state.user_scrolled {
        state.log_scroll.min(total.saturating_sub(visible_height))
    } else {
        total.saturating_sub(visible_height)
    };

    let end = (start + visible_height).min(total);

    let items: Vec<ListItem<'_>> = state.log_entries[start..end]
        .iter()
        .map(|entry| {
            let ts = format!("+{:.1}s", entry.elapsed_secs);
            let style = match entry.kind {
                LogKind::NewBest => Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
                LogKind::Phase => Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
                LogKind::Skip => Style::default().fg(Color::DarkGray),
                LogKind::EarlyStop => Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
                LogKind::MetaAgent => Style::default().fg(Color::Magenta),
                LogKind::Normal => Style::default().fg(Color::White),
            };
            let line = Line::from(vec![
                Span::styled(format!("{:>8} ", ts), Style::default().fg(Color::DarkGray)),
                Span::styled(&entry.message, style),
            ]);
            ListItem::new(line)
        })
        .collect();

    let scroll_indicator = if total > visible_height && state.user_scrolled {
        format!("  [{}/{}]", start + 1, total)
    } else {
        String::new()
    };

    let title = if state.finished {
        format!(" Log (done){} ", scroll_indicator)
    } else {
        format!(" Log (j/k scroll, q quit){} ", scroll_indicator)
    };

    List::new(items).block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    )
}
