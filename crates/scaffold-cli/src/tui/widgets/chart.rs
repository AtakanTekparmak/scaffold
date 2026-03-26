use ratatui::prelude::*;
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, GraphType};

use crate::tui::app::AppState;

pub fn render_chart<'a>(state: &'a AppState) -> Chart<'a> {
    let data = &state.score_history;

    let x_max = if data.is_empty() {
        10.0
    } else {
        data.last().unwrap().0.max(1.0)
    };

    let datasets = vec![Dataset::default()
        .name("score")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(data)];

    let x_labels: Vec<Line<'_>> = vec![
        Line::from("0"),
        Line::from(format!("{}", (x_max / 2.0) as usize)),
        Line::from(format!("{}", x_max as usize)),
    ];

    let y_labels: Vec<Line<'_>> = vec![
        Line::from("0.0"),
        Line::from("0.5"),
        Line::from("1.0"),
    ];

    Chart::new(datasets)
        .block(
            Block::default()
                .title(format!(" Score Progression  [phase: {}] ", state.current_phase))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .x_axis(
            Axis::default()
                .title("candidate")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("score")
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, 1.0])
                .labels(y_labels),
        )
}
