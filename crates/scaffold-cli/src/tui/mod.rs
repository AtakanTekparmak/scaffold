pub mod app;
mod widgets;

use std::io::{self, Write as _};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Gauge, LineGauge};
use ratatui::Terminal;

use scaffold_runtime::{HierarchicalReport, OptEvent};

use self::app::AppState;
use self::widgets::{chart, lineage, log};

/// Run the TUI event loop. Returns the final report when the optimizer finishes.
///
/// - `event_rx`: receives OptEvent from the optimizer thread
/// - `result_rx`: receives the final Result<HierarchicalReport> when optimizer is done
pub fn run_tui(
    event_rx: tokio::sync::mpsc::UnboundedReceiver<OptEvent>,
    result_rx: std::sync::mpsc::Receiver<Result<HierarchicalReport, String>>,
) -> Result<HierarchicalReport, String> {
    // Setup terminal
    enable_raw_mode().map_err(|e| format!("failed to enable raw mode: {}", e))?;
    io::stdout()
        .execute(EnterAlternateScreen)
        .map_err(|e| format!("failed to enter alternate screen: {}", e))?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal =
        Terminal::new(backend).map_err(|e| format!("failed to create terminal: {}", e))?;

    let result = run_loop(&mut terminal, event_rx, &result_rx);

    // Teardown terminal — must happen before any blocking wait
    disable_raw_mode().ok();
    io::stdout().execute(LeaveAlternateScreen).ok();
    // Force a flush so the alternate screen is fully exited
    drop(terminal);

    result
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut event_rx: tokio::sync::mpsc::UnboundedReceiver<OptEvent>,
    result_rx: &std::sync::mpsc::Receiver<Result<HierarchicalReport, String>>,
) -> Result<HierarchicalReport, String> {
    let mut state = AppState::new();
    let mut quit_requested = false;

    // Incremental log file — survives early quit
    let log_path = std::path::Path::new("runs").join("latest.log");
    std::fs::create_dir_all("runs").ok();
    let mut log_file = std::fs::File::create(&log_path).ok();

    loop {
        // Draw
        terminal
            .draw(|frame| draw_ui(frame, &state))
            .map_err(|e| format!("draw error: {}", e))?;

        // Poll crossterm events (100ms timeout)
        if event::poll(Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind == KeyEventKind::Press {
                    // Ctrl+C always quits immediately
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        quit_requested = true;
                    } else {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                quit_requested = true;
                            }
                            KeyCode::Tab => state.toggle_focus(),
                            KeyCode::Char('j') | KeyCode::Down => state.scroll_down(),
                            KeyCode::Char('k') | KeyCode::Up => state.scroll_up(),
                            _ => {}
                        }
                    }
                }
            }
        }

        // Drain optimizer events
        loop {
            match event_rx.try_recv() {
                Ok(evt) => {
                    if let Some(ref mut f) = log_file {
                        let _ = writeln!(
                            f,
                            "+{:.1}s {:?}",
                            state.start_time.elapsed().as_secs_f64(),
                            evt
                        );
                    }
                    state.process_event(evt);
                }
                Err(_) => break,
            }
        }

        // Check if optimizer is done (result available)
        if let Ok(result) = result_rx.try_recv() {
            // Drain any remaining events
            loop {
                match event_rx.try_recv() {
                    Ok(evt) => {
                        if let Some(ref mut f) = log_file {
                            let _ = writeln!(
                                f,
                                "+{:.1}s {:?}",
                                state.start_time.elapsed().as_secs_f64(),
                                evt
                            );
                        }
                        state.process_event(evt);
                    }
                    Err(_) => break,
                }
            }

            if quit_requested {
                return result;
            }

            // Show final state, wait for q to exit
            state.finished = true;
            state.push_log_direct(
                "Optimizer finished. Press q to exit.".into(),
                app::LogKind::Phase,
            );

            loop {
                terminal
                    .draw(|frame| draw_ui(frame, &state))
                    .map_err(|e| format!("draw error: {}", e))?;

                if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                    if let Ok(Event::Key(key)) = event::read() {
                        if key.kind == KeyEventKind::Press {
                            if key.code == KeyCode::Char('c')
                                && key.modifiers.contains(KeyModifiers::CONTROL)
                            {
                                return result;
                            }
                            match key.code {
                                KeyCode::Char('q') | KeyCode::Esc => return result,
                                KeyCode::Tab => state.toggle_focus(),
                                KeyCode::Char('j') | KeyCode::Down => state.scroll_down(),
                                KeyCode::Char('k') | KeyCode::Up => state.scroll_up(),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // If user pressed q but optimizer is still running, restore terminal
        // and exit immediately — no waiting for the optimizer thread.
        if quit_requested {
            disable_raw_mode().ok();
            io::stdout().execute(LeaveAlternateScreen).ok();
            // Flush to ensure terminal escape sequences are written
            let _ = io::stdout().flush();
            let _ = io::stderr().flush();
            eprintln!("Cancelled.");
            std::process::exit(0);
        }
    }
}

fn draw_ui(frame: &mut Frame, state: &AppState) {
    let area = frame.area();
    let has_meta = state.meta_model.is_some();

    // Top 55%, progress bar 3 lines, optional meta context 1 line, bottom rest
    let main_layout = if has_meta {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(55),
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Min(5),
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(55),
                Constraint::Length(3),
                Constraint::Length(0),
                Constraint::Min(5),
            ])
            .split(area)
    };

    // Top row: lineage 50% | chart 50%
    let top_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_layout[0]);

    // Render lineage tree with scroll
    let (lineage_widget, total_lines) = lineage::render_lineage(state);
    let lineage_visible = top_layout[0].height.saturating_sub(2) as usize; // minus borders
    let scroll_offset = if state.lineage_user_scrolled {
        // Manual scroll — clamp to valid range
        state.lineage_scroll.min(total_lines.saturating_sub(lineage_visible))
    } else {
        // Auto-scroll to bottom
        total_lines.saturating_sub(lineage_visible)
    };
    frame.render_widget(
        lineage_widget.scroll((scroll_offset as u16, 0)),
        top_layout[0],
    );

    // Render score chart
    let chart_widget = chart::render_chart(state);
    frame.render_widget(chart_widget, top_layout[1]);

    // Render progress bar
    let gauge = render_progress(state);
    frame.render_widget(gauge, main_layout[1]);

    // Render meta-agent context bar (persistent when meta-model is active)
    if has_meta {
        let ctx_bar = render_meta_context(state);
        frame.render_widget(ctx_bar, main_layout[2]);
    }

    // Render log panel
    let log_height = main_layout[3].height as usize;
    let log_widget = log::render_log(state, log_height);
    frame.render_widget(log_widget, main_layout[3]);
}

fn render_progress(state: &AppState) -> Gauge<'_> {
    let models_tag = if state.harness_models.is_empty() {
        String::new()
    } else {
        format!("  models: [{}]", state.harness_models.join(", "))
    };
    let meta_tag = match &state.meta_model {
        Some(m) => format!("  meta: {}", m),
        None => String::new(),
    };

    if let Some(ref prog) = state.eval_progress {
        let ratio = if prog.total_cases > 0 {
            prog.cases_done as f64 / prog.total_cases as f64
        } else {
            0.0
        };
        let label = format!(
            "Candidate #{}: {}/{} cases ({} passed)",
            prog.candidate_id, prog.cases_done, prog.total_cases, prog.cases_passed,
        );
        Gauge::default()
            .block(
                Block::default()
                    .title(format!(
                        " Evaluating  [{}]{}{} ",
                        state.current_phase, models_tag, meta_tag
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow)),
            )
            .gauge_style(Style::default().fg(Color::Yellow).bg(Color::DarkGray))
            .ratio(ratio.min(1.0))
            .label(label)
    } else if state.finished {
        Gauge::default()
            .block(
                Block::default()
                    .title(" Done ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Green)),
            )
            .gauge_style(Style::default().fg(Color::Green).bg(Color::DarkGray))
            .ratio(1.0)
            .label(format!(
                "Best score: {:.4}  |  {} candidates",
                state.best_score,
                state.candidates.len()
            ))
    } else if let Some((done, total)) = state.summarizing {
        let ratio = if total > 0 {
            done as f64 / total as f64
        } else {
            0.0
        };
        Gauge::default()
            .block(
                Block::default()
                    .title(format!(
                        " Compressing failure logs{}{} ",
                        models_tag, meta_tag,
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::LightBlue)),
            )
            .gauge_style(Style::default().fg(Color::LightBlue).bg(Color::DarkGray))
            .ratio(ratio.min(1.0))
            .label(format!(
                "Summarizing {}/{} failures (context too large)",
                done, total
            ))
    } else if state.meta_thinking {
        Gauge::default()
            .block(
                Block::default()
                    .title(format!(
                        " {}  [{}]{}{} ",
                        state.current_sub.as_deref().unwrap_or("Optimizer"),
                        state.current_phase,
                        models_tag,
                        meta_tag,
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Magenta)),
            )
            .gauge_style(Style::default().fg(Color::Magenta).bg(Color::DarkGray))
            .ratio(0.0)
            .label("Meta-agent thinking...")
    } else {
        Gauge::default()
            .block(
                Block::default()
                    .title(format!(
                        " {}  [{}]{}{} ",
                        state.current_sub.as_deref().unwrap_or("Optimizer"),
                        state.current_phase,
                        models_tag,
                        meta_tag,
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
            .ratio(0.0)
            .label("Waiting...")
    }
}

fn render_meta_context(state: &AppState) -> LineGauge<'_> {
    let ctx_window = 128_000usize;
    match state.meta_context_tokens {
        Some(tok) => {
            let ratio = (tok as f64 / ctx_window as f64).min(1.0);
            let pct = ratio * 100.0;
            let color = if pct > 80.0 {
                Color::Red
            } else if pct > 60.0 {
                Color::Yellow
            } else {
                Color::Magenta
            };
            let label = format!(
                " latest meta ctx: ~{}k / {}k tok ({:.0}%) ",
                tok / 1000,
                ctx_window / 1000,
                pct,
            );
            LineGauge::default()
                .filled_style(Style::default().fg(color))
                .unfilled_style(Style::default().fg(Color::DarkGray))
                .ratio(ratio)
                .label(label)
        }
        None => LineGauge::default()
            .filled_style(Style::default().fg(Color::DarkGray))
            .unfilled_style(Style::default().fg(Color::DarkGray))
            .ratio(0.0)
            .label(" latest meta ctx: waiting for first proposal "),
    }
}
