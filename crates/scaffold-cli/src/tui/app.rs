use std::time::Instant;

use scaffold_runtime::{OptEvent, OptPhase};

/// A candidate entry for the lineage tree.
pub struct CandidateEntry {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub score: f64,
    pub mutations: Vec<String>,
    pub is_best: bool,
}

/// A log entry with relative timestamp and style.
pub struct LogEntry {
    pub elapsed_secs: f64,
    pub message: String,
    pub kind: LogKind,
}

pub enum LogKind {
    Normal,
    NewBest,
    Phase,
    Skip,
    EarlyStop,
    MetaAgent,
}

/// Progress of the currently evaluating candidate.
pub struct EvalProgress {
    pub candidate_id: usize,
    pub cases_done: usize,
    pub total_cases: usize,
    pub cases_passed: usize,
}

pub struct AppState {
    pub candidates: Vec<CandidateEntry>,
    pub score_history: Vec<(f64, f64)>, // (index as f64, score)
    pub best_score: f64,
    pub best_id: Option<usize>,
    pub log_entries: Vec<LogEntry>,
    pub current_phase: String,
    pub current_sub: Option<String>,
    pub eval_progress: Option<EvalProgress>,
    pub log_scroll: usize,
    pub user_scrolled: bool,
    pub finished: bool,
    pub start_time: Instant,
    /// Meta-agent model name (None = random mutations).
    pub meta_model: Option<String>,
    /// True while meta-agent is making an LLM call.
    pub meta_thinking: bool,
    /// Models used by harness nodes.
    pub harness_models: Vec<String>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            score_history: Vec::new(),
            best_score: 0.0,
            best_id: None,
            log_entries: Vec::new(),
            current_phase: "Starting".into(),
            current_sub: None,
            eval_progress: None,
            log_scroll: 0,
            user_scrolled: false,
            finished: false,
            start_time: Instant::now(),
            meta_model: None,
            meta_thinking: false,
            harness_models: Vec::new(),
        }
    }

    fn elapsed(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    fn push_log(&mut self, message: String, kind: LogKind) {
        self.log_entries.push(LogEntry {
            elapsed_secs: self.elapsed(),
            message,
            kind,
        });
        if !self.user_scrolled {
            // Auto-scroll to bottom
            self.log_scroll = self.log_entries.len().saturating_sub(1);
        }
    }

    fn update_best(&mut self) {
        // Recompute best
        let mut best_score = 0.0_f64;
        let mut best_id = None;
        for c in &self.candidates {
            if c.score > best_score {
                best_score = c.score;
                best_id = Some(c.id);
            }
        }
        self.best_score = best_score;
        self.best_id = best_id;

        // Update is_best flags
        for c in &mut self.candidates {
            c.is_best = Some(c.id) == self.best_id;
        }
    }

    pub fn process_event(&mut self, event: OptEvent) {
        match event {
            OptEvent::SubObjectiveStarted { sub_name, graph_name } => {
                // Clear candidate state — new phase restarts IDs from 0
                self.candidates.clear();
                self.score_history.clear();
                self.best_score = 0.0;
                self.best_id = None;
                self.eval_progress = None;

                self.current_sub = Some(sub_name.clone());
                self.push_log(
                    format!("Sub-objective '{}' started (graph: {})", sub_name, graph_name),
                    LogKind::Phase,
                );
            }
            OptEvent::SubObjectiveCompleted { sub_name, best_score, total_candidates } => {
                self.push_log(
                    format!(
                        "Sub '{}' completed: {} candidates, best={:.4}",
                        sub_name,
                        total_candidates,
                        best_score.unwrap_or(0.0)
                    ),
                    LogKind::Phase,
                );
                self.current_sub = None;
            }
            OptEvent::ParentPhaseStarted { objective_name } => {
                // Clear candidate state — parent phase restarts IDs from 0
                self.candidates.clear();
                self.score_history.clear();
                self.best_score = 0.0;
                self.best_id = None;
                self.eval_progress = None;

                self.push_log(
                    format!("Parent objective '{}' optimization started", objective_name),
                    LogKind::Phase,
                );
            }
            OptEvent::PhaseChanged { phase } => {
                let name = match phase {
                    OptPhase::Seeding => "Seeding",
                    OptPhase::TunableSweep => "Tunable Sweep",
                    OptPhase::Evolutionary => "Evolutionary",
                };
                self.current_phase = name.to_string();
                self.push_log(format!("Phase: {}", name), LogKind::Phase);
            }
            OptEvent::EvaluationStarted { candidate_id, total_cases } => {
                self.meta_thinking = false;
                self.eval_progress = Some(EvalProgress {
                    candidate_id,
                    cases_done: 0,
                    total_cases,
                    cases_passed: 0,
                });
                self.push_log(
                    format!("Evaluating candidate #{} ({} cases)...", candidate_id, total_cases),
                    LogKind::Normal,
                );
            }
            OptEvent::CaseCompleted { candidate_id, case_index, total_cases, passed, case_id } => {
                if let Some(ref mut prog) = self.eval_progress {
                    if prog.candidate_id == candidate_id {
                        prog.cases_done = case_index;
                        if passed { prog.cases_passed += 1; }
                    }
                }
                let status = if passed { "PASS" } else { "FAIL" };
                let id_str = case_id.unwrap_or_else(|| format!("{}", case_index));
                self.push_log(
                    format!(
                        "  case {} {}/{} {}",
                        id_str, case_index, total_cases, status
                    ),
                    if passed { LogKind::Normal } else { LogKind::Skip },
                );
            }
            OptEvent::CandidateEvaluated {
                candidate_id,
                parent_id,
                score,
                metric_scores: _,
                best_so_far: _,
                mutations,
                generation,
                max_generations,
            } => {
                self.eval_progress = None;
                let is_new_best = score > self.best_score;

                self.candidates.push(CandidateEntry {
                    id: candidate_id,
                    parent_id,
                    score,
                    mutations: mutations.clone(),
                    is_best: false,
                });

                let best_so_far = self.score_history
                    .last()
                    .map(|&(_, prev_best)| prev_best.max(score))
                    .unwrap_or(score);
                self.score_history
                    .push((candidate_id as f64, best_so_far));

                self.update_best();

                let label = if mutations.is_empty() {
                    "seed".to_string()
                } else {
                    mutations.join(", ")
                };

                let best_marker = if is_new_best { " *NEW BEST*" } else { "" };
                self.push_log(
                    format!(
                        "Candidate #{} [{}] score={:.4} (gen {}/{}){}",
                        candidate_id, label, score, generation, max_generations, best_marker
                    ),
                    if is_new_best { LogKind::NewBest } else { LogKind::Normal },
                );
            }
            OptEvent::MutationSkipped { reason } => {
                self.push_log(format!("Mutation skipped: {}", reason), LogKind::Skip);
            }
            OptEvent::MetaProposal { reasoning, mutation_label } => {
                self.meta_thinking = false;
                self.push_log(
                    format!("Meta-agent: {} [{}]", reasoning, mutation_label),
                    LogKind::MetaAgent,
                );
            }
            OptEvent::MetaAgentActive { model } => {
                self.push_log(
                    format!("Meta-agent active: {}", model),
                    LogKind::MetaAgent,
                );
                self.meta_model = Some(model);
            }
            OptEvent::MetaAgentThinking => {
                self.meta_thinking = true;
            }
            OptEvent::EarlyStopped { score, target } => {
                self.push_log(
                    format!("Early stop! score={:.4} >= target={:.4}", score, target),
                    LogKind::EarlyStop,
                );
            }
            OptEvent::Completed { objective_name, best_score, total_candidates } => {
                self.push_log(
                    format!(
                        "Completed '{}': {} candidates, best={:.4}",
                        objective_name,
                        total_candidates,
                        best_score.unwrap_or(0.0)
                    ),
                    LogKind::Phase,
                );
                self.finished = true;
            }
            OptEvent::HarnessInfo { models } => {
                self.harness_models = models;
            }
            OptEvent::Log { message } => {
                self.push_log(message, LogKind::Normal);
            }
        }
    }

    pub fn push_log_direct(&mut self, message: String, kind: LogKind) {
        self.push_log(message, kind);
    }

    pub fn scroll_down(&mut self) {
        self.user_scrolled = true;
        if self.log_scroll < self.log_entries.len().saturating_sub(1) {
            self.log_scroll += 1;
        }
    }

    pub fn scroll_up(&mut self) {
        self.user_scrolled = true;
        self.log_scroll = self.log_scroll.saturating_sub(1);
    }
}
