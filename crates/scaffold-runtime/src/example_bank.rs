//! Example banks: per-node I/O example storage for few-shot prompting.
//!
//! After seed evaluation, the optimizer builds an example bank from train results.
//! Each node can have an attached example selection policy that determines which
//! examples are injected into its prompt at execution time.
//!
//! **Shared memory role**: The `ExampleBank` is the cross-candidate shared memory
//! mechanism. It is built once from seed evaluation results, wrapped in `Arc`, and
//! shared read-only across all candidate evaluations. Per-candidate customization
//! happens via `ExamplePolicy` (strategy + k), not by mutating the bank itself.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single I/O example for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Truncated input excerpt.
    pub input_excerpt: String,
    /// Correct output for this subtask.
    pub output: String,
    /// Tags for categorization (e.g., "confusion", "boundary", "recovery").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Optional domain/category label for domain-conditioned selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

/// Lightweight eval case for building an example bank without circular deps on optimizer.
pub struct EvalCaseForBank {
    pub case_id: String,
    pub input_excerpt: String,
    pub expected_output: String,
    pub passed: bool,
    /// Optional domain/category label from the dataset.
    pub domain: Option<String>,
}

/// Per-node example bank: maps node names to their example sets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExampleBank {
    pub examples: HashMap<String, Vec<Example>>,
}

impl ExampleBank {
    pub fn new() -> Self {
        Self {
            examples: HashMap::new(),
        }
    }

    /// Get examples for a node, if any.
    pub fn get(&self, node: &str) -> Option<&[Example]> {
        self.examples.get(node).map(|v| v.as_slice())
    }

    /// Add examples for a node.
    pub fn insert(&mut self, node: String, examples: Vec<Example>) {
        self.examples.insert(node, examples);
    }

    /// Build an example bank from seed evaluation results.
    ///
    /// Passed cases get no tags; failed cases get a `"failed"` tag.
    /// The `ConfusionCover` strategy prioritizes `"failed"` examples, so
    /// candidates that request confusion-cover will preferentially see
    /// the cases the seed got wrong.
    pub fn build_from_eval(cases: &[EvalCaseForBank], node_name: &str) -> Self {
        let examples: Vec<Example> = cases
            .iter()
            .map(|c| Example {
                input_excerpt: c.input_excerpt.clone(),
                output: c.expected_output.clone(),
                tags: if c.passed {
                    vec![]
                } else {
                    vec!["failed".to_string()]
                },
                domain: c.domain.clone(),
            })
            .collect();
        let mut bank = Self::new();
        bank.insert(node_name.to_string(), examples);
        bank
    }

    /// Check if the bank has any examples.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty() || self.examples.values().all(|v| v.is_empty())
    }

    /// Select examples for a node based on a policy.
    ///
    /// `input_domain` is used by `DomainConditioned` strategy to prefer same-domain examples.
    pub fn select(&self, node: &str, policy: &ExamplePolicy, seed: u64, input_domain: Option<&str>) -> Vec<Example> {
        let Some(all_examples) = self.examples.get(node) else {
            return vec![];
        };
        if all_examples.is_empty() {
            return vec![];
        }

        let k = policy.k.min(all_examples.len());
        if k == 0 {
            return vec![];
        }

        match policy.strategy {
            ExampleStrategy::Random => {
                // Deterministic selection based on seed
                let mut indices: Vec<usize> = (0..all_examples.len()).collect();
                // Simple deterministic shuffle using seed
                for i in (1..indices.len()).rev() {
                    let j = ((seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64))
                        % (i as u64 + 1)) as usize;
                    indices.swap(i, j);
                }
                indices.truncate(k);
                indices.iter().map(|&i| all_examples[i].clone()).collect()
            }
            ExampleStrategy::ConfusionCover => {
                // Prioritize examples tagged "confusion" or "failed"
                let mut selected: Vec<Example> = all_examples
                    .iter()
                    .filter(|e| {
                        e.tags.contains(&"confusion".to_string())
                            || e.tags.contains(&"failed".to_string())
                    })
                    .take(k)
                    .cloned()
                    .collect();
                // Fill remaining with others
                if selected.len() < k {
                    for e in all_examples {
                        if selected.len() >= k {
                            break;
                        }
                        if !e.tags.contains(&"confusion".to_string())
                            && !e.tags.contains(&"failed".to_string())
                        {
                            selected.push(e.clone());
                        }
                    }
                }
                selected
            }
            ExampleStrategy::NearestPlusHardNegative => {
                // Take first as "nearest", second as hard negative, fill rest
                all_examples.iter().take(k).cloned().collect()
            }
            ExampleStrategy::SyntheticAliases => {
                // Take examples tagged "alias" first
                let mut selected: Vec<Example> = all_examples
                    .iter()
                    .filter(|e| e.tags.contains(&"alias".to_string()))
                    .take(k)
                    .cloned()
                    .collect();
                if selected.len() < k {
                    for e in all_examples {
                        if selected.len() >= k {
                            break;
                        }
                        if !e.tags.contains(&"alias".to_string()) {
                            selected.push(e.clone());
                        }
                    }
                }
                selected
            }
            ExampleStrategy::DomainConditioned => {
                // Prefer same-domain examples, fill from other domains
                let mut selected: Vec<Example> = Vec::new();
                if let Some(domain) = input_domain {
                    // Same-domain first
                    for e in all_examples {
                        if selected.len() >= k {
                            break;
                        }
                        if e.domain.as_deref() == Some(domain) {
                            selected.push(e.clone());
                        }
                    }
                }
                // Fill remaining from other domains
                if selected.len() < k {
                    for e in all_examples {
                        if selected.len() >= k {
                            break;
                        }
                        // Skip already-selected same-domain examples
                        let is_same_domain = input_domain
                            .map(|d| e.domain.as_deref() == Some(d))
                            .unwrap_or(false);
                        if !is_same_domain {
                            selected.push(e.clone());
                        }
                    }
                }
                selected
            }
        }
    }
}

/// Example selection policy for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamplePolicy {
    /// Selection strategy.
    pub strategy: ExampleStrategy,
    /// Number of examples to include.
    pub k: usize,
}

/// Strategy for selecting examples from the bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExampleStrategy {
    /// Cover the most-confused label pairs (from train confusion matrix).
    ConfusionCover,
    /// Nearest input + one hard negative.
    NearestPlusHardNegative,
    /// Synthetic alias variations (for normalizer nodes).
    SyntheticAliases,
    /// Random uniform sample from train.
    Random,
    /// Prefer same-domain examples, fill from other domains.
    DomainConditioned,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_bank() {
        let bank = ExampleBank::new();
        assert!(bank.is_empty());
        assert!(bank.get("foo").is_none());
    }

    #[test]
    fn test_insert_and_get() {
        let mut bank = ExampleBank::new();
        bank.insert(
            "classify".to_string(),
            vec![Example {
                input_excerpt: "hello".to_string(),
                output: "greeting".to_string(),
                tags: vec![],
                domain: None,
            }],
        );
        assert!(!bank.is_empty());
        assert_eq!(bank.get("classify").unwrap().len(), 1);
    }

    #[test]
    fn test_select_random() {
        let mut bank = ExampleBank::new();
        bank.insert(
            "node".to_string(),
            (0..10)
                .map(|i| Example {
                    input_excerpt: format!("input_{}", i),
                    output: format!("output_{}", i),
                    tags: vec![],
                    domain: None,
                })
                .collect(),
        );

        let policy = ExamplePolicy {
            strategy: ExampleStrategy::Random,
            k: 3,
        };
        let selected = bank.select("node", &policy, 42, None);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_select_confusion_cover() {
        let mut bank = ExampleBank::new();
        bank.insert(
            "node".to_string(),
            vec![
                Example {
                    input_excerpt: "a".to_string(),
                    output: "1".to_string(),
                    tags: vec!["confusion".to_string()],
                    domain: None,
                },
                Example {
                    input_excerpt: "b".to_string(),
                    output: "2".to_string(),
                    tags: vec![],
                    domain: None,
                },
                Example {
                    input_excerpt: "c".to_string(),
                    output: "3".to_string(),
                    tags: vec!["confusion".to_string()],
                    domain: None,
                },
            ],
        );

        let policy = ExamplePolicy {
            strategy: ExampleStrategy::ConfusionCover,
            k: 2,
        };
        let selected = bank.select("node", &policy, 0, None);
        assert_eq!(selected.len(), 2);
        // Both should be confusion-tagged
        assert!(selected.iter().all(|e| e.tags.contains(&"confusion".to_string())));
    }

    #[test]
    fn test_select_k_exceeds_available() {
        let mut bank = ExampleBank::new();
        bank.insert(
            "node".to_string(),
            vec![Example {
                input_excerpt: "a".to_string(),
                output: "1".to_string(),
                tags: vec![],
                domain: None,
            }],
        );

        let policy = ExamplePolicy {
            strategy: ExampleStrategy::Random,
            k: 5,
        };
        let selected = bank.select("node", &policy, 0, None);
        assert_eq!(selected.len(), 1); // capped at available
    }

    #[test]
    fn test_select_domain_conditioned() {
        let mut bank = ExampleBank::new();
        bank.insert(
            "node".to_string(),
            vec![
                Example {
                    input_excerpt: "a".to_string(),
                    output: "1".to_string(),
                    tags: vec![],
                    domain: Some("medical".to_string()),
                },
                Example {
                    input_excerpt: "b".to_string(),
                    output: "2".to_string(),
                    tags: vec![],
                    domain: Some("legal".to_string()),
                },
                Example {
                    input_excerpt: "c".to_string(),
                    output: "3".to_string(),
                    tags: vec![],
                    domain: Some("medical".to_string()),
                },
            ],
        );

        let policy = ExamplePolicy {
            strategy: ExampleStrategy::DomainConditioned,
            k: 2,
        };
        let selected = bank.select("node", &policy, 0, Some("medical"));
        assert_eq!(selected.len(), 2);
        // Both should be medical domain
        assert!(selected.iter().all(|e| e.domain.as_deref() == Some("medical")));
    }
}
