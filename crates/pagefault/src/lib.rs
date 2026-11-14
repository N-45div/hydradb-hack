//! pagefault: a compaction API for coding agents.
//!
//! The context window is a cache. HydraDB is the memory. The graph learns
//! what to page back in after compaction, from every page fault it causes.

pub mod experiment;
pub mod extract;
pub mod graph;
pub mod hydra;
pub mod learn;
pub mod model;
pub mod policy;
pub mod sim;
