# llm-weighted-consensus-benchmarks

We believe that llm-weighted-consensus is a production-ready system which enables novel applications of LLMs as well as the application of LLMs into domains which would not be otherwise feasible.

This repository contains verifiable & reproducible benchmarks which attempt to prove:
- llm-weighted-consensus increases LLM correctness (and, arguably, intelligence)
- llm-weighted-consensus increases LLM reliability
- llm-weighted-consensus decreases LLM cost

Each directory wthin the repository contains a script for running the benchmark, the results of the benchmark, and a writeup for the results.

## Humanity's Last Exam

We demonstrate that a Score Model can increase correctness on Humanity's Last Exam beyond what the LLM inside the Score Model would typically be capable of.

We also demonstrate that composing the Score Model of heterogeneous LLMs can exceed the individual capabilities of those within it, even when some of those LLMs are individually less correct than others.

## Poll Approximation

We contrast between:
- Using a traditional LLM to approximate or guess Gallup Poll results.
- Using a Score Model to simulate the poll itself.

## IT Ticket Classification

We demonstrate that a Score Model can simultaneously increase correctness for the task while being cheaper than a traditional LLM.
