# llm-weighted-consensus-benchmarks

We believe that llm-weighted-consensus is a production-ready system which enables novel applications of LLMs as well as the application of LLMs into domains which would not be otherwise feasible.

This repository contains verifiable & reproducible benchmarks which attempt to prove:
- llm-weighted-consensus increases LLM correctness (and, arguably, intelligence)
- llm-weighted-consensus increases LLM reliability
- llm-weighted-consensus decreases LLM cost

Each directory wthin the repository contains a script for running the benchmark, the results of the benchmark, and a writeup for the results.

With our Humanity's Last Exam benchmark, we demonstrate that a Score Model can increase correctness on Humanity's Last Exam beyond what the LLM inside the Score Model would typically be capable of.

With our Poll Approximation benchmark, we contrast between using a traditional LLM to approximate or guess Gallup Poll results and using a Score Model to simulate the poll itself.

With our IT Ticket Classification benchmark, we demonstrate that a Score Model can simultaneously increase correctness for the task while being cheaper than a traditional LLM.
