# Project Directives: Ponytail Mode

This project follows **Ponytail** principles (intensity: `full`).

## Core Rules
1. **YAGNI & Simplicity**: Question whether code needs to exist before writing it. Avoid speculative abstractions, unnecessary wrapper classes, factories, or premature generalizations.
2. **The Ladder**:
   - Stdlib / native platform capabilities first.
   - Already-installed dependencies before adding new ones.
   - Minimum code that works; shortest working diff wins.
3. **Fewest Files & Deletion Over Addition**:
   - Keep files consolidated where appropriate.
   - Delete unused code and boilerplate.
4. **Pragmatic Testing**:
   - Non-trivial logic keeps one runnable self-check or minimal test. No heavy test frameworks/fixtures unless explicitly required.
5. **Concise Output**:
   - Code first, minimal prose.
