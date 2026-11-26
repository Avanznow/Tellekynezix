# Issue #90 - Wycheproof Download Reliability

## Contributors
- Ayush Silwal
- Samyog Ucchai
- Prabinb Sapkota

## Problem
Intermittent network failures can interrupt
automated Wycheproof test-vector downloads.

## Proposed Solution
- Maximum of three download attempts.
- Exponential backoff of one and two seconds.
- Network exception handling.
- Cache clearing after failed requests.
- Controlled skipping after exhausted retries.

## Testing Results
Four isolated functional tests passed:
- Successful download
- Temporary network failure
- Persistent network failure
- HTTP error recovery

## Upstream Contribution
https://github.com/open-quantum-safe/liboqs/pull/2746

## Limitations
The implementation belongs to upstream liboqs.
Full cryptographic and CI testing remains pending.
This documentation does not integrate liboqs
into the Tellekynezix codebase.

## Development Disclosure
ChatGPT assisted with implementation, testing
code and documentation. Tests were executed locally.
