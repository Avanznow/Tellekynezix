# Avatar link management

`robot_services.link_manager.LinkManager` is a transport-independent state
machine for authenticated LoRa heartbeats. The network adapter is responsible
for authenticating frames before calling `heartbeat()` and for polling the
manager regularly with monotonic time.

## States

- `disconnected`: no peer session has been established.
- `connected`: recent heartbeat and signal measurements are above configured
  limits (or unavailable).
- `degraded`: heartbeat is recent, but a reported RSSI or SNR is below its
  configured threshold.
- `link_lost`: the heartbeat timeout elapsed.
- `fail_safe`: the application safe-action callback ran after link loss.
- `reconnecting`: the reconnect callback is being attempted on the bounded
  exponential-backoff schedule.

The state path is `connected -> degraded -> link_lost -> fail_safe ->
reconnecting -> connected`. A valid newer heartbeat recovers the session and
resets retry backoff. Duplicate or stale sequence numbers are rejected, with
32-bit sequence wrap supported. Missing sequence numbers increment
`missed_heartbeats` and mark the link degraded until the next in-order heartbeat.
The optional state-change callback receives every transition (including the
transient `link_lost` state) so a UI or event log can report the full lifecycle.

Callbacks are supplied by the application. The fail-safe callback must be
idempotent and perform only the product-approved safe action. The manager does
not retry or replay movement commands, own sockets, or claim a physical device
has reached a safe posture. Applications call `poll()` from their own event
loop and pass the same monotonic time base used for heartbeat receipt times.
Callbacks run synchronously within `poll()` and should be bounded and
non-blocking.
