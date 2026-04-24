# Server Static Configuration for Remote Access (Ticket #9)

## Goal
Configure a stable static IP + secure remote access (SSH) so students can reliably connect to the Avatar server for benchmarking and training tasks.

> ⚠️ Note: This document intentionally avoids sensitive details (no real IPs, usernames, private keys, or internal hostnames).

---

## 1) Static IP (Netplan)
**Location:** `/etc/netplan/01-static-ip.yaml`  
**Template in repo:** `server/network/01-netplan-static-ip.template.yaml`

Steps (run on server):
1. Identify active interface:
   - `ip a`
   - Example interface name: `eno1` or `enp0s31f6`
2. Copy template → netplan path and update:
   - IP address
   - gateway
   - DNS
3. Apply:
   - `sudo netplan generate`
   - `sudo netplan apply`
4. Verify:
   - `ip a`
   - `ip route`
   - `resolvectl status` (or check DNS resolution)

---

## 2) SSH Setup + Hardening
**Reference in repo:** `server/ssh/sshd_hardening.md`

Minimum requirements:
- SSH enabled
- Key-based auth for students
- Disable password auth after keys confirmed working

Verification:
- From an external machine: `ssh -i <keyfile> user@<server-ip>`

---

## 3) Firewall Rules
**Reference in repo:** `server/firewall/ufw_rules.md`

Minimum:
- allow SSH (port 22 or your custom port)
- deny everything else by default
- allow only what you need later (RDP, app ports, etc.)

---

## 4) External Connectivity Test
Test from outside the network:
- SSH works over the static IP (or public-facing NAT if used)
- Connection is stable and repeatable

Checklist:
- [ ] SSH key auth works
- [ ] Password auth disabled (after verification)
- [ ] Firewall active and minimal
- [ ] Static IP persists after reboot

---

## 5) Notes / Next Steps
- If public access is required, coordinate NAT/port forwarding and document it separately (without exposing sensitive values).
- Consider adding Fail2Ban and rate limiting if exposed publicly.