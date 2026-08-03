# SSH Hardening Notes (Ticket #9)

## Recommended baseline
- Use SSH keys (required)
- Disable password auth after verifying keys work
- Optional: change SSH port only if your org wants it (security-by-obscurity is not a replacement)

---

## Key-based login
1. Create user (if needed):
   - `sudo adduser <student_user>`
2. Add their public key:
   - `sudo mkdir -p /home/<student_user>/.ssh`
   - `sudo nano /home/<student_user>/.ssh/authorized_keys`
   - `sudo chmod 700 /home/<student_user>/.ssh`
   - `sudo chmod 600 /home/<student_user>/.ssh/authorized_keys`
   - `sudo chown -R <student_user>:<student_user> /home/<student_user>/.ssh`

---

## sshd_config changes (example)
File: `/etc/ssh/sshd_config`

Recommended:
- `PubkeyAuthentication yes`
- `PasswordAuthentication no`  (after confirming key login works)
- `PermitRootLogin no`
- `X11Forwarding no` (unless needed)
- `AllowUsers <student_user> <admin_user>` (optional allowlist)

Apply:
- `sudo systemctl restart ssh`
- `sudo systemctl status ssh`

---

## Validate from external machine
- `ssh -i <keyfile> <student_user>@<server-ip>`