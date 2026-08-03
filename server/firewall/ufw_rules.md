# Firewall Rules (UFW) - Ticket #9

## Baseline setup
1) Enable UFW:
- `sudo ufw default deny incoming`
- `sudo ufw default allow outgoing`

2) Allow SSH:
- `sudo ufw allow 22/tcp`
  - If using a custom SSH port: replace `22`

3) Enable:
- `sudo ufw enable`
- `sudo ufw status verbose`

## Notes
- Keep rules minimal.
- Only open additional ports if required (document the reason and port).