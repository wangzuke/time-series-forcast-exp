---
name: codex-remote-network-fix
description: Diagnose and fix Codex CLI or Codex App remote Connection failures on SSH servers caused by blocked OpenAI/ChatGPT access, DNS pollution, missing SSH RemoteForward proxy tunnels, stale Codex versions, broken remote codex wrappers, or PATH issues. Use when a user wants Codex App to work on another remote server through an SSH connection, or when remote codex works locally only after manual proxy/ssh commands.
---

# Codex Remote Network Fix

## Overview

Use this skill to make a remote SSH server usable from Codex App or Codex CLI when the server cannot directly reach OpenAI/ChatGPT endpoints. Prefer a host-specific SSH `RemoteForward` plus a remote `codex` wrapper that injects proxy env vars only for Codex, leaving the rest of the server environment unchanged.

## Workflow

1. Identify the SSH host alias the user uses in Codex App Connection, such as `dw_root_2040`.
2. Check local proxy availability. Probe common local ports (`7897`, `7890`, `7891`, `1080`, `10808`, `8080`) with `curl -x http://127.0.0.1:<port> https://api.openai.com/v1/models`; a `401` response is a good unauthenticated OpenAI reachability signal.
3. Check remote direct reachability and DNS:
   - `getent ahostsv4 api.openai.com`, `getent ahostsv6 api.openai.com`
   - `curl -I -L --connect-timeout 8 --max-time 20 https://api.openai.com/v1/models`
   - `curl -I -L --connect-timeout 8 --max-time 20 https://chatgpt.com`
   Treat suspicious Meta/Facebook-looking `api.openai.com` or `chatgpt.com` answers, IPv6-only failures, or 443 timeouts as DNS/direct-network problems.
4. Add `RemoteForward 127.0.0.1:<remote_port> 127.0.0.1:<local_port>` to the same local `~/.ssh/config` `Host` block that Codex App uses. Also add `ServerAliveInterval 30` and `ServerAliveCountMax 3`. Back up `~/.ssh/config` first.
5. Restart the Codex App connection so it reopens SSH and receives the reverse tunnel. For manual verification, run `ssh -fN <host>` or open a normal SSH session, then test on the server with `curl -x http://127.0.0.1:<remote_port> https://api.openai.com/v1/models`.
6. On the server, ensure the login shell can find a Codex wrapper before the npm shim:
   - prepend `/root/.local/bin:/root/.npm-global/bin` or the relevant user paths to `.profile` and `.bashrc`
   - create/update `~/.local/bin/codex` to export `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`, then exec the real Codex binary
7. Upgrade or reinstall Codex when the model requires a newer CLI, optional platform packages are missing, or `codex --version` fails. Use a Node >=16 runtime for npm installs when possible; older system Node may still run neither npm nor the JS shim correctly.
8. Validate with:
   - `bash -lc 'which codex; codex --version'` over SSH
   - `codex doctor` as a diagnostic signal, not the sole source of truth
   - `cd /tmp && codex exec --skip-git-repo-check --sandbox read-only --ephemeral '只回复 OK，不要其他内容'`

## Script

Use `scripts/remote_codex_network_fix.sh` for repeatable checks and remote setup.

Typical dry run:

```bash
scripts/remote_codex_network_fix.sh dw_root_2040 --local-port 7897 --remote-port 7897
```

Apply remote Codex wrapper and PATH changes after SSH `RemoteForward` is configured:

```bash
scripts/remote_codex_network_fix.sh dw_root_2040 --local-port 7897 --remote-port 7897 --apply-remote
```

The script does not edit local `~/.ssh/config`; use `apply_patch` for that edit so the exact Host block remains reviewable.

## Notes

- Codex App remote Connection uses SSH and the remote user's login environment, so fix the exact SSH host alias and remote user Codex App uses.
- Prefer a Codex-only wrapper over global proxy exports. Global proxy variables can break package managers, Git, apt, or internal services that already have direct network access.
- If `codex exec` succeeds but `codex doctor` still reports ChatGPT HTTP/WebSocket warnings, record the warning but prioritize the real request result. Some Cloudflare-protected ChatGPT probes can fail while Codex HTTPS fallback still works.
- If a remote port is already occupied, a second SSH session may warn `remote port forwarding failed`; verify whether an existing tunnel is already serving that port before changing ports.
