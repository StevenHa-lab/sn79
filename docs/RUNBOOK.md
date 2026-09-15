# SN79 miner runbook

How to bring up one or more SN79 (taos.im) miners from this repository on a fresh Ubuntu box.
Every command below was run on 64.176.218.105 (Ubuntu 24.04, 4 vCPU, 7 GB RAM) on 2026-09-14/15.

## 0. What you are deploying

- `taos/im/neurons/miner.py` serves a Bittensor axon. The validator (one on mainnet today,
  hotkey `5EWwd...`) pushes one simulation state per simulated second, about every 5 to 9
  wall-clock seconds, and expects a reply within 3 s.
- The agent that decides what to do lives in `taos/agent/`. `MinerAgent_V7` is the current one:
  a two-sided passive market maker on all 128 books with FIFO accounting, budget pacing, trend
  lean, frozen-book control and on-disk state. `MinerAgent_V6` is the simpler predecessor.
- One miner process per hotkey. Several miners can share one box and one checkout; they differ
  only in hotkey, axon port and pm2 name.
- Scoring, the cap on daily volume (500k quote per book), and the dashboard are described in
  the repo discussion; nothing in this runbook depends on them.

## 1. Prerequisites

| Item | Requirement |
|---|---|
| OS | Ubuntu 22.04 or 24.04, root or sudo |
| Size | 2 to 4 vCPU, 4 GB RAM and 20 GB disk per box are enough for several miners (each uses ~200 MB RAM, <5% CPU) |
| Network | A public IPv4 the validator can reach, inbound TCP open on the axon port(s) |
| Wallet | A coldkey (public part is enough for mining) and one hotkey per miner, **registered on netuid 79** |
| Python | 3.10.9 exactly (repo pins it in `.python-version`) |

## 2. System packages and Python 3.10.9

```bash
apt-get update
apt-get install -y build-essential git curl htop tmux \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
  liblzma-dev libncursesw5-dev tk-dev uuid-dev xz-utils

curl -fsSL https://pyenv.run | bash
cat >> ~/.bashrc <<'B'
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
B
source ~/.bashrc
pyenv install -s 3.10.9
pyenv global 3.10.9
python --version   # Python 3.10.9
```

## 3. Repository and Python packages

```bash
cd /root
git clone git@github.com:StevenHa-lab/sn79.git      # or https://github.com/StevenHa-lab/sn79
cd sn79
python -m pip install -U pip wheel setuptools pyopenssl cryptography
# CPU-only torch saves ~3 GB of CUDA libraries; the miner does not need a GPU
python -m pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
mkdir -p ~/.taos && cp -r agents ~/.taos/agents
python -c "import taos, bittensor; print(taos.__version__, bittensor.__version__)"   # 0.6.0 10.5.x
python -c "from taos.agent.MinerAgent_V7 import MinerAgent_V7; print('agent ok')"
```

Keep `upstream` handy for future syncs; the fork already carries upstream 0.6.0:

```bash
git remote add upstream https://github.com/taos-im/sn-79
```

## 4. Node and pm2

```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"
nvm install --lts
npm install --location=global pm2
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 100M
pm2 set pm2-logrotate:compress true
```

## 5. Wallet and registration

Either copy an existing wallet from another box:

```bash
mkdir -p ~/.bittensor/wallets/<coldkey>/hotkeys
# copy coldkeypub.txt and hotkeys/<hotkey> (+ <hotkey>pub.txt) from the old box, then:
chmod 600 ~/.bittensor/wallets/<coldkey>/hotkeys/*
```

or create and register a new hotkey (btcli in its own venv keeps it off the miner's packages):

```bash
python -m venv ~/btcli-venv && ~/btcli-venv/bin/pip install -U bittensor-cli
ln -sf ~/btcli-venv/bin/btcli /usr/local/bin/btcli
btcli wallet new_hotkey --wallet.name <coldkey> --wallet.hotkey <hotkey>
btcli subnet register --netuid 79 --wallet.name <coldkey> --wallet.hotkey <hotkey> --network finney
```

Verify before starting anything (prints the UID):

```bash
python - <<'PY'
import bittensor as bt
w = bt.Wallet(name="<coldkey>", hotkey="<hotkey>", path="~/.bittensor/wallets/")
st = bt.Subtensor(network="finney"); hk = w.hotkey.ss58_address
print(hk, "registered:", st.is_hotkey_registered_on_subnet(hk, 79), "uid:", st.get_uid_for_hotkey_on_subnet(hk, 79))
PY
```

## 6. Firewall

```bash
ufw allow 22/tcp
ufw allow 7900:7999/tcp     # one port per miner from this range
ufw --force enable
ufw status
```

If the provider has a cloud firewall (Vultr, Hetzner, AWS), open the same ports there too.
A miner whose port is closed logs normally but never receives a state: the symptom is a log
with only the startup lines and no `Decompressed` lines.

## 7. Start a miner

Do **not** use `run_miner.sh` when more than one miner runs on the box: it hardcodes the pm2
name `miner7` and deletes it first. Start pm2 directly:

```bash
export PYENV_ROOT="$HOME/.pyenv"; export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"; eval "$(pyenv init -)"
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"
export BT_NO_PARSE_CLI_ARGS=false
cd /root/sn79/taos/im/neurons

pm2 start miner.py --name miner-v7-<hotkey> --interpreter python --cwd /root/sn79/taos/im/neurons -- \
  --netuid 79 --subtensor.chain_endpoint wss://entrypoint-finney.opentensor.ai:443 \
  --wallet.path /root/.bittensor/wallets/ --wallet.name <coldkey> --wallet.hotkey <hotkey> \
  --axon.port 7901 --logging.info \
  --agent.path /root/sn79/taos/agent --agent.name MinerAgent_V7 \
  --agent.params lazy_load=1 size=1.0
pm2 save
pm2 startup     # prints a command to run once so pm2 resurrects after a reboot
```

Notes:
- `pm2` must be started from a shell where `python` resolves to the pyenv 3.10.9. pm2 records
  `PATH` at start; `pm2 restart --update-env` from a shell without pyenv on `PATH` breaks the app.
- Agent parameters go after `--agent.params` as `key=value` pairs; every V7 parameter and its
  default is listed in the docstring at the top of `taos/agent/MinerAgent_V7.py`.
- Repeat with another hotkey, port and pm2 name for each additional miner.

## 8. Verify

```bash
pm2 ls
ss -ltnp | grep 790                       # the port is listening
pm2 logs miner-v7-<hotkey> --nostream --lines 80 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "Serving miner axon|Miner starting|Decompressed|V7 \["
```

Expected within two minutes:

- `Serving miner axon at <ip>:<port>` and `Miner starting at block ... with UID <n>`.
- A `Decompressed (...)` line every 5 to 9 s: states are arriving.
- Every 30 states, one summary line beginning `V7 [<validator>]` with fills, wins, losses,
  realized P&L, inventory, 24 h volume, frozen books, and the ledger reconciliation counters.

Validator-side view of the same UID, no login needed (the User-Agent header is required):

```bash
curl -s -A curl/8.5.0 'https://taos.simulate.trading/api/datasources/proxy/uid/c7665619-8b04-4227-9882-a44e8dad57a6/api/v1/query' \
  --data-urlencode 'query=miners{netuid="79",agent_id="<uid>"}' | python3 -m json.tool | grep -E '"(placement|score|kappa|kappa_score|pnl_score|total_daily_volume|total_realized_pnl|activity_factor)"'
```

## 9. Operate

| Task | Command |
|---|---|
| Restart, same parameters | `pm2 restart miner-v7-<hotkey>` (from a pyenv shell) |
| Change parameters | `pm2 delete miner-v7-<hotkey>` then the `pm2 start` line from step 7 |
| Follow the log | `pm2 logs miner-v7-<hotkey>` |
| Update the code | `cd /root/sn79 && git pull && python -m pip install -e . && pm2 restart miner-v7-<hotkey>` |
| Sync upstream | `git fetch upstream && git merge upstream/main`, resolve, `pip install -e .`, restart |
| Agent state | `taos/im/neurons/agent_state/v7_state_<uid>.json`; delete it to start the ledger fresh (V7 rebuilds inventory from the account anyway) |
| Stop | `pm2 delete miner-v7-<hotkey> && pm2 save` |

## 10. Known messages and issues

- `SETTLEMENT NOT READY ... btcli proxy add` at startup: the 0.6.0 exchange mechanism, which is
  not live on mainnet. Ignore it. Do not grant the proxy.
- Two tracebacks (`logging ... EOFError`, `metagraph_worker ... KeyboardInterrupt`) right after
  a restart are shutdown noise from the previous process.
- `timeouts` on the dashboard above zero: the miner answered later than 3 s. Keep
  `lazy_load=1`, keep the box near the validator (Europe), avoid other CPU load.
- Kappa `None` for hours after registration while other new UIDs get one: under
  investigation on 2026-09-15; the working hypothesis is a minimum trade size on the live
  validator, being tested with `size=1.0` and `size=3.0`.
- A new UID is not scored for the first 1.5 simulated hours, and its weight follows a 3 hour
  EMA after that; expect about one simulated day (2 to 3 real days) before the rank settles.
