# AgentOS-Lite Shell

An interactive, Unix-style REPL over the kernel's HTTP API — the primary demo
surface. Every command maps onto an existing endpoint.

## Running

Start the backend first (from the project root, venv active):

```bash
uvicorn api.main:app --port 8000
```

Then launch the shell:

```bash
python shell/repl.py                                  # http://localhost:8000 as root
python shell/repl.py --url http://localhost:8000      # explicit URL
python shell/repl.py --agent alice                    # act as a USER-level agent
```

- `--url` sets the backend base URL (default `http://localhost:8000`).
- `--agent` sets the identity the shell acts as (default `root`, which is
  KERNEL-privileged). Run as another agent to demo permission differences.

## Commands

| Command            | Endpoint                          | Description                              |
|--------------------|-----------------------------------|------------------------------------------|
| `ps`               | `GET /scheduler/state`            | process table (PID/STATE/ARRIVAL/…)      |
| `top`              | `GET /scheduler/state` + `/resources/state` | auto-refresh every 2s (Ctrl+C to stop) |
| `kill [-t] <pid>`  | `POST /scheduler/terminate/{pid}` | terminate a process (`-t` = whole subtree)|
| `limits [agent]`   | `GET /quotas/{agent}`             | quota usage vs limit (pages + call rate) |
| `ls [agent]`       | `GET /fs/list/{agent}`            | list an agent's files                    |
| `cat <filename>`   | `GET /fs/read`                    | print a file's contents                  |
| `find <query>`     | `POST /fs/search`                 | natural-language semantic file search    |
| `mem <agent>`      | `GET /memory/state/{agent}`       | RAM pages vs pages swapped to ChromaDB   |
| `strace [n]`       | `GET /syscalls/log?limit=n`       | recent syscalls (default 20)             |
| `pstree`           | `GET /scheduler/tree`             | process hierarchy as an ASCII tree       |
| `spawn [pid]`      | `POST /scheduler/spawn`           | fork a child process                     |
| `wait <p> [child]` | `POST /scheduler/wait/{pid}`      | reap a zombie child, read its exit status|
| `run <prompt>`     | `POST /generate`                  | issue an LLM_CALL, shows serving driver  |
| `help`             | —                                 | list commands                            |
| `exit` / `quit`    | —                                 | leave the shell                          |

Commands that default to an agent (`limits`, `ls`) use the shell's `--agent`
identity when no argument is given.

## Example session

```
$ python shell/repl.py
AgentOS-Lite shell  —  http://localhost:8000  (agent: root)
type 'help' for commands, 'exit' to quit.

aios:root$ ps
PID  STATE       ARRIVAL  REMAINING  PRIO
---  ----------  -------  ---------  ----
P1   terminated  0        0          1
P2   running     1        1          2
P3   ready       2        8          0
P4   waiting     3        2          1

aios:root$ kill P2
killed 'P2'

aios:root$ limits demo
quotas for 'demo':
RESOURCE          USED  LIMIT
----------------  ----  -----
memory pages      7     20
LLM calls / 60s   0     10

aios:root$ find how do plants make energy
FILE         SCORE  SNIPPET
-----------  -----  ------------------------------------------
note_a.txt   0.477  Plants perform photosynthesis, converting …

aios:root$ run say hello in one word
[groq] Hello.

aios:root$ strace 5
TIME      AGENT   SYSCALL     STATUS   LATENCY
--------  ------  ----------  -------  -------
12:01:07  root    LLM_CALL    success  812.4ms
...
```

## Permission demo

Run the shell as a USER-level agent and watch privileged operations get
rejected cleanly:

```
$ python shell/repl.py --agent mallory
aios:mallory$ kill P1
permission denied: USER-level agent 'mallory' may not terminate process 'P1' (requires KERNEL privilege)
```

Errors are always printed as a clean one-liner — `403 → permission denied`,
`429 → quota exceeded`, `404 → not found` — never a raw traceback.
```

