# VulnMitigator - RL Mitigation Planner (Defensive)

VulnMitigator is a **PyQt6 GUI tool** that uses **Q-learning** to recommend the optimal patching order for system vulnerabilities. Designed for **educational and testing purposes** in **cybersecurity** and **risk management**.

---

![ScreenShot](ScrenShot.png)

## Features

- Load vulnerabilities from a CSV file with columns:
  - `cve_id`
  - `cvss`
  - `exploit_likelihood`
  - `impact`
- Generate example vulnerabilities (up to 12 for tabular Q-learning)
- Train a **Q-learning agent** to recommend patching priorities
- Heuristic recommendations for larger vulnerability sets (>15)
- Simulate an episode of patching with cumulative expected risk reduction
- Save and load trained agents (`.joblib`)
- Apply patches to selected vulnerabilities
- Activity log displayed in the GUI

---

```bash

bash

pip install PyQt6 numpy pandas joblib
Ensure the icon file AI7.ico is in the same folder as the script.

Usage
Run the application:

bash

python VulnMitigator.py
In the GUI:

Load Vulnerabilities CSV: load your CSV file.

Generate Example: generate a test set of vulnerabilities.

Train RL Agent: train the Q-learning agent.

Save/Load Agent: save or load a trained agent.

Recommend Next Patch: get the next patch recommendation.

Apply Selected Patch: apply the patch to the selected vulnerability.

Simulate Episode: simulate a patching episode and see risk reduction.

Track activity via the log panel at the bottom-left.

Q-learning Agent
Tabular Q-learning:

States: bitmask of all vulnerabilities (0 = unpatched, 1 = patched)

Actions: index of unpatched vulnerabilities

Reward: negative expected risk of remaining unpatched vulnerabilities

Parameters:

learning_rate

discount (gamma)

epsilon (exploration rate)

episodes and patch budget per episode

Heuristic Recommendation
If the agent is not trained or dataset >15 items, a simple heuristic is used:

ini

score = exploit_likelihood * impact
Priority: patch the vulnerability with the highest score.

CSV Format Example
csv

cve_id,cvss,exploit_likelihood,impact
CVE-2025-1000,7.5,0.3,8
CVE-2025-1001,5.2,0.1,4
CVE-2025-1002,9.0,0.5,10
Notes
Tabular Q-learning works best with <=15 vulnerabilities. For larger sets, recommendations rely on heuristics.

Episode simulation shows step-by-step expected risk reduction.

GUI is styled with QSS (Qt Style Sheets) for a dark theme.

License
This project is open-source and intended for educational purposes only.
For commercial use, contact the author.
