import sys
import os
import math
import traceback
import random
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from PyQt6.QtGui import QIcon

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QMessageBox, QTextEdit,
    QProgressBar, QGroupBox, QFormLayout, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Q-learning engine (tabular)
class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, discount=0.99, epsilon=0.2):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = {}  # maps state(int bitmask) -> q-values array

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=float)
        return self.q_table[state]

    def act(self, state, available_actions=None):
        q = self.get_q(state)
        # mask unavailable actions
        if available_actions is not None:
            mask = np.full(self.n_actions, -np.inf)
            mask[available_actions] = 0.0
            q_eff = q + mask
        else:
            q_eff = q
        if random.random() < self.epsilon:
            # choose random available action
            if available_actions is None:
                return random.randrange(self.n_actions)
            return random.choice(list(available_actions))
        return int(np.nanargmax(q_eff))

    def learn(self, s, a, r, s2, available_actions_s2=None):
        q = self.get_q(s)
        q_next = self.get_q(s2)
        if available_actions_s2 is not None:
            # zero Q for unavailable actions
            mask = np.full(self.n_actions, -np.inf)
            mask[available_actions_s2] = 0.0
            q_next_eff = q_next + mask
            next_max = np.max(q_next_eff)
        else:
            next_max = np.max(q_next)
        td = r + self.gamma * next_max - q[a]
        q[a] += self.lr * td

    def save(self, path):
        joblib.dump(self.q_table, path)

    def load(self, path):
        self.q_table = joblib.load(path)

# Environment (simulated)
class PatchEnv:
    def __init__(self, vulnerabilities):
        # vulnerabilities: list of dicts with keys: cve_id, cvss, exploit_likelihood, impact
        self.vulns = vulnerabilities
        self.n = len(vulnerabilities)

    def initial_state(self):
        return 0  # bitmask all zero

    def available_actions(self, state):
        # actions are indices of unpatched vulnerabilities
        avail = [i for i in range(self.n) if not ((state >> i) & 1)]
        return avail

    def step(self, state, action):
        # apply patch action -> new state
        if (state >> action) & 1:
            # already patched
            return state, 0.0
        new_state = state | (1 << action)
        # compute reward: negative expected risk of remaining unpatched
        reward = -self.expected_risk(new_state)
        return new_state, reward

    def expected_risk(self, state):
        # sum over unpatched vulns: exploit_likelihood * impact
        risk = 0.0
        for i, v in enumerate(self.vulns):
            if not ((state >> i) & 1):
                risk += float(v.get('exploit_likelihood', 0.0)) * float(v.get('impact', 1.0))
        return float(risk)

# Trainer Thread
class RLTrainerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, env, episodes=2000, budget=None, lr=0.1, gamma=0.99, epsilon=0.2):
        super().__init__()
        self.env = env
        self.episodes = int(episodes)
        self.budget = budget if budget is not None else env.n
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self):
        try:
            n_actions = self.env.n
            agent = QLearningAgent(n_actions, learning_rate=self.lr, discount=self.gamma, epsilon=self.epsilon)
            for ep in range(1, self.episodes + 1):
                state = self.env.initial_state()
                total_reward = 0.0
                steps = 0
                while True:
                    avail = self.env.available_actions(state)
                    if not avail or steps >= self.budget:
                        break
                    action = agent.act(state, available_actions=avail)
                    next_state, r = self.env.step(state, action)
                    total_reward += r
                    steps += 1
                    # next available actions
                    avail2 = self.env.available_actions(next_state)
                    agent.learn(state, action, r, next_state, available_actions_s2=avail2)
                    state = next_state
                if ep % max(1, self.episodes // 20) == 0:
                    pct = int(ep * 100 / self.episodes)
                    self.progress.emit(pct)
            self.progress.emit(100)
            self.finished.emit(agent)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)

# QSS (basic)
QSS = r"""
QMainWindow { background: #061226; color: #e6eef8; }
QLabel { color: #e6eef8; }
QPushButton { background: #0f1724; color: #e6eef8; padding: 8px; border-radius: 6px; }
QTableWidget { background: #071029; color: #e6eef8; }
QProgressBar { background: #03101a; color: #e6eef8; border-radius: 6px; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('VulnMitigator - RL Mitigation Planner (Defensive)')
        self.setWindowIcon(QIcon("AI7.ico"))
        self.resize(1100, 700)
        self.vulns = []
        self.env = None
        self.agent = None
        self.training_thread = None

        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(); central.setLayout(main)

        # left: controls
        left = QVBoxLayout(); main.addLayout(left, 1)
        btn_load = QPushButton('Load Vulnerabilities CSV')
        btn_load.clicked.connect(self.load_csv)
        btn_gen = QPushButton('Generate Example (<=12)')
        btn_gen.clicked.connect(self.generate_example)
        btn_train = QPushButton('Train RL Agent')
        btn_train.clicked.connect(self.train_agent)
        btn_save_agent = QPushButton('Save Agent (.joblib)')
        btn_save_agent.clicked.connect(self.save_agent)
        btn_load_agent = QPushButton('Load Agent (.joblib)')
        btn_load_agent.clicked.connect(self.load_agent)
        left.addWidget(btn_load); left.addWidget(btn_gen); left.addWidget(btn_train); left.addWidget(btn_save_agent); left.addWidget(btn_load_agent)

        # training config
        cfg = QGroupBox('Training config')
        form = QFormLayout()
        self.spin_episodes = QSpinBox(); self.spin_episodes.setRange(100,20000); self.spin_episodes.setValue(2000)
        self.spin_budget = QSpinBox(); self.spin_budget.setRange(1,50); self.spin_budget.setValue(5)
        self.in_lr = QLineEdit('0.1')
        self.in_gamma = QLineEdit('0.99')
        self.in_eps = QLineEdit('0.2')
        form.addRow('episodes', self.spin_episodes); form.addRow('patch budget per episode', self.spin_budget); form.addRow('learning_rate', self.in_lr); form.addRow('discount (gamma)', self.in_gamma); form.addRow('epsilon', self.in_eps)
        cfg.setLayout(form)
        left.addWidget(cfg)

        self.progress = QProgressBar(); left.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(180); left.addWidget(self.log)

        # right: table and actions
        right = QVBoxLayout(); main.addLayout(right, 3)
        self.table = QTableWidget(); self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['idx','cve_id','cvss','exploit_likelihood','impact','patched'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right.addWidget(self.table)

        actions = QHBoxLayout();
        self.btn_recommend = QPushButton('Recommend Next Patch'); self.btn_recommend.clicked.connect(self.recommend_next)
        self.btn_apply = QPushButton('Apply Selected Patch'); self.btn_apply.clicked.connect(self.apply_selected)
        self.btn_simulate = QPushButton('Simulate Episode'); self.btn_simulate.clicked.connect(self.simulate_episode)
        actions.addWidget(self.btn_recommend); actions.addWidget(self.btn_apply); actions.addWidget(self.btn_simulate)
        right.addLayout(actions)

        self.status_label = QLabel('Agent: not trained')
        right.addWidget(self.status_label)
        self.setStyleSheet(QSS)

    # Logging
    def log_msg(self, s):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log.append(f'[{ts}] {s}')

    # Data loading / generation
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open vulnerabilities CSV', os.getcwd(), 'CSV Files (*.csv)')
        if not path:
            return
        try:
            df = pd.read_csv(path)
            # expect columns or try to map
            required = ['cve_id','cvss','exploit_likelihood','impact']
            for c in required:
                if c not in df.columns:
                    QMessageBox.warning(self, 'Invalid CSV', f'Missing column: {c}. CSV must contain cve_id, cvss, exploit_likelihood, impact')
                    return
            self.vulns = df.to_dict(orient='records')
            if len(self.vulns) > 15:
                QMessageBox.information(self, 'Large set', 'Loaded more than 15 vulnerabilities. RL uses tabular Q-learning and performs best with <=15 items; for larger sets the app will use heuristic recommendations.')
            self.env = PatchEnv(self.vulns)
            self.populate_table()
            self.log_msg(f'Loaded {len(self.vulns)} vulnerabilities from {os.path.basename(path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))
            self.log_msg('Load CSV error: ' + str(e))

    def generate_example(self):
        n = 8
        rows = []
        for i in range(n):
            cvss = round(max(0.1, min(10, random.gauss(7,1.5))),1)
            exploit = round(min(1.0, max(0.0, random.random() * (cvss/10.0 + 0.2))),3)
            impact = round(random.randint(1,10),1)
            rows.append({'cve_id': f'CVE-2025-{1000+i}','cvss': cvss,'exploit_likelihood': exploit,'impact': impact})
        self.vulns = rows
        self.env = PatchEnv(self.vulns)
        self.populate_table()
        self.log_msg('Generated example vulnerabilities (n=%d)'%len(self.vulns))

    def populate_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.vulns))
        for r, v in enumerate(self.vulns):
            self.table.setItem(r, 0, QTableWidgetItem(str(r)))
            self.table.setItem(r, 1, QTableWidgetItem(str(v.get('cve_id',''))))
            self.table.setItem(r, 2, QTableWidgetItem(str(v.get('cvss',''))))
            self.table.setItem(r, 3, QTableWidgetItem(str(v.get('exploit_likelihood',''))))
            self.table.setItem(r, 4, QTableWidgetItem(str(v.get('impact',''))))
            patched = '0'
            self.table.setItem(r, 5, QTableWidgetItem(patched))
        self.table.blockSignals(False)

    # Training
    def train_agent(self):
        if not self.env:
            QMessageBox.warning(self, 'No data', 'Load or generate vulnerabilities first')
            return
        try:
            episodes = int(self.spin_episodes.value())
            budget = int(self.spin_budget.value())
            lr = float(self.in_lr.text())
            gamma = float(self.in_gamma.text())
            eps = float(self.in_eps.text())
            self.training_thread = RLTrainerThread(self.env, episodes=episodes, budget=budget, lr=lr, gamma=gamma, epsilon=eps)
            self.training_thread.progress.connect(self.progress.setValue)
            self.training_thread.finished.connect(self._on_trained)
            self.training_thread.error.connect(self._on_train_error)
            self.training_thread.start()
            self.log_msg('Training RL agent in background...')
        except Exception as e:
            QMessageBox.critical(self, 'Train error', str(e))
            self.log_msg('Train error: ' + str(e))

    def _on_trained(self, agent):
        self.agent = agent
        self.progress.setValue(0)
        self.status_label.setText('Agent trained (Q-table ready)')
        self.log_msg('RL agent training finished')

    def _on_train_error(self, tb):
        QMessageBox.critical(self, 'Training failed', 'See log')
        self.log_msg('Training error:\n' + tb)

    # Agent persistence
    def save_agent(self):
        if not self.agent:
            QMessageBox.warning(self, 'No agent', 'Train an agent first')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save agent', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        try:
            joblib.dump(self.agent.q_table, path)
            self.log_msg('Agent Q-table saved to %s' % path)
        except Exception as e:
            QMessageBox.critical(self, 'Save error', str(e))

    def load_agent(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load agent', os.getcwd(), 'Joblib Files (*.joblib)')
        if not path:
            return
        try:
            qtable = joblib.load(path)
            n_actions = len(self.vulns) if self.vulns else max(1, max((len(v) for v in qtable.keys()), default=1))
            agent = QLearningAgent(n_actions)
            agent.q_table = qtable
            self.agent = agent
            self.status_label.setText('Agent loaded')
            self.log_msg('Agent loaded from %s' % path)
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))
            self.log_msg('Load agent failed: ' + str(e))

    # Recommendation & application
    def recommend_next(self):
        if not self.env:
            QMessageBox.warning(self, 'No data', 'Load vulnerabilities first')
            return
        if not self.agent:
            QMessageBox.information(self, 'No agent', 'Agent not trained - showing heuristic recommendation')
            # heuristic: exploit_likelihood * impact
            scores = [v['exploit_likelihood'] * v['impact'] for v in self.vulns]
            idx = int(np.argmax(scores))
            QMessageBox.information(self, 'Recommendation', f'Heuristic suggests patch CVE index {idx} ({self.vulns[idx]["cve_id"]})')
            return
        # compute current state from table patched column
        state = 0
        for r in range(self.table.rowCount()):
            try:
                patched = int(self.table.item(r,5).text())
                if patched:
                    state |= (1 << r)
            except Exception:
                pass
        avail = self.env.available_actions(state)
        if not avail:
            QMessageBox.information(self, 'All patched', 'All vulnerabilities are patched')
            return
        action = self.agent.act(state, available_actions=avail)
        QMessageBox.information(self, 'Recommendation', f'RL agent recommends patch index {action} ({self.vulns[action]["cve_id"]})')

    def apply_selected(self):
        cur = self.table.currentRow()
        if cur < 0:
            QMessageBox.information(self, 'Select row', 'Select a row to apply patch')
            return
        # mark patched
        self.table.setItem(cur,5, QTableWidgetItem('1'))
        self.log_msg(f'Applied patch to index {cur} ({self.vulns[cur]["cve_id"]})')

    def simulate_episode(self):
        if not self.env:
            QMessageBox.warning(self, 'No data', 'Load vulnerabilities first')
            return
        # perform simulated episode using agent or heuristic and show cumulative expected risk reduction
        state = self.env.initial_state()
        budget = int(self.spin_budget.value())
        steps = 0
        trace = []
        initial_risk = self.env.expected_risk(state)
        while steps < budget:
            avail = self.env.available_actions(state)
            if not avail:
                break
            if self.agent:
                a = self.agent.act(state, available_actions=avail)
            else:
                scores = [self.vulns[i]['exploit_likelihood'] * self.vulns[i]['impact'] for i in avail]
                a = avail[int(np.argmax(scores))]
            new_state, r = self.env.step(state, a)
            trace.append({'step': steps+1, 'action': a, 'cve': self.vulns[a]['cve_id'], 'expected_risk_after': self.env.expected_risk(new_state)})
            state = new_state
            steps += 1
        final_risk = self.env.expected_risk(state)
        reduction = initial_risk - final_risk
        # display
        s = f'Simulation finished. Initial risk: {initial_risk:.3f}, Final risk: {final_risk:.3f}, Reduction: {reduction:.3f}\nTrace:\n'
        for t in trace:
            s += f"Step {t['step']}: patched {t['cve']} -> risk after: {t['expected_risk_after']:.3f}\n"
        QMessageBox.information(self, 'Simulation', s)
        self.log_msg('Simulation run complete')

# Main entrypoint
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())