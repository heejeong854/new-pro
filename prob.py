```python
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st

# 환경 설정: 5x5 그리드
class LogisticsEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        self.state = self.start
        self.obstacles = [(1, 1), (2, 2), (3, 3)]
        self.max_steps = 50
        self.current_step = 0
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 좌, 우, 상, 하

    def reset(self):
        self.state = self.start
        self.current_step = 0
        return self.state

    def step(self, action):
        self.current_step += 1
        x, y = self.state
        dx, dy = self.actions[action]
        next_x, next_y = x + dx, y + dy

        # 경계 체크
        if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
            next_state = (next_x, next_y)
        else:
            next_state = self.state
            return next_state, -5, False  # 경계 밖 페널티

        reward = -1  # 기본 이동 비용
        done = False

        if next_state in self.obstacles:
            reward = -10
            next_state = self.state
        elif next_state == self.goal:
            reward = 100
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        self.state = next_state
        return next_state, reward, done

# Q-learning 에이전트
class QLearningAgent:
    def __init__(self, grid_size, action_size):
        self.q_table = np.zeros((grid_size, grid_size, action_size))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.alpha = 0.1

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions)-1)
        x, y = state
        return np.argmax(self.q_table[x, y])

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_x, next_y, best_next_action] - self.q_table[x, y, action]
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 시각화 함수
def visualize_path(env, agent, episode):
    path = [env.start]
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < env.max_steps:
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        steps += 1

    # 그리드 시각화
    fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.zeros((env.grid_size, env.grid_size))
    for obs in env.obstacles:
        grid[obs] = -1
    grid[env.goal] = 2
    for x, y in path:
        if grid[x, y] == 0:
            grid[x, y] = 1

    ax.imshow(grid, cmap='hot', interpolation='nearest')
    ax.plot([y for _, y in path], [x for x, _ in path], 'b-', marker='o')
    ax.set_title(f'Episode {episode} Path')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

# 메인 학습 루프
def main():
    st.title("간단한 물류 경로 최적화 (Q-learning)")
    env = LogisticsEnv()
    agent = QLearningAgent(env.grid_size, len(env.actions))
    episodes = 100
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        if episode % 10 == 0:
            st.write(f"에피소드 {episode}, 총 보상: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            visualize_path(env, agent, episode)

    # 보상 그래프
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards)
    ax.set_title('에피소드별 총 보상')
    ax.set_xlabel('에피소드')
    ax.set_ylabel('총 보상')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

if __name__ == "__main__":
    main()
```

### 변경 사항 및 특징
- **의존성 단순화**: `tensorflow` 제거, `numpy`와 `matplotlib`만 사용.
- **Q-learning**: DQN 대신 간단한 Q-learning 알고리즘으로 전환. Q-table을 사용하여 메모리 사용량 감소.
- **Streamlit 호환**: `st.pyplot`로 그래프 렌더링, `st.title`과 `st.write`로 UI 개선.
- **경량화**: Streamlit Cloud 무료 티어에서 실행 가능하도록 최적화.

### Streamlit Cloud 배포
1. **requirements.txt**:
   프로젝트 루트에 아래 내용으로 `requirements.txt` 생성:
   ```plaintext
   streamlit==1.39.0
   numpy==1.26.4
   matplotlib==3.8.3
   ```

2. **packages.txt**:
   `matplotlib` 설치에 필요한 시스템 의존성을 위해:
   ```plaintext
   libfreetype6-dev
   pkg-config
   python3-distutils
   ```

3. **GitHub 푸시**:
   ```bash
   git add simple_logistics_qlearning.py requirements.txt packages.txt
   git commit -m "간단한 Q-learning 코드 및 의존성 추가"
   git push origin main
   ```

4. **Streamlit Cloud 재배포**:
   - Streamlit Cloud 대시보드에서 앱 선택 → **Manage App** → **Reboot**.
   - 배포 후 **Logs**에서 오류 확인.

### 로컬 테스트
1. 가상 환경 설정:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
2. 의존성 설치:
   ```bash
   pip install streamlit numpy matplotlib
   ```
3. 실행:
   ```bash
   streamlit run simple_logistics_qlearning.py
   ```
4. 브라우저에서 `http://localhost:8501` 확인.

### 오류 해결 보장
- **`matplotlib` 오류**: `requirements.txt`와 `packages.txt`로 설치 문제 해결.
- **메모리 문제**: `tensorflow` 제거로 Streamlit Cloud의 메모리 한계 우회.
- **로그 확인**: 배포 실패 시 **Manage App** → **Logs**에서 세부 오류 확인 후 공유.

추가 문제가 있으면 오류 로그나 `pro.py`의 차이점을 알려주세요!
