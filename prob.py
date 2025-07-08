```python
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st

# 환경: 5x5 그리드
class SimpleEnv:
    def __init__(self):
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.state = self.start
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 우, 좌, 하, 상

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        next_x, next_y = x + dx, y + dy

        if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
            self.state = (next_x, next_y)
            reward = 10 if self.state == self.goal else -1
            done = self.state == self.goal
        else:
            reward = -5
            done = False
        return self.state, reward, done

# 에이전트: Q-learning + 소프트맥스
class Agent:
    def __init__(self):
        self.q_table = np.zeros((5, 5, 4))
        self.temp = 1.0  # 소프트맥스 온도

    def softmax(self, q_values):
        exp_q = np.exp(q_values / self.temp)
        return exp_q / np.sum(exp_q)

    def choose_action(self, state):
        x, y = state
        probs = self.softmax(self.q_table[x, y])
        return np.random.choice(4, p=probs)

    def update(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        self.q_table[x, y, action] += 0.1 * (
            reward + 0.95 * np.max(self.q_table[next_x, next_y]) - self.q_table[x, y, action]
        )

# 경로 시각화
def plot_path(env, agent, episode):
    path = [env.start]
    state = env.reset()
    done = False
    for _ in range(20):
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    fig, ax = plt.subplots(figsize=(5, 5))
    grid = np.zeros((5, 5))
    grid[env.goal] = 2
    for x, y in path:
        if grid[x, y] == 0:
            grid[x, y] = 1

    ax.imshow(grid, cmap='hot')
    ax.plot([y for _, y in path], [x for x, _ in path], 'b-o')
    ax.set_title(f'에피소드 {episode}')
    st.pyplot(fig)
    plt.close(fig)

# 메인
def main():
    st.title("확률 기반 경로 최적화")
    env = SimpleEnv()
    agent = Agent()
    rewards = []

    for episode in range(30):
        state = env.reset()
        total_reward = 0
        for _ in range(20):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        if episode % 10 == 0:
            st.write(f"에피소드 {episode}, 보상: {total_reward}")
            plot_path(env, agent, episode)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rewards)
    ax.set_title('보상 그래프')
    st.pyplot(fig)
    plt.close(fig)

if __name__ == "__main__":
    main()
```

### 의존성 파일
Streamlit Cloud 배포를 위해 프로젝트 루트에 아래 파일 추가:

- **`requirements.txt`**:
  ```plaintext
  streamlit==1.39.0
  numpy==1.26.4
  matplotlib==3.8.3
  ```

- **`packages.txt`**:
  ```plaintext
  libfreetype6-dev
  pkg-config
  python3-distutils
  ```

### 수행

2. **GitHub 푸시**:
   ```bash
   git add prob.py requirements.txt packages.txt
   git commit -m "간단한 확률 최적화 코드"
   git push origin main
   ```

3. **Streamlit 재배포**:
   - Streamlit Cloud → **Manage App** → **Reboot**.

4. **로컬 테스트**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install streamlit numpy matplotlib
   streamlit run prob.py
   ```

### 확률 요소
- **소프트맥스 정책**: `Agent.softmax`에서 Q-값을 확률로 변환해 행동 선택. 높은 Q-값의 행동이 더 자주 선택되지만, 낮은 Q-값도 가끔 탐험.
- **랜덤 선택**: `np.random.choice`로 확률 분포 기반 행동 뽑기.

### 왜 간단해?
- `tensorflow` 없애고 Q-learning만 사용.
- 에피소드 30개, 스텝 20개로 학습 빠름.
- 최소한의 시각화로 Streamlit Cloud 부담 ↓.

문제 있으면 로그나 에러 바로 말해! 😄
