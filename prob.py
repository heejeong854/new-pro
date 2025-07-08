```python
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st

class LogisticsEnv:
    def __init__(self):
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.state = self.start
        self.obstacles = [(1, 1), (2, 2), (3, 3)]
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 좌, 우, 상, 하

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        next_x, next_y = x + dx, y + dy

        if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
            next_state = (next_x, next_y)
        else:
            return self.state, -5, False

        reward = -1
        done = False

        if next_state in self.obstacles:
            reward = -10
            next_state = self.state
        elif next_state == self.goal:
            reward = 100
            done = True

        self.state = next_state
        return next_state, reward, done

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((5, 5, 4))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        x, y = state
        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        self.q_table[x, y, action] += 0.1 * (
            reward + 0.95 * np.max(self.q_table[next_x, next_y]) - self.q_table[x, y, action]
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def plot_path(env, agent, episode):
    path = [env.start]
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state

    fig, ax = plt.subplots(figsize=(5, 5))
    grid = np.zeros((5, 5))
    for obs in env.obstacles:
        grid[obs] = -1
    grid[env.goal] = 2
    for x, y in path:
        if grid[x, y] == 0:
            grid[x, y] = 1

    ax.imshow(grid, cmap='hot')
    ax.plot([y for _, y in path], [x for x, _ in path], 'b-o')
    ax.set_title(f'에피소드 {episode}')
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.title("물류 경로 최적화")
    env = LogisticsEnv()
    agent = QLearningAgent()
    rewards = []

    for episode in range(50):
        state = env.reset()
        total_reward = 0
        for _ in range(50):
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

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards)
    ax.set_title('보상 그래프')
    ax.set_xlabel('에피소드')
    ax.set_ylabel('보상')
    st.pyplot(fig)
    plt.close(fig)

if __name__ == "__main__":
    main()
```

### 의존성 파일
Streamlit Cloud 배포를 위해 루트 디렉토리에 아래 파일 추가:

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
1. **코드 저장**:
   - `prob.py`를 위 코드로 덮어쓰기.
   - 또는 177번째 줄 삭제 후 코드 확인.

2. **파일 푸시**:
   ```bash
   git add prob.py requirements.txt packages.txt
   git commit -m "SyntaxError 수정"
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

### 추가
- **로그 확인**: 배포 실패 시 **Manage App** → **Logs** 공유.
- **코드 다른 점**: `prob.py`가 위와 다르면 내용 공유해 주세요.

이렇게 하면 오류 해결되고 배포 잘 될 거예요! 문제 있으면 바로 말해.
