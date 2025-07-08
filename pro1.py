```python
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# 환경
class Env:
    def __init__(self):
        self.size = 5
        self.state = (0, 0)
        self.goal = (4, 4)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.size and 0 <= ny < self.size:
            self.state = (nx, ny)
            reward = 10 if self.state == self.goal else -1
            done = self.state == self.goal
        else:
            reward = -5
            done = False
        return self.state, reward, done

# 에이전트
class Agent:
    def __init__(self):
        self.q_table = np.zeros((5, 5, 4))
        self.temp = 1.0

    def softmax(self, q_values):
        exp_q = np.exp(q_values / self.temp)
        return exp_q / np.sum(exp_q)

    def choose_action(self, state):
        x, y = state
        return np.random.choice(4, p=self.softmax(self.q_table[x, y]))

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        self.q_table[x, y, action] += 0.1 * (reward + 0.95 * np.max(self.q_table[nx, ny]) - self.q_table[x, y, action])
        self.temp = max(0.1, self.temp * 0.95)

# 시각화
def plot_path(env, agent, ep):
    path = [env.reset()]
    for _ in range(10):
        action = agent.choose_action(path[-1])
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break
    fig, ax = plt.subplots(figsize=(4, 4))
    grid = np.zeros((5, 5))
    grid[env.goal] = 2
    for x, y in path:
        if grid[x, y] == 0:
            grid[x, y] = 1
    ax.imshow(grid, cmap='hot')
    ax.plot([y for _, y in path], [x for x, _ in path], 'b-o')
    st.pyplot(fig)
    plt.close(fig)

# 메인
def main():
    st.title("간단 확률 경로")
    episodes = int(st.number_input("에피소드 수", min_value=10, max_value=50, value=20))
    if st.button("학습"):
        env = Env()
        agent = Agent()
        rewards = []
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            for _ in range(10):
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
            if ep % (episodes // 5) == 0:
                st.write(f"에피소드 {ep}, 보상: {total_reward}")
                plot_path(env, agent, ep)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(rewards)
        st.pyplot(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
```

### 입력 기능 설명
- **어디서 입력?**:
  - `st.number_input("에피소드 수", min_value=10, max_value=50, value=20)`: 웹 화면에 "에피소드 수" 입력란 표시. 10~50 사이 숫자 입력, 기본값 20.
  - `st.button("학습")`: "학습" 버튼 클릭 시 입력된 에피소드 수로 학습 시작.
- **입력 후 동작**:
  - 입력한 에피소드 수(예: 20)만큼 학습.
  - 매 5번 에피소드마다 보상(예: "에피소드 0, 보상: -7")과 경로 그래프 표시.
  - 마지막에 보상 그래프(에피소드 수만큼 점) 출력, 보상은 -5~-10에서 +10 근처로 올라감.

### 의존성 (Requirements)
Streamlit Cloud에서 이 앱(`mini_prob.py`)을 배포하려면 아래 두 파일이 필요해:

1. **`requirements.txt`**:
   ```plaintext
   streamlit==1.39.0
   numpy==1.26.4
   matplotlib==3.8.3
   ```
   - **streamlit**: 웹 앱 실행.
   - **numpy**: Q-table과 소프트맥스 계산.
   - **matplotlib**: 경로와 보상 그래프.

2. **`packages.txt`**:
   ```plaintext
   libfreetype6-dev
   pkg-config
   python3-distutils
   ```
   - `matplotlib` 설치에 필요한 시스템 패키지.

### 수행
1. **코드 저장**:
   - `prob.py`를 위 코드로 덮어쓰기. `→` 같은 오류 문자 없음 확인.

2. **GitHub 푸시**:
   ```bash
   git add prob.py requirements.txt packages.txt
   git commit -m "입력 기능
