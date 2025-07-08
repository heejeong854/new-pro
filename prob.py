import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st

# 환경: 5x5 그리드
class Env:
    def __init__(self):
        self.size = 5
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
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.size and 0 <= ny < self.size:
            self.state = (nx, ny)
            reward = 10 if self.state == self.goal else -1
            done = self.state == self.goal
        else:
            reward = -5
            done = False
        return self.state, reward, done

# 에이전트: 소프트맥스 Q-learning
class Agent:
    def __init__(self):
        self.q_table = np.zeros((5, 5, 4))
        self.temp = 1.0  # 소프트맥스 온도

    def softmax(self, q_values):
        exp_q = np.exp(q_values / self.temp)
        return exp_q / np.sum(exp_q)

    def choose_action(self, state):
        x, y = state
        return np.random.choice(4, p=self.softmax(self.q_table[x, y]))

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        self.q_table[x, y, action] += 0.1 * (
            reward + 0.95 * np.max(self.q_table[nx, ny]) - self.q_table[x, y, action]
        )

# 경로 시각화
def plot_path(env, agent, ep):
    path = [env.start]
    state = env.reset()
    done = False
    for _ in range(10):
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
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
    ax.set_title(f'에피소드 {ep}')
    st.pyplot(fig)
    plt.close(fig)

# 메인
def main():
    st.title("확률 경로 최적화")
    env = Env()
    agent = Agent()
    rewards = []

    for ep in range(20):
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
        if ep % 5 == 0:
            st.write(f"에피소드 {ep}, 보상: {total_reward}")
            plot_path(env, agent, ep)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(rewards)
    ax.set_title('보상')
    st.pyplot(fig)
    plt.close(fig)

if __name__ == "__main__":
    main()
