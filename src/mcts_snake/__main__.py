import logger_provider

from src.mcts_snake.snake_game import SnakeWorldEnv


def main():

    log = logger_provider.get_logger(__name__)

    env = SnakeWorldEnv(render_mode="human")
    obs = env.reset()

    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        # Render the game
        env.render()

        if done == True:
            break

    env.close()


if __name__ == "__main__":
    main()
