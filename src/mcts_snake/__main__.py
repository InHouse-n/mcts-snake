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





    ##TODO

    ## put the pieces togther and see if it makes sense

    # set up environment and model
    # predict a step -> run the mcts to get the best move
    # take this step
    # get returns from this step
    # train on batch?
    # keep losses?


if __name__ == "__main__":
    main()
