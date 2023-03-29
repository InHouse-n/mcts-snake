from src.mcts_snake.snake_game import SnakeWorldEnv
import logger_provider

def main():
    """Main function, start coding here"""
    log = logger_provider.get_logger(__name__)

    log.info("helll90 world")



    env = SnakeWorldEnv(render_mode='human')
    obs = env.reset()

    while True:
        log.info("hey")
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # Render the game
        env.render()
        
        if done == True:
            break

    env.close()
    log.info("hoer")


if __name__ == "__main__":
    main()
    