import config
import subprocess, os
from train import train
from selfplay import selfplay

for _ in range(config.LEARNING_EPOCHS):

    selfplay()
    train(config.SELFPLAY_BATCH_SIZE, "./saved_games")
    os.makedirs("./used_saved_games", exist_ok=True)
    try:
        if config.SELFPLAY_BATCH_SIZE != 1:
            os.makedirs(f"./used_saved_games/batch_{config.CURR_BATCH_NUMBER+_}", exist_ok=False)
            for __ in range(1, config.SELFPLAY_BATCH_SIZE + 1):
                subprocess.run(["mv", f"./saved_games/game_{__}", f"./used_saved_games/batch_{config.CURR_BATCH_NUMBER+_}"])
                
        else:
            os.makedirs("./used_saved_games/solo_datasets", exist_ok=True)
            subprocess.run(["mv", "saved_games/game_1", f"saved_games/game_{config.CURR_SINGLE_BATCH_NUMBER+_}"])
            subprocess.run(["mv", f"saved_games/game_{config.CURR_SINGLE_BATCH_NUMBER+_}", f"./used_saved_games/solo_datasets"])
    except:
        raise ValueError("Please check/update the 'config.CURR_SINGLE_BATCH_NUMBER' or 'config.CURR_BATCH_NUMBER' parameters in config.py .")