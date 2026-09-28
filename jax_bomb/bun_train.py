"""地图 806 抢包子专用 PPO 训练入口。"""

import os

os.environ["JAXBOMB_RULE"] = "bun"

from .jax_train import main


if __name__ == "__main__":
    main()
