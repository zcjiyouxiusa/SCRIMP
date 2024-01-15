# import gym
# import time
# env=gym.make('CartPole-v1')
# state=env.reset()
# while True:
#     env.render()
#     action=env.action_space.sample()
#     state,reward,done,*info=env.step(action)
#     print('state={0},reward={1}'.format(state,reward))
#     if done:
#         print('over')
#         break
# time.sleep(0.1)
# env.close()
    
# # 首先，导入库文件（包括gym模块和gym中的渲染模块）
# import gym
# from gym.envs.classic_control import rendering

# # 我们生成一个类，该类继承 gym.Env. 同时，可以添加元数据，改变渲染环境时的参数
# class Test(gym.Env):
#     # 如果你不想改参数，下面可以不用写
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 2
#     }
# # 我们在初始函数中定义一个 viewer ，即画板
#     def __init__(self):
#         self.viewer = rendering.Viewer(600, 400)   # 600x400 是画板的长和框
#     # 继承Env render函数
#     def render(self, mode='human', close=False):
#         # 下面就可以定义你要绘画的元素了
#         line1 = rendering.Line((100, 300), (500, 300))
#         line2 = rendering.Line((100, 200), (500, 200))
#         # 给元素添加颜色
#         line1.set_color(0, 0, 0)
#         line2.set_color(0, 0, 0)
#         # 把图形元素添加到画板中
#         self.viewer.add_geom(line1)
#         self.viewer.add_geom(line2)

#         return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
# # 最后运行
# if __name__ == '__main__':
#     t = Test()
#     while True:
#         t.render()



# import time
# import gym

# env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
# # env = gym.make('ALE/Breakout-v4')
# print("Observation Space: ", env.observation_space)
# print("Action Space       ", env.action_space)


# obs = env.reset()

# for i in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     obs, reward, done, *info = env.step(action)
#     time.sleep(0.01)
# env.close()


# from matplotlib import pyplot as plt

# x = [1, 2, 3, 4, 5, 6]
# y = [10, 20, 30, 40, 50, 60]

# plt.plot(x, y)
    
# plt.show()

import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()