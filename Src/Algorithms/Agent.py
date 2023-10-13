# Parent to all algorithm
class Agent:

    def __init__(self, config):
        #self.state_low, self.state_high = config.env.observation_space.low, config.env.observation_space.high
        #self.state_diff = self.state_high - self.state_low

        # try:
        #     if config.env.action_space.dtype == np.float32:
        #         self.action_low, self.action_high = config.env.action_space.low, config.env.action_space.high
        #         self.action_diff = self.action_high - self.action_low
        # except:
        #     print('-------------- Warning: Possible action type mismatch ---------------')

        # self.state_dim = config.env.observation_space.shape[0]

       
        # self.action_dim = config.env.n_actions
      
        self.config = config

        # Abstract class variables
        self.modules = None

    def init(self):
         for name, m in self.modules:
             m.to(self.config.device)

    def clear_gradients(self):
        for _, module in self.modules:
            module.optim.zero_grad()
            
    def clear_actor_gradients(self):
        self.modules[0][1].optim.zero_grad()

    def clear_critic_gradients(self):
        self.modules[1][1].optim.zero_grad()

    def save(self):
        if self.config.save_model:
            for name, module in self.modules:
                module.save(self.config.paths['checkpoint'] + name+'.pt')

    def step(self, loss, clip_norm=False):
        self.clear_gradients()
        loss.backward()
        for _, module in self.modules:
            module.step(clip_norm)
            
    def actor_step(self, loss, clip_norm=False):
        self.clear_actor_gradients()
        loss.backward()
        self.modules[0][1].step(clip_norm)
    def critic_step(self, loss, clip_norm=False):
        self.clear_actor_gradients()
        loss.backward()
        self.modules[1][1].step(clip_norm)

    def reset(self):
        for _, module in self.modules:
            module.reset()
