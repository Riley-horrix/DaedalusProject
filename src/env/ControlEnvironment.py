sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from lib.NeuralPlane.envs.control_env import ControlEnv
from lib.NeuralPlane.envs.models.UAV_model import UAVModel
from lib.NeuralPlane.envs.models.F16_model import F16Model
from lib.NeuralPlane.envs.tasks.heading_task import HeadingTask
from lib.NeuralPlane.envs.tasks.control_task import ControlTask
from lib.NeuralPlane.envs.tasks.tracking_task import TrackingTask


class ControlEnvironment(ControlEnv):
    """
    ControlEnvironment is a fly control environment for a single agent to do control tasks.

    Wraps the NP ControlEnv to allow adding new tasks.
    """
    def __init__(self, num_envs=1, config='waypoint', model='UAV', random_seed=None, device="cuda:0"):
        super().__init__(num_envs, config, model, random_seed, device)

    def load(self, random_seed, config, model):
        if random_seed is not None:
            self.seed(random_seed)

        if model == 'F16':
            self.model = F16Model(self.config, self.n, self.device, random_seed)
        elif model == 'UAV':
            self.model = UAVModel(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError

        if config == 'heading':
            self.task = HeadingTask(self.config, self.n, self.device, random_seed)
        elif config == 'control':
            self.task = ControlTask(self.config, self.n, self.device, random_seed)
        elif config == 'tracking':
            self.task = TrackingTask(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError