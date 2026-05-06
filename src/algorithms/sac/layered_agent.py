from src.algorithms.sac.sac_agent import SACAgent

class LayeredSACAgent(SACAgent):
    """This agent splits the control into layers, into attitude control and position / waypoint control. The SAC agent is used for both layers, but they are trained separately. The position / waypoint control layer outputs desired attitude and altitude, which are then tracked by the attitude control layer.

    Args:
        SACAgent (_type_): _description_
    """
    def __init__(self, obs_dim, action_dim, num_envs, device, config, attitude_config):
        super().__init__(obs_dim, action_dim, num_envs, device, config)

        self.attitude_agent = SACAgent(obs_dim, action_dim, num_envs, device, attitude_config)

    def act(self, obs):
        """observation(dim 22):
            0. ego_delta_npos      (unit: km)
            1. ego_delta_epos       (unit km)
            2. ego_delta_altitude            (unit: km)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
            16. ego_T                  (unit: %)
            17. ego_el                 (unit: %)
            18. ego_ail                (unit: %)
            19. ego_rud                (unit: %)
            20. ego_lef                (unit: %)
            21. EAS2TAS
        """
        # First get the desired attitude and altitude from the position control agent
        obs_clone = obs.clone()
        att_alt = super().act(obs_clone)

        # Sub in the action for the attitude actor observation
        obs_clone[0:3] = att_alt
        action = self.attitude_agent.act(obs_clone)

        return action