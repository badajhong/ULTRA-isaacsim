"""Explicit environment conditions for tracking evaluation, applied before scene creation."""


def configure_tracking_randomization(cfg, mode):
    """Use unrandomized physics/sensing or preserve the caller's training configuration.

    Object mass, COM and inertia randomization is not gated by domain_rand_general
    in the task, so switching that flag alone does not create a unrandomized benchmark.
    This function is called only for tracking evaluation; training YAMLs are untouched.
    """
    if mode == 'config':
        return
    if mode != 'none':
        raise ValueError(f'Unknown evaluation randomization mode: {mode}')
    cfg['domain_rand'].update(
        domain_rand_general=False, randomize_gravity=False, randomize_friction=False,
        randomize_base_mass=False, randomize_base_com=False, randomize_motor=False,
        push_robots=False, action_delay=False,
        obj_mass_range=[1.0, 1.0], obj_mass_rare_range=[1.0, 1.0], obj_mass_rare_every=0,
        obj_com_range=[0.0, 0.0], obj_inertia_range=[1.0, 1.0],
    )
    cfg['noise']['add_noise'] = False
    cfg['noise']['noise_level'] = 0.0
    if 'task' in cfg:
        cfg['task']['randomize'] = False
    for name in ('pointNoiseStd', 'pointDropoutProb', 'pointOutlierProb', 'pointDepthNoiseScale',
                 'pointOcclusionProb', 'pointClusterNoiseStd', 'pointTranslationNoise',
                 'cameraRotNoise', 'cameraPosNoise'):
        cfg['env'][name] = 0.0
    for name in ('pointDensityMin', 'pointDensityMax', 'pointScaleMin', 'pointScaleMax'):
        cfg['env'][name] = 1.0
