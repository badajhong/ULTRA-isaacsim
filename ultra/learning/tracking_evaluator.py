"""Sampled student tracking evaluation using world-space reference positions."""

import json
import math
from collections import deque
from pathlib import Path

import torch

from learning.ultra_players_distill import UltraPlayerContinuousDistill


class TrackingSuccessPlayer(UltraPlayerContinuousDistill):
    def __init__(self, config, episodes, full_clips=False, output='', checkpoint='', motion_dir='', seed=None,
                 training_conditions=False, eval_policy='prior', horizon_length=8,
                 model_type='distillation', checkpoint_epoch=None, env_config='', train_config=''):
        super().__init__(config)
        self.eval_episodes = episodes
        self.full_clips = full_clips
        self.output = output
        self.checkpoint = checkpoint
        self.motion_dir = motion_dir
        self.seed = seed
        self.training_conditions = training_conditions
        self.eval_policy = eval_policy
        self.horizon_length = horizon_length
        self.model_type = model_type
        self.checkpoint_epoch = checkpoint_epoch
        self.env_config = env_config
        self.train_config = train_config

    def get_action(self, obs_dict, is_determenistic=False):
        if self.eval_policy == 'teacher':
            if self._last_expert is None or 'teacher_obs' not in self._last_expert:
                raise RuntimeError('Teacher evaluation requires privileged observations')
            task = self.env.task
            task.model.eval()
            with torch.no_grad():
                result = task.model({'is_train': False, 'prev_actions': None,
                                     'obs': self._last_expert['teacher_obs'].to(task.device),
                                     'rnn_states': None})
            action = result['mus'] if is_determenistic else result['actions']
            return torch.clamp(action, -1.0, 1.0).to(self.device)
        if self.eval_policy == 'prior':
            return super().get_action(obs_dict, is_determenistic)
        # Privileged diagnostic: use the same prior + encoder latent as the training path.
        obs = self._preproc_obs(obs_dict['obs'])
        if self._last_expert is None or 'teacher_obs' not in self._last_expert:
            raise RuntimeError('Privileged encoder evaluation requires teacher observations')
        if self._vae_noise is None or self._vae_noise.shape[0] != obs.shape[0]:
            vae_dim = getattr(self.model.a2c_network, 'vae_dim', self.config.get('vae_dim', 64))
            self._vae_noise = torch.randn(obs.shape[0], vae_dim, device=self.device)
        input_dict = {
            'is_train': False, 'prev_actions': None, 'obs': obs, 'rnn_states': self.states,
            'teacher_obs': self._last_expert['teacher_obs'].to(self.device),
            'vae_noise': self._vae_noise, 'with_encoder': True, 'with_vae': False, 'skip_critic': True,
        }
        self.model.eval()
        with torch.no_grad():
            result = self.model(input_dict)
        action = result['mus'] if is_determenistic else result['actions']
        return torch.clamp(action, -1.0, 1.0)

    def run(self):
        if self.full_clips:
            return self.run_full_clips()
        task = self.env.task
        # Evaluate moving references only, rather than the training-only stand-still augmentation.
        task.stand_still_ratio = 0.0
        obs = self.env_reset()
        humanoid_ok = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
        object_ok = humanoid_ok.clone()
        completed = human_success = combined_success = 0
        if self.is_rnn:
            self.init_rnn()

        print(f"[tracking evaluation] Collecting {self.eval_episodes} sampled episodes; threshold 0.3 m.", flush=True)
        print("[tracking evaluation] Humanoid error is mean world-space distance over 39 body origins; "
              "object error is world-space center distance.", flush=True)
        print(f"[tracking evaluation] {task.num_motions} available clips; random start frames; "
              f"maximum rollout {task.rollout_length} steps; domain randomization "
              f"{task.cfg['domain_rand']['domain_rand_general']}.", flush=True)
        while completed < self.eval_episodes:
            action = self.get_action(obs if isinstance(obs, dict) else {'obs': obs}, is_determenistic=True)
            obs, _, done, _ = self.env_step(self.env, action)
            current, reference = task._curr_obs, task._curr_ref_obs
            body_error = (current[:, 84:201].reshape(-1, 39, 3) -
                          reference[:, 84:201].reshape(-1, 39, 3)).norm(dim=-1).mean(dim=-1)
            object_error = (current[:, 71:74] - reference[:, 71:74]).norm(dim=-1)
            humanoid_ok &= torch.isfinite(body_error) & (body_error < 0.3) & ~task._terminate_buf.bool()
            object_ok &= torch.isfinite(object_error) & (object_error < 0.3)

            done_ids = done.reshape(-1).nonzero(as_tuple=False).flatten()
            if done_ids.numel():
                selected = done_ids[:self.eval_episodes - completed]
                human_success += humanoid_ok[selected].sum().item()
                combined_success += (humanoid_ok[selected] & object_ok[selected]).sum().item()
                completed += selected.numel()
                print(f"[tracking evaluation] {completed}/{self.eval_episodes}: "
                      f"humanoid {human_success/completed:.2%}, "
                      f"humanoid + object {combined_success/completed:.2%}", flush=True)
                if completed >= self.eval_episodes:
                    break
                if self.is_rnn:
                    for state in self.states:
                        state[:, done_ids, :] = 0
                humanoid_ok[done_ids] = True
                object_ok[done_ids] = True
                obs = self.env_reset(done_ids)

        print(f"[tracking evaluation] FINAL: {completed} episodes; "
              f"humanoid {human_success}/{completed} ({100 * human_success/completed:.2f}%); "
              f"humanoid + object {combined_success}/{completed} ({100 * combined_success/completed:.2f}%).",
              flush=True)

    def run_full_clips(self):
        """Evaluate each input clip once, using its matching, fixed object in the simulator."""
        from env.tasks.ultra import Ultra

        task = self.env.task
        num_envs, device = task.num_envs, task.device
        num_objects = len(task.object_name)
        if num_envs < num_objects:
            raise ValueError(f'Need at least {num_objects} environments, one per object type/scale; got {num_envs}')

        queues = [deque() for _ in range(num_objects)]
        for motion_id, object_id in enumerate(task.object_id.tolist()):
            queues[object_id].append(motion_id)
        fallback = [queue[0] for queue in queues]
        task._eval_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        active = [False] * num_envs

        def assign(env_id):
            object_id = env_id % num_objects
            active[env_id] = bool(queues[object_id])
            motion_id = queues[object_id].popleft() if active[env_id] else fallback[object_id]
            task._eval_motion_ids[env_id] = motion_id

        for env_id in range(num_envs):
            assign(env_id)
        checkpoint_epoch = self.checkpoint_epoch
        curriculum_step = None
        mask_schedule = None
        if self.training_conditions:
            if task._state_init != Ultra.StateInit.Hybrid:
                raise ValueError('Training-conditions evaluation requires stateInit: Hybrid')
            checkpoint_epoch = self.checkpoint_epoch
            if checkpoint_epoch is None:
                checkpoint_data = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
                if 'epoch' not in checkpoint_data:
                    raise ValueError('Checkpoint has no epoch; cannot reconstruct training mask curriculum')
                checkpoint_epoch = int(checkpoint_data['epoch'])
                del checkpoint_data
            mask_schedule = task.update_obs_keep_probabilities(checkpoint_epoch)
            # The checkpoint does not save this task counter. Estimate per-rank environment steps from the epoch.
            curriculum_step = checkpoint_epoch * int(self.horizon_length)
            task.common_step_counter = curriculum_step
        else:
            task._state_init = Ultra.StateInit.Start
            task.stand_still_ratio = 0.0
            # The loader appends 20 copies of the last reference frame for training.
            task.max_episode_length -= 20
        if (task.max_episode_length < 2).any():
            raise ValueError('Every evaluation motion must contain at least two frames')
        if not self.training_conditions:
            task.rollout_length = int(task.max_episode_length.max().item()) + 1
        obs = self.env_reset()
        if self.is_rnn:
            self.init_rnn()

        ok_h = torch.ones(num_envs, dtype=torch.bool, device=device)
        ok_o = ok_h.clone()
        frames = torch.zeros(num_envs, dtype=torch.long, device=device)
        jitter_frames = frames.clone()
        # Sums of per-frame mean distances, in meters, degrees, and m/s^3.
        sums = torch.zeros((num_envs, 5), dtype=torch.float64, device=device)
        history = torch.zeros((3, num_envs, 39, 3), device=device)
        results = []

        def include_initial_frame(env_ids):
            # Reset refreshes simulation state and references, but not _curr_obs.
            # Measure frame zero too: otherwise an initial threshold violation
            # can be missed and the final report contains only N-1 frames.
            task._compute_hoi_observations(env_ids)
            current = task._curr_obs[env_ids]
            reference = task.hoi_data[task.data_id[env_ids], task.progress_buf[env_ids]]
            body = current[:, 84:201].reshape(-1, 39, 3)
            ref_body = reference[:, 84:201].reshape(-1, 39, 3)
            global_error = (body - ref_body).norm(dim=-1).mean(dim=-1)
            relative_error = ((body - current[:, None, :3]) -
                              (ref_body - reference[:, None, :3])).norm(dim=-1).mean(dim=-1)
            object_error = (current[:, 71:74] - reference[:, 71:74]).norm(dim=-1)
            q = torch.nn.functional.normalize(current[:, 74:78], dim=-1)
            q_ref = torch.nn.functional.normalize(reference[:, 74:78], dim=-1)
            rot_error = 2 * torch.acos((q * q_ref).sum(dim=-1).abs().clamp(max=1.0)) * (180 / math.pi)
            sums[env_ids, :4] = torch.stack((global_error, relative_error, object_error, rot_error), dim=1).double()
            history[0, env_ids] = body
            frames[env_ids] = 1
            fell = task._rigid_body_pos[env_ids, 0, 2] < task._termination_heights
            ok_h[env_ids] = torch.isfinite(global_error) & (global_error < 0.3) & ~fell
            ok_o[env_ids] = torch.isfinite(object_error) & (object_error < 0.3)

        include_initial_frame(torch.arange(num_envs, device=device))
        print(f'[table1 evaluation] Evaluating {task.num_motions} clips; '
              f'{num_envs} environments, {num_objects} object types/scales; '
              f'conditions={"training" if self.training_conditions else "full_clip"}; '
              f'policy={self.eval_policy}; model={self.model_type}; '
              f'randomization={task.cfg.get("_eval_randomization", "config") }.' , flush=True)
        if self.training_conditions:
            print(f'[table1 evaluation] Checkpoint epoch {checkpoint_epoch}; estimated training step '
                  f'{curriculum_step}; random start and observation masks, rollout {task.rollout_length}.', flush=True)

        while len(results) < task.num_motions:
            action = self.get_action(obs if isinstance(obs, dict) else {'obs': obs}, is_determenistic=True)
            obs, _, done, _ = self.env_step(self.env, action)
            current, reference = task._curr_obs, task._curr_ref_obs
            body = current[:, 84:201].reshape(-1, 39, 3)
            ref_body = reference[:, 84:201].reshape(-1, 39, 3)
            global_error = (body - ref_body).norm(dim=-1).mean(dim=-1)
            relative_error = ((body - current[:, None, :3]) -
                              (ref_body - reference[:, None, :3])).norm(dim=-1).mean(dim=-1)
            object_error = (current[:, 71:74] - reference[:, 71:74]).norm(dim=-1)
            q = torch.nn.functional.normalize(current[:, 74:78], dim=-1)
            q_ref = torch.nn.functional.normalize(reference[:, 74:78], dim=-1)
            rot_error = 2 * torch.acos((q * q_ref).sum(dim=-1).abs().clamp(max=1.0)) * (180 / math.pi)
            jerk = (body - 3 * history[0] + 3 * history[1] - history[2]).norm(dim=-1).mean(dim=-1) / task.dt**3
            valid_jitter = frames >= 3
            jitter_frames += valid_jitter.long()
            sums += torch.stack((global_error, relative_error, object_error, rot_error,
                                 torch.where(valid_jitter, jerk, 0)), dim=1).double()
            history[2] = history[1].clone()
            history[1] = history[0].clone()
            history[0] = body
            frames += 1
            fell = (task._rigid_body_pos[:, 0, 2] < task._termination_heights) | task._terminate_buf.bool()
            ok_h &= torch.isfinite(global_error) & (global_error < 0.3) & ~fell
            ok_o &= torch.isfinite(object_error) & (object_error < 0.3)

            done_ids = done.reshape(-1).nonzero(as_tuple=False).flatten()
            if not done_ids.numel():
                continue
            for env_id in done_ids.tolist():
                if active[env_id]:
                    motion_id = int(task._eval_motion_ids[env_id].item())
                    n = int(frames[env_id].item())
                    j = int(jitter_frames[env_id].item())
                    v = sums[env_id].tolist()
                    results.append({
                        'motion': task.motion_file[motion_id], 'frames_evaluated': n,
                        'start_frame': int(task.start_times[env_id].item()),
                        'stand_still': bool(task.is_stand_still[env_id].item()),
                        'reference_frames': int(task.max_episode_length[motion_id].item()),
                        'terminated_early': bool(task._terminate_buf[env_id].item()),
                        'success_humanoid': bool(ok_h[env_id].item()),
                        'success_humanoid_object': bool((ok_h[env_id] & ok_o[env_id]).item()),
                        'global_body_error_m': v[0] / n, 'root_relative_body_error_m': v[1] / n,
                        'object_position_error_m': v[2] / n, 'object_rotation_error_deg': v[3] / n,
                        'body_jitter_m_per_s3': v[4] / j if j else None,
                        '_sums': v, '_jitter_frames': j,
                    })
                    print(f'[table1 evaluation] {len(results)}/{task.num_motions} clips complete', flush=True)
                assign(env_id)
            ok_h[done_ids] = True
            ok_o[done_ids] = True
            frames[done_ids] = 0
            jitter_frames[done_ids] = 0
            sums[done_ids] = 0
            history[:, done_ids] = 0
            if self.is_rnn:
                for state in self.states:
                    state[:, done_ids, :] = 0
            if len(results) < task.num_motions:
                obs = self.env_reset(done_ids)
                include_initial_frame(done_ids)

        total_frames = sum(r['frames_evaluated'] for r in results)
        total_jitter_frames = sum(r['_jitter_frames'] for r in results)
        totals = [sum(r['_sums'][i] for r in results) for i in range(5)]
        summary = {
            'clips': len(results), 'frames_evaluated': total_frames,
            'success_humanoid_pct': 100 * sum(r['success_humanoid'] for r in results) / len(results),
            'success_humanoid_object_pct': 100 * sum(r['success_humanoid_object'] for r in results) / len(results),
            'global_body_error_cm': 100 * totals[0] / total_frames,
            'root_relative_body_error_cm': 100 * totals[1] / total_frames,
            'object_position_error_cm': 100 * totals[2] / total_frames,
            'object_rotation_error_deg': totals[3] / total_frames,
            'body_jitter_m_per_s3': totals[4] / total_jitter_frames if total_jitter_frames else None,
        }
        for result in results:
            del result['_sums'], result['_jitter_frames']
        report = {'protocol': {
            'warning': 'Approximation, not official Table I reproduction; exact split and metric definitions unavailable.',
            'checkpoint': self.checkpoint, 'motion_directory': self.motion_dir,
            'model_type': self.model_type, 'environment_task': type(task).__name__,
            'env_config': self.env_config, 'train_config': self.train_config,
            'finetune_goal_offsets': False,
            'seed': self.seed, 'mode': task.task_mode, 'clips_once': True,
            'parallel_environments': task.num_envs,
            'conditions': 'training' if self.training_conditions else 'full_clip',
            'start_frame': 'training_random' if self.training_conditions else 0,
            'policy': self.eval_policy,
            'policy_note': ({'encoder': 'Privileged student encoder uses teacher observations; no teacher-action mixing.',
                             'prior': 'Prior-only deployable student; no teacher-action mixing.',
                             'teacher': 'Privileged teacher using unmasked normalized teacher observations.'}[self.eval_policy]),
            'action_selection': ('teacher mean action' if self.eval_policy == 'teacher'
                                 else 'mean action; sampled VAE noise'),
            'checkpoint_epoch': checkpoint_epoch, 'estimated_common_step_counter': curriculum_step,
            'checkpoint_horizon_length': self.horizon_length,
            'mask_curriculum': mask_schedule, 'rollout_length': task.rollout_length,
            'stand_still_ratio': task.stand_still_ratio,
            'success_threshold_m': 0.3, 'body_origins': 39,
            'initial_frame_included': True,
            'relative_error': 'Subtract each root position; no rotational alignment.',
            'rotation_error': 'Quaternion geodesic angle, degrees.',
            'jitter': 'Mean magnitude of third finite difference of world body positions divided by dt^3.',
            'error_aggregation': 'Frame-weighted over simulated frames, including failed episodes up to termination.',
            'dt_seconds': task.dt, 'domain_randomization': task.cfg['domain_rand']['domain_rand_general'],
            'randomization_mode': task.cfg.get('_eval_randomization', 'config'),
            'physics_asset_reference': {'source': 'Isaac Sim URDF conversion and convex decomposition'},
            'evaluation_split': 'unspecified (supplied motion directory; no ID/OOD split generated)',
            'domain_randomization_config': task.cfg['domain_rand'],
            'observation_noise': task.cfg['noise']['add_noise'],
            'padded_final_frames_excluded': 0 if self.training_conditions else 20,
        }, 'summary': summary, 'motions': results}
        print('[table1 evaluation] APPROXIMATE RESULTS: ' + json.dumps(summary, indent=2), flush=True)
        print('[table1 evaluation] | Humanoid % | +Object % | Global cm | Root-relative cm | '
              'Jitter m/s^3 | Object cm | Rotation deg |', flush=True)
        jitter = summary['body_jitter_m_per_s3']
        jitter_text = f'{jitter:.2f}' if jitter is not None else 'N/A'
        print(f"[table1 evaluation] | {summary['success_humanoid_pct']:.2f} | "
              f"{summary['success_humanoid_object_pct']:.2f} | {summary['global_body_error_cm']:.2f} | "
              f"{summary['root_relative_body_error_cm']:.2f} | {jitter_text} | "
              f"{summary['object_position_error_cm']:.2f} | {summary['object_rotation_error_deg']:.2f} |", flush=True)
        if self.output:
            output = Path(self.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(f'[table1 evaluation] Saved {output}', flush=True)
