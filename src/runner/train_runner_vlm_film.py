"""Train runner that wires FlowModelVLMFiLM without touching existing files."""

from __future__ import annotations

from src.runner.train_runner import TrainRunner as BaseTrainRunner


class TrainRunnerVLMFiLM(BaseTrainRunner):
    """FlowModel runner with VLM-FiLM support (new entrypoint)."""

    def _create_model(self):
        common_args = {
            'arch_name': self.model_name,
            'learning_rate': self.model_cfg.get('learning_rate', self.model_info.default_lr),
            'weight_decay': self.model_cfg.get('weight_decay', 1e-5),
            'experiment_name': self.model_cfg.get('experiment_name', f"{self.dataset_name}/{self.model_name}"),
            'data_name': self.model_cfg.get('data_name', self.dataset_name),
            'image_size': self.model_cfg.get('image_size', 224),
            'num_classes': self.model_cfg.get('num_classes', 2),
        }

        if self.model_info.task != 'flow':
            return super()._create_model()

        from src.archs.flow_model_vlm_film import FlowModelVLMFiLM
        return FlowModelVLMFiLM(
            **common_args,
            patch_plan=self.model_cfg.get('patch_plan', None),
            dim=self.model_cfg.get('dim', 32),
            timesteps=self.model_cfg.get('timesteps', 15),
            sigma=self.model_cfg.get('sigma', 0.25),
            num_ensemble=self.model_cfg.get('num_ensemble', 1),
            log_image_enabled=self.model_cfg.get('log_image_enabled', False),
            log_image_names=self.model_cfg.get('log_image_names', None),
            use_sliding_infer=self.model_cfg.get('use_sliding_infer', True),
            model_channels=self.model_cfg.get('model_channels', 32),
            channel_mult=self.model_cfg.get('channel_mult', [1, 2, 4, 8]),
            channel_mult_emb=self.model_cfg.get('channel_mult_emb', 4),
            num_blocks=self.model_cfg.get('num_blocks', 3),
            attn_resolutions=self.model_cfg.get('attn_resolutions', [16, 16, 8, 8]),
            dropout=self.model_cfg.get('dropout', 0.0),
            label_dim=self.model_cfg.get('label_dim', 0),
            augment_dim=self.model_cfg.get('augment_dim', 0),
            # Loss configuration
            loss_type=self.model_cfg.get('loss_type', 'l2'),
            bce_weight=self.model_cfg.get('bce_weight', 0.5),
            l2_weight=self.model_cfg.get('l2_weight', 0.1),
            dice_weight=self.model_cfg.get('dice_weight', 0.2),
            lambda_soft=self.model_cfg.get('lambda_soft', 1.0),
            loss=self.model_cfg.get('loss', None),
            use_gradient_checkpointing=self.model_cfg.get('use_gradient_checkpointing', False),
            mode=self.model_cfg.get('mode', 'cfm_continuous'),
            dfm_sampler=self.model_cfg.get('dfm_sampler', 'euler'),
            dfm_eps=self.model_cfg.get('dfm_eps', 1e-6),
            debug_dfm=self.model_cfg.get('debug_dfm', False),
            # VLM-FiLM
            use_vlm_film=self.model_cfg.get('use_vlm_film', False),
            vlm_film_config=self.model_cfg.get('vlm_film_config', None),
            vlm_film_decoder_stages=self.model_cfg.get('vlm_film_decoder_stages', None),
            vlm_update_interval=self.model_cfg.get('vlm_update_interval', 50),
            vlm_update_interval_eval=self.model_cfg.get('vlm_update_interval_eval', 1),
            # Junction-aware FiLM gating
            junction_gating_config=self.model_cfg.get('junction_gating_config', None),
        )
