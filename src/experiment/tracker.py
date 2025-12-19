"""Experiment tracking and management system."""

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Experiment:
    """Experiment metadata."""
    id: str
    model: str
    dataset: str
    config: Dict[str, Any]
    git_hash: Optional[str]
    git_branch: Optional[str]
    created_at: str
    dir: Path
    status: str = 'running'  # running, completed, failed
    final_metrics: Optional[Dict[str, float]] = None
    best_checkpoint: Optional[str] = None
    tag: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['dir'] = str(d['dir'])
        d['created_at'] = str(d['created_at'])
        return d


class ExperimentTracker:
    """
    Centralized experiment tracking system.
    
    Features:
    - Automatic experiment ID generation
    - Directory structure management
    - Config versioning
    - Git integration for reproducibility
    - Metrics tracking
    - Experiment database (JSON)
    """

    def __init__(self, root_dir: str = "experiments"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "experiments.json"
        self._load_db()

    def _load_db(self):
        """Load experiment database."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = {}

    def _save_db(self):
        """Save experiment database."""
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)

    def _generate_id(self, model: str, dataset: str, tag: Optional[str] = None) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tag:
            return f"{model}_{dataset}_{tag}_{timestamp}"
        return f"{model}_{dataset}_{timestamp}"

    def _get_git_info(self) -> tuple:
        """Get current git hash and branch."""
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()

            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()

            return git_hash, git_branch
        except Exception:
            return None, None

    def create_experiment(
        self,
        model: str,
        dataset: str,
        config: Dict[str, Any],
        tag: Optional[str] = None,
    ) -> Experiment:
        """
        Create new experiment with automatic tracking.
        
        Args:
            model: Model name
            dataset: Dataset name
            config: Full configuration dict
            tag: Optional tag for identification
        
        Returns:
            Experiment object with all metadata
        """
        # Generate ID and directory (include tag in ID if provided)
        exp_id = self._generate_id(model, dataset, tag)
        exp_dir = self.root / model / dataset / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Get git info for reproducibility
        git_hash, git_branch = self._get_git_info()

        # Create experiment object
        experiment = Experiment(
            id=exp_id,
            model=model,
            dataset=dataset,
            config=config,
            git_hash=git_hash,
            git_branch=git_branch,
            created_at=datetime.now().isoformat(),
            dir=exp_dir,
            status='running',
            tag=tag,
        )

        # Save config to experiment directory
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Save git info
        if git_hash:
            git_info_path = exp_dir / "git_info.txt"
            with open(git_info_path, 'w') as f:
                f.write(f"Hash: {git_hash}\n")
                f.write(f"Branch: {git_branch}\n")
                f.write(f"Date: {experiment.created_at}\n")

        # Register in database
        self.db[exp_id] = experiment.to_dict()
        self._save_db()

        return experiment

    def update_experiment(self, exp_id: str, **kwargs):
        """Update experiment metadata."""
        if exp_id not in self.db:
            raise ValueError(f"Experiment {exp_id} not found")

        self.db[exp_id].update(kwargs)
        self._save_db()

    def finish_experiment(
        self,
        exp_id: str,
        final_metrics: Dict[str, float],
        best_checkpoint: str = None,
    ):
        """
        Mark experiment as completed and save final results.
        
        Args:
            exp_id: Experiment ID
            final_metrics: Final validation/test metrics
            best_checkpoint: Path to best checkpoint
        """
        if exp_id not in self.db:
            raise ValueError(f"Experiment {exp_id} not found")

        # Update database
        self.db[exp_id]['status'] = 'completed'
        self.db[exp_id]['final_metrics'] = final_metrics
        if best_checkpoint:
            self.db[exp_id]['best_checkpoint'] = str(best_checkpoint)
        self._save_db()

        # Save summary to experiment directory
        exp_dir = Path(self.db[exp_id]['dir'])
        summary_path = exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'experiment_id': exp_id,
                'status': 'completed',
                'final_metrics': final_metrics,
                'best_checkpoint': str(best_checkpoint) if best_checkpoint else None,
            }, f, indent=2)

    def mark_failed(self, exp_id: str, error: str):
        """Mark experiment as failed."""
        if exp_id not in self.db:
            return

        self.db[exp_id]['status'] = 'failed'
        self.db[exp_id]['error'] = error
        self._save_db()

    def find_experiments(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list:
        """
        Find experiments matching criteria.
        
        Args:
            model: Filter by model name
            dataset: Filter by dataset name
            status: Filter by status (running, completed, failed)
            tag: Filter by tag
        
        Returns:
            List of experiment dicts
        """
        results = []
        for exp_id, exp_data in self.db.items():
            if model and exp_data['model'] != model:
                continue
            if dataset and exp_data['dataset'] != dataset:
                continue
            if status and exp_data['status'] != status:
                continue
            if tag and exp_data.get('tag') != tag:
                continue
            results.append(exp_data)

        return results

    def get_best_checkpoint(
        self,
        model: str,
        dataset: str,
        metric: str = 'dice',
    ) -> Optional[Path]:
        """
        Find best checkpoint for model/dataset combination.
        
        Args:
            model: Model name
            dataset: Dataset name
            metric: Metric to use for selection (default: dice)
        
        Returns:
            Path to best checkpoint or None
        """
        experiments = self.find_experiments(model=model, dataset=dataset, status='completed')

        if not experiments:
            return None

        # Find experiment with best metric
        best_exp = max(
            experiments,
            key=lambda e: e.get('final_metrics', {}).get(metric, float('-inf'))
        )

        checkpoint = best_exp.get('best_checkpoint')
        return Path(checkpoint) if checkpoint else None

    def list_experiments(self, verbose: bool = False) -> list:
        """List all experiments."""
        if verbose:
            return list(self.db.values())
        else:
            return [
                {
                    'id': exp['id'],
                    'model': exp['model'],
                    'dataset': exp['dataset'],
                    'status': exp['status'],
                    'created_at': exp['created_at'],
                }
                for exp in self.db.values()
            ]
