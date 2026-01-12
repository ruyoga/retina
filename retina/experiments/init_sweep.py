import wandb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'experiments'))

from retina.modules.sweep_config import get_sweep_config


def main():
    print("\nAvailable sweep types:")
    print("  1. resnet50_focal     ")
    print("  2. efficientnet_focal ")
    print("  3. densenet_focal     ")
    print("  4. resnet50_quick     ")
    print("  5. all_models         ")

    choice = input("\nEnter sweep type (1/2/3/4/5)").strip()

    sweep_types = {
        '1': 'resnet50_focal',
        '2': 'efficientnet_focal',
        '3': 'densenet_focal',
        '4': 'resnet50_quick',
        '5': 'all_models',
        '': 'resnet50_focal'
    }

    sweep_type = sweep_types.get(choice, 'resnet50_focal')


    sweep_config = get_sweep_config(sweep_type)

    project = input("\nEnter W&B project name").strip()
    if not project:
        project = 'retina-binary-focal-sweep'

    print("\nInitializing sweep...")

    try:
        sweep_id = wandb.sweep(
            sweep_config,
            project=project
        )

        print(f"\nSweep ID: {sweep_id}")
        print(f"Project: {project}")

        import wandb as wb
        entity = wb.Api().default_entity

        print(f"\nTo run the sweep:")
        print(f"  wandb agent {entity}/{project}/{sweep_id}")
        print(f"\nTo run limited number of trials:")
        print(f"  wandb agent --count 20 {entity}/{project}/{sweep_id}")
        print(f"\nView results at:")
        print(f"  https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")

        with open('sweep_id.txt', 'w') as f:
            f.write(f"{sweep_id}\n")
            f.write(f"project: {project}\n")
            f.write(f"type: {sweep_type}\n")

    except Exception as e:
        print(f"\n✗ Error initializing sweep: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())