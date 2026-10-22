import argparse

from training.engine import run_training


def main():
    parser = argparse.ArgumentParser(description='Train scalar-density baseline models.')
    parser.add_argument('--experiment', required=True, help='JSON experiment preset.')
    parser.add_argument('--data', required=True, help='Local EMMa paths JSON.')
    parser.add_argument('--name', help='Experiment name.')
    parser.add_argument('--output-root', default='logs')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--resume')
    args = parser.parse_args()

    run_training(
        args.experiment, args.data, expected_family='baseline', output_root=args.output_root,
        resume=args.resume,
        overrides={
            'name': args.name,
            'train.epochs': args.epochs,
            'train.seed': args.seed,
            'train.lr': args.lr,
        },
    )


if __name__ == '__main__':
    main()
