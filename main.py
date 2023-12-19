import argparse
from utils import load_config
from experiment import ExperimentTracker

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image classification experiments')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)
    
    # Overwrite config with CLI arguments if provided
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    experiment = ExperimentTracker(config)
    experiment.run()

if __name__ == '__main__':
    main()