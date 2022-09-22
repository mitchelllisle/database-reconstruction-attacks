import argparse

from dra.attack import DatabaseConstructionAttack
from dra.logger import logger


def main():
    logger.info('starting DatabaseReconstructionAttack runner...')
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input', type=str, help='an input CSV containing the block stats')
    parser.add_argument('solution', type=str, help='n solution CSV showing the final stats')
    parser.add_argument('output', type=str, help='an output CSV to put the reconstructed database')
    parser.add_argument('--min-age', type=int, help='the minimum age constraint', default=0)
    parser.add_argument('--max-age', type=int, help='the maximum age constraint', default=115)

    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace):
    logger.info('running DRA...')
    attack = DatabaseConstructionAttack(
        stats_file=args.input, solutions_file=args.solution, min_age=args.min_age, max_age=args.max_age
    )
    model = attack.run()
    model.to_csv(args.output, index=False)
    logger.info(f'DRA complete. Accuracy achieved {round(attack.check_accuracy())}%')
    logger.info(f'reconstructed database: {args.output}')


if __name__ == '__main__':
    main()
