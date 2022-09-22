from dra.attack import DatabaseConstructionAttack
from dra.database import solution, stats


def main():
    attack = DatabaseConstructionAttack(stats)
    model = attack.run()
    print(model)
    print(attack.check_accuracy(model, solution))


if __name__ == '__main__':
    main()
