from datasets.dataset_loader import load_dataset

dataset_name = 'cifar10'  # in ['mnist', 'cifar10', 'cifar100']


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)


if __name__ == '__main__':
    main()
