# --*-- coding:utf-8 --*--

from datasets import get_cifar10, get_mnist




def main():
    dataset = 'cifar10'
	
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()
    elif dataset == 'stock':
    	pass


if __name__ == '__main__':
    main()
