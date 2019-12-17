from utils.data_preprocessing import SampleImageArrangement

DATATYPE = 'cifar10'
NORMTYPE = 'normalize'
METHOD = 'holdout'
SPLITRATE = 0.2
K = 5

if __name__ == '__main__':
    sample = SampleImageArrangement(imgtype=DATATYPE)

    learns, preds = sample.get_datasets()
    print('learning images: {}'.format(learns[0].shape))
    print('learning labels: {}'.format(learns[1].shape))
    print('predict images: {}'.format(preds[0].shape))
    print('predict labels: {}'.format(preds[1].shape))
    print('predict names: {}'.format(preds[2].shape))
    class_dict = sample.get_classdict(inverse=True)
    print(class_dict)

    trains, vals = sample.split_dataset(
            X=learns[0], y=learns[1], method=METHOD,
            splitrate=SPLITRATE, K=K
            )
    print('train images: {}'.format(trains[0].shape))
    print('train labels: {}'.format(trains[1].shape))
    print('val images: {}'.format(vals[0].shape))
    print('val labels: {}'.format(vals[1].shape))
