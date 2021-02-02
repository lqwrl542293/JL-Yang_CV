import os
from dataset.ussd import USSDataset, USSDValDataset, USSDTrainInform


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    data_dir = os.path.abspath(os.path.join('../dataset', dataset))
    print(data_dir)

    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
    dataset_list = train_data_list
    val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
    inform_data_file = os.path.join('../dataset/inform', dataset + '_inform.pkl')

    dataCollect = USSDTrainInform(data_dir, 6, train_set_file=dataset_list,
                                  inform_data_file=inform_data_file)

    datas = dataCollect.collectDataAndSave()
    if datas is None:
        print("error while pickling data. Please check.")


if __name__ == '__main__':
    build_dataset_train('ussd', 0, 0, 'train', 0, 0, 0)
