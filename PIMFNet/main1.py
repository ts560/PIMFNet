import argparse
from copy import deepcopy
import pickle
import torch
import lightning as L
import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from data_processing_2 import dataProcessing_2,CustomTensorDataset,collect_data
from data_processing_3 import dataProcessing_3,scalar_stand,valid_test_slice
from trainer import PIMFuseTrainer
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_pressure', type=float, default=1)
    parser.add_argument('--lambda_disentangle_vibration', type=float, default=1)
    parser.add_argument('--lambda_pred_pressure', type=float, default=1)
    parser.add_argument('--lambda_pred_vibration', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--lambda_pred_phy', type=float, default=1)
    parser.add_argument('--lambda_attn_aux', type=float, default=1)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    # set number of threads allowed
    torch.set_num_threads(5)

    # set seed
    L.seed_everything(args.seed)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_x, test_x, train_y, test_y = dataProcessing_3(file_path="D:/process date/data_1_S2")
    train_x = train_x.reshape(2688, 5120, 1)
    test_x = test_x.reshape(672, 5120, 1)
    train_x1, test_x1, train_y1, test_y1 = dataProcessing_2(file_path="D:/process date/data_2_S2")
    Train_X1, Test_X1 = scalar_stand(train_x1, test_x1)
    Train_X2 = np.concatenate((Train_X1, train_x), axis=2)
    Test_X2 = np.concatenate((Test_X1, test_x), axis=2)
    Train_X, Train_Y, Val_X, Val_Y = valid_test_slice(Train_X2, train_y1, 0.25)
    Train_X, Val_X, Test_X = Train_X[:, np.newaxis, :], Val_X[:, np.newaxis, :], Test_X2[:, np.newaxis, :]
    Train_X = torch.tensor(Train_X, dtype=torch.float)
    Val_X = torch.tensor(Val_X, dtype=torch.float)
    Test_X = torch.tensor(Test_X, dtype=torch.float)
    Train_Y = torch.tensor(Train_Y, dtype=torch.long)
    Val_Y = torch.tensor(Val_Y, dtype=torch.float)
    Test_Y = torch.tensor(test_y1, dtype=torch.long)
    Test_Xp = Test_X.clone()
    columns_to_zero = [1, 2, 3]
    Test_Xp[:, :, :, columns_to_zero] = torch.tensor(0).float()
    Test_X_partial =Test_Xp
    train_dataset = CustomTensorDataset(Train_X,Train_Y)
    valid_dataset = CustomTensorDataset(Val_X,Val_Y)
    test_dataset = CustomTensorDataset(Test_X,Test_Y)
    test_partial_dataset = CustomTensorDataset(Test_X_partial,Test_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collect_data,persistent_workers=True,
                                  shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collect_data,
                                persistent_workers=True,shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collect_data,persistent_workers=True,
                                 shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    test_partial_dataloader = DataLoader(test_partial_dataset, batch_size=args.batch_size, collate_fn=collect_data,persistent_workers=True,
                                         shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)

    model = PIMFuseTrainer(args=args,label_names=["0","1","2","3","4","5","6","7","8","9","10","11","12","13"])
    callback_metric = 'val_Accuracy_avg/final'
    early_stop_callback = EarlyStopping(monitor=callback_metric,
                                        min_delta=0.00,
                                        patience=args.patience,
                                        verbose=False,
                                        mode="max")
    trainer = L.Trainer(devices=[0],
                        accelerator='gpu',
                        max_epochs=args.epochs,
                        min_epochs=min(args.epochs, 10),
                        log_every_n_steps=20,
                        callbacks=[early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    # do testing

    results = {
        'best_val_accuracy': trainer.callback_metrics['val_Accuracy_avg/final'].item(),
    }
    logpath = trainer.logger.log_dir
    trainer.loggers = None

    trainer.test(model=model, dataloaders=test_dataloader)
    results['paired_test_results'] = deepcopy(model.test_results)

    trainer.test(model=model, dataloaders=test_partial_dataloader)
    results['partial_test_results'] = deepcopy(model.test_results)

    with open(logpath + '/test_results', 'wb') as f:
        pickle.dump(results, f)



