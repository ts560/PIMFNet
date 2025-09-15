import argparse
from copy import deepcopy
import pickle
import torch
import lightning as L
import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from data_processing import dataProcessing,standardize_with_train,collect_data1
from data_processing_3 import valid_test_slice,CustomTensorDataset,dataProcessing_3
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
    parser.add_argument('--hidden_size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(5)


    L.seed_everything(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Train_P, Train_V, Train_Yf, Test_P, Test_V, Test_Yf=dataProcessing(file_path="D:/process date/data_2_S1")
    Train_V_std, Test_V_std = standardize_with_train(Train_V, Test_V)
    Train_P_std, Test_P_std = standardize_with_train(Train_P, Test_P)
    train_x, test_x, train_y, test_y = dataProcessing_3(file_path="D:/process date/data_1_S1")
    train_x = train_x.reshape(-1, train_x.shape[2])
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = test_x.reshape(-1, test_x.shape[2])
    test_x = np.expand_dims(test_x, axis=-1)
    Train_x=np.concatenate((Train_P_std,Train_V_std,train_x),axis=2)
    Test_x=np.concatenate((Test_P_std,Test_V_std,test_x),axis=2)
    Train_X, Train_Y, Val_X, Val_Y = valid_test_slice(Train_x, Train_Yf, 0.25)
    Train_X = torch.tensor(Train_X, dtype=torch.float)
    Val_X = torch.tensor(Val_X, dtype=torch.float)
    Test_X = torch.tensor(Test_x, dtype=torch.float)
    Train_Y = torch.tensor(Train_Y, dtype=torch.long)
    Val_Y = torch.tensor(Val_Y, dtype=torch.float)
    Test_Y = torch.tensor(Test_Yf, dtype=torch.long)
    Test_Xp = Test_X.clone()
    columns_to_zero = [1,2,3]
    Test_Xp[:, :, columns_to_zero] = torch.tensor(0).float()
    Test_X_partial =Test_Xp
    train_dataset = CustomTensorDataset(Train_X,Train_Y)
    valid_dataset = CustomTensorDataset(Val_X,Val_Y)
    test_dataset = CustomTensorDataset(Test_X,Test_Y)
    test_partial_dataset = CustomTensorDataset(Test_X_partial,Test_Y)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collect_data1,persistent_workers=True,
                                  shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collect_data1,
                                persistent_workers=True,shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collect_data1,persistent_workers=True,
                                 shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    test_partial_dataloader = DataLoader(test_partial_dataset, batch_size=args.batch_size, collate_fn=collect_data1,persistent_workers=True,
                                         shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)

    model = PIMFuseTrainer(args=args,label_names=["0","1","2","3","4","5","6","7","8"])
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
