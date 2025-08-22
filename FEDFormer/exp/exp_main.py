import os
import time
import warnings
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import shap
from torch.utils.data import Subset, DataLoader
import pickle


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        model = model.to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, use_full_data=False):
        data_set, data_loader = data_provider(self.args, flag, use_full_data=use_full_data)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.train_model_on_data(setting, train_loader, vali_data, vali_loader)
        return self.model

    def train_model_on_data(self, setting, train_loader, vali_data, vali_loader):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, vali_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                pred = pred_data.inverse_transform(pred[0])
                preds.append(pred)

        print('Calculating SHAP values...')
        explanation = self.calculate_shap(setting, to_explain = (batch_x, batch_x_mark, dec_inp, batch_y_mark))

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        prediction_timestamps = pred_data.df_stamp['date'].values[-self.args.pred_len:]
        prediction_cols = pred_data.columns_to_predict


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #torch.save(self.model.state_dict(), "best_models/" + 'checkpoint.pth')

        data_dict = {
            'predictions': preds,
            'timestamps': prediction_timestamps,
            'columns': prediction_cols,
        }
        with open(folder_path + "prediction.pkl", "wb") as f:
            pickle.dump(data_dict, f)
                # Save SHAP values

        with open(folder_path + 'shap_explanation_full.pkl', 'wb') as f:
            pickle.dump(explanation, f)

        return

    def calculate_shap(self, setting, n_background=100, to_explain=None):
        train_data, train_loader = self._get_data(flag='train')

        # Load best model if needed
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        # --- Prepare background from train_loader ---
        background_batches = []
        total_samples = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            background_batches.append((batch_x, batch_x_mark, dec_inp, batch_y_mark))
            total_samples += batch_x.shape[0]
            if total_samples >= n_background:
                break

        # Stack background tensors
        batch_x_bg = torch.cat([x[0] for x in background_batches], dim=0)
        batch_x_mark_bg = torch.cat([x[1] for x in background_batches], dim=0)
        dec_inp_bg = torch.cat([x[2] for x in background_batches], dim=0)
        batch_y_mark_bg = torch.cat([x[3] for x in background_batches], dim=0)

        sizes = [
            batch_x_bg.shape[-1],
            batch_x_mark_bg.shape[-1],
            dec_inp_bg.shape[-1],
            batch_y_mark_bg.shape[-1]
        ]
        batch_x_to_exp, batch_x_mark_to_exp, dec_inp_to_exp, batch_y_mark_to_exp = to_explain
        # Prepare encoder inputs
        background = torch.cat([batch_x_bg, batch_x_mark_bg], dim=-1)
        to_explain = torch.cat([batch_x_to_exp, batch_x_mark_to_exp], dim=-1)

        # Build wrapper and set decoder parts
        wrapped_model = FullModelWrapper(
            self.model, sizes[:2],
            dec_inp_bg, batch_y_mark_bg,
            dec_inp_to_exp, batch_y_mark_to_exp
        ).to(self.device)

        # Set mode for background
        wrapped_model.set_mode('background')
        explainer = shap.GradientExplainer(wrapped_model, background)

        # Set mode for to_explain
        wrapped_model.set_mode('explain')
        explanation = explainer(to_explain)

        return explanation

    def cross_validate(self, setting, k_folds=5):
        """
        Perform k-fold cross-validation.
        """
        results = []
        # Load the full dataset
        full_dataset, _ = self._get_data(flag='train', use_full_data=True)
        indices = np.arange(len(full_dataset))
        kf = TimeSeriesSplit(n_splits=k_folds)

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"Starting Fold {fold + 1}/{k_folds}")
            print(f"Train len: {len(train_idx)}, Validation indices: {len(val_idx)}")
            # Create train and validation subsets
            self.model = self._build_model()  # Rebuild model for each fold
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            # Create DataLoaders for the subsets
            train_loader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.args.batch_size, shuffle=False)

            model, val_loss = self.train_model_on_data(setting, train_loader=train_loader, vali_data=val_subset, vali_loader=val_loader)
            results.append(val_loss)

        # Print overall results
        avg_loss = np.mean(results)
        print(f"Cross-Validation Results: {results}")
        print(f"Average Validation Loss: {avg_loss:.4f}")
        return avg_loss
    

class FullModelWrapper(torch.nn.Module):
    def __init__(self, model, sizes, dec_inp_bg, batch_y_mark_bg, dec_inp_explain, batch_y_mark_explain):
        super().__init__()
        self.model = model
        self.sizes = sizes  # list of feature sizes for each input
        self.dec_inp_bg = dec_inp_bg
        self.batch_y_mark_bg = batch_y_mark_bg
        self.dec_inp_explain = dec_inp_explain
        self.batch_y_mark_explain = batch_y_mark_explain
        self.mode = 'background'  # default mode

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x):
        # x: [batch, seq_len, total_features]
        batch_size, seq_len, _ = x.shape
        idx = 0
        splits = []
        for size in self.sizes[:2]:  # Only encoder inputs
            splits.append(x[:, :, idx:idx+size])
            idx += size
        batch_x, batch_x_mark = splits

        # Select decoder inputs based on mode
        if self.mode == 'background':
            dec_inp = self.dec_inp_bg.expand(batch_size, -1, -1)
            batch_y_mark = self.batch_y_mark_bg.expand(batch_size, -1, -1)
        else:
            dec_inp = self.dec_inp_explain.expand(batch_size, -1, -1)
            batch_y_mark = self.batch_y_mark_explain.expand(batch_size, -1, -1)

        output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return output.view(batch_size, -1)