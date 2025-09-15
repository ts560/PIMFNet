import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
import lightning.pytorch as pl
from Net import PIMFuseModel
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, masks):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks.sum())

class PIMFuseTrainer(pl.LightningModule):
    def __init__(self, args,label_names):
        super().__init__()
        self.model = PIMFuseModel(hidden_size=args.hidden_size,num_classes=len(label_names))
        self.save_hyperparameters(args)
        self.pred_criterion = nn.CrossEntropyLoss(reduction='none')
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.triplet_loss = nn.TripletMarginLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.jsd = JSD()
        self.val_preds = {k: [] for k in ['final', 'vibration','pressure','phy']}
        self.val_labels = []
        self.val_pairs = []

        self.test_preds = []
        self.test_preds_phy = []
        self.test_labels = []
        self.test_pairs = []
        self.test_feats = {k: [] for k in ['feat_vibration_shared', 'feat_vibration_distinct','feat_pressure_shared', 'feat_pressure_distinct','y_30']}
        self.test_attns = []
        self.label_names = label_names

    def _compute_masked_pred_loss(self, input, target, mask):
        device = input.device
        mask = mask.to(device)
        target = target.to(device)
        return (self.pred_criterion(input, target) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_abs_cos_sim(self, x, y, mask):
        device = x.device
        y = y.to(device)
        mask = mask.to(device)
        return (self.alignment_cos_sim(x, y).abs() * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y) * mask).sum() / max(mask.sum(), 1e-6)

    def _masked_mse(self, x, y, mask):
        return (self.mse_loss(x, y).mean(dim=1) * mask).sum() / max(mask.sum(), 1e-6)

    def _disentangle_loss_jsd(self, model_output, pairs, log=True, mode='train'):
        mask = torch.ones_like(pairs)
        loss_sim_vibration = self._masked_abs_cos_sim(model_output['feat_vibration_shared'],
                                                model_output['feat_vibration_distinct'], pairs)
        loss_sim_pressure = self._masked_abs_cos_sim(model_output['feat_pressure_shared'],
                                                model_output['feat_pressure_distinct'], mask)

        jsd = self.jsd(model_output['feat_vibration_shared'].sigmoid(),model_output['feat_pressure_shared'].sigmoid(), pairs)

        loss_disentanglement = (self.hparams.lambda_disentangle_shared * jsd +
                                self.hparams.lambda_disentangle_vibration * loss_sim_vibration +
                                self.hparams.lambda_disentangle_pressure * loss_sim_pressure)
        if log:
            self.log_dict({
                f'disentangle_{mode}/vibration_distinct': loss_sim_vibration.detach(),
                f'disentangle_{mode}/pressure_distinct': loss_sim_pressure.detach(),
                f'disentangle_{mode}/shared_jsd': jsd.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=pairs.shape[0])

        return loss_disentanglement
    def _compute_prediction_losses(self, model_output, y_gt, pairs, log=True, mode='train'):
        mask = torch.ones_like(model_output['pred_final'][:, 0])
        loss_pred_final = self._compute_masked_pred_loss(model_output['pred_final'], y_gt, mask)
        loss_pred_pressure=self._compute_masked_pred_loss(model_output['pred_pressure'], y_gt, mask)
        loss_pred_vibration = self._compute_masked_pred_loss(model_output['pred_vibration'], y_gt, pairs)
        loss_pred_shared = self._compute_masked_pred_loss(model_output['pred_shared'], y_gt, mask)
        loss_pred_phy = self._compute_masked_pred_loss(model_output['y_pred_phy'], y_gt, mask)
        if log:
            self.log_dict({
                f'{mode}_loss/pred_final': loss_pred_final.detach(),
                f'{mode}_loss/pred_shared': loss_pred_shared.detach(),
                f'{mode}_loss/pred_pressure': loss_pred_pressure.detach(),
                f'{mode}_loss/pred_vibration': loss_pred_vibration.detach(),
                f'{mode}_loss/y_pred_phy': loss_pred_phy.detach(),
                'step': float(self.current_epoch)
            }, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])
        return loss_pred_final,loss_pred_pressure, loss_pred_vibration, loss_pred_shared,loss_pred_phy


    def pressure_pulsation_model(self, params, data):
        params=params.to(device)
        a = params[:, :12]
        b = params[:, 12:21]
        c = params[:, 21:30]

        t = torch.arange(0, 1, 1 / 5120, dtype=torch.float).to(device)
        t = t.view(1, -1)

        pressure_signal = torch.zeros(data.size(0), 5120).to(device)

        dict1 = {0: 10, 1: 20, 2: 29, 3: 39, 4: 49, 5: 59, 6: 68, 7: 79, 8: 89, 9: 98, 10: 107, 11: 120}
        dict2 = {0: 147, 1: 294, 2: 437, 3: 581, 4: 732, 5: 880, 6: 1027, 7: 1180, 8: 1324}#S1

        x_complex = data.type(torch.complex64)
        yf = torch.fft.fft(x_complex).to(device)
        for m in range(len(dict1)):
            angle = torch.atan2(-yf[:, dict1[m]].imag, yf[:, dict1[m]].real).to(device)
            angle = angle.view(-1, 1)
            pressure_signal += 2 / 5120 * a[:, m:m + 1] * torch.cos(2 * dict1[m] * np.pi * t - angle)
            pressure_signal=pressure_signal.to(device)
        for n in range(len(dict2)):
            angle = torch.atan2(-yf[:, dict2[n]].imag, yf[:, dict2[n]].real).to(device)
            angle = angle.view(-1, 1)

            a1 = 2 / 5120 * (b[:, n:n + 1] + c[:, n:n + 1] * torch.cos(2 * np.pi * 10 * t))
            a2 = torch.cos(2 * np.pi * dict2[n] * t - angle)
            pressure_signal += a1 * a2
            pressure_signal=pressure_signal.to(device)

        pressure_signal += torch.mean(data)
        pressure_signal=pressure_signal.to(device)
        return pressure_signal


    def custom_loss(self, model_output):
        ddd=model_output['S_P1']
        ddd = ddd.view(ddd.size(0), 5120)
        reconstructed_signal = self.pressure_pulsation_model(model_output['y_30'], ddd)
        reconstructed_signal = reconstructed_signal.unsqueeze(1)
        loss=torch.mean(torch.square(ddd - reconstructed_signal))
        loss_phy = loss.sum()
        return loss_phy

    def _compute_and_log_loss(self, model_output, y_gt, pairs, log=True, mode='train'):
        prediction_losses = self._compute_prediction_losses(model_output, y_gt, pairs, log, mode)
        loss_pred_final, loss_pred_pressure, loss_pred_vibration, loss_pred_shared,loss_pred_phy = prediction_losses

        loss_prediction = (self.hparams.lambda_pred_shared * loss_pred_shared +
                           self.hparams.lambda_pred_pressure * loss_pred_pressure +
                           self.hparams.lambda_pred_vibration * loss_pred_vibration+
                           self.hparams.lambda_pred_phy * loss_pred_phy)


        loss_prediction = loss_pred_final + loss_prediction

        loss_disentanglement = self._disentangle_loss_jsd(model_output, pairs, log, mode)

        loss_total = loss_prediction + loss_disentanglement
        epoch_log = {}

        raw_pred_loss_pressure = F.cross_entropy(model_output['pred_pressure'].data, y_gt, reduction='none')
        raw_pred_loss_vibration = F.cross_entropy(model_output['pred_vibration'].data, y_gt, reduction='none')
        raw_pred_loss_shared = F.cross_entropy(model_output['pred_shared'].data, y_gt, reduction='none')
        raw_pred_loss_phy = F.cross_entropy(model_output['y_pred_phy'].data, y_gt, reduction='none')

        pairs = pairs.unsqueeze(1)
        attn_weights = model_output['attn_weights']
        attn_pressure, attn_shared, attn_phy,attn_vibration = attn_weights[:, :, 0], attn_weights[:, :, 1], attn_weights[:, :, 2],attn_weights[:, :, 3]

        vibration_overweights_pressure = 2 * (raw_pred_loss_vibration < raw_pred_loss_pressure).long() - 1
        vibration_overweights_pressure=vibration_overweights_pressure.view(-1,1)
        loss_attn1 = pairs * F.margin_ranking_loss(attn_vibration, attn_pressure, vibration_overweights_pressure, reduction='none')
        loss_attn1 = loss_attn1.sum() / max(1e-6, loss_attn1[loss_attn1>0].numel())

        shared_overweights_pressure = 2 * (raw_pred_loss_shared < raw_pred_loss_pressure).long() - 1
        shared_overweights_pressure=shared_overweights_pressure.view(-1,1)
        loss_attn2 = pairs * F.margin_ranking_loss(attn_shared, attn_pressure, shared_overweights_pressure, reduction='none')
        loss_attn2 = loss_attn2.sum() / max(1e-6, loss_attn2[loss_attn2>0].numel())

        shared_overweights_vibration = 2 * (raw_pred_loss_shared < raw_pred_loss_vibration).long() - 1
        shared_overweights_vibration=shared_overweights_vibration.view(-1,1)
        loss_attn3 = pairs * F.margin_ranking_loss(attn_shared, attn_vibration, shared_overweights_vibration, reduction='none')
        loss_attn3 = loss_attn3.sum() / max(1e-6, loss_attn3[loss_attn3>0].numel())
        
        phy_overweights_vibration = 2 * (raw_pred_loss_phy < raw_pred_loss_vibration).long() - 1
        phy_overweights_vibration=phy_overweights_vibration.view(-1,1)
        loss_attn4 = pairs * F.margin_ranking_loss(attn_phy, attn_vibration, phy_overweights_vibration, reduction='none')
        loss_attn4 = loss_attn4.sum() / max(1e-6, loss_attn4[loss_attn4>0].numel())

        phy_overweights_shared = 2 * (raw_pred_loss_phy < raw_pred_loss_shared).long() - 1
        phy_overweights_shared = phy_overweights_shared.view(-1, 1)
        loss_attn5 = pairs * F.margin_ranking_loss(attn_phy, attn_shared, phy_overweights_shared,reduction='none')
        loss_attn5 = loss_attn5.sum() / max(1e-6, loss_attn5[loss_attn5 > 0].numel())
        
        phy_overweights_pressure = 2 * (raw_pred_loss_phy < raw_pred_loss_pressure).long() - 1
        phy_overweights_pressure = phy_overweights_pressure.view(-1, 1)
        loss_attn6 = pairs * F.margin_ranking_loss(attn_phy, attn_pressure, phy_overweights_pressure,reduction='none')
        loss_attn6 = loss_attn6.sum() / max(1e-6, loss_attn6[loss_attn6 > 0].numel())

        loss_attn_ranking = (loss_attn1 + loss_attn2 + loss_attn3+loss_attn4 + loss_attn5 + loss_attn6) / 6

        loss_total = loss_total + self.hparams.lambda_attn_aux * loss_attn_ranking
        
        loss_phy = self.custom_loss(model_output)
        
        loss_total = loss_total+loss_phy
        
        epoch_log[f'{mode}_loss/attn_aux'] = loss_attn_ranking.detach()

        if log:
            epoch_log.update({
                f'{mode}_loss/total': loss_total.detach(),
                f'{mode}_loss/prediction': loss_prediction.detach(),
                'step': float(self.current_epoch)
            })
            self.log_dict(epoch_log, on_epoch=True, on_step=False, batch_size=y_gt.shape[0])

        return loss_total

    def _get_batch_data(self,batch):
        y,S_P,S_V,pairs,S_P1=batch
        y=torch.from_numpy(y).long().to(self.device)
        S_V = torch.from_numpy(S_V).float().to(self.device)
        S_P = torch.from_numpy(S_P).float().to(self.device)
        S_P1 = torch.from_numpy(S_P1).float().to(self.device)
        pairs = torch.FloatTensor(pairs).to(self.device)
        return y,S_P,S_V,pairs,S_P1

    def training_step(self, batch, batch_idx):
        y,S_P,S_V,pairs,S_P1 = self._get_batch_data(batch)
        out = self.model(pairs, S_V, S_P,S_P1)
        return self._compute_and_log_loss(out, y_gt=y, pairs=pairs)

    def validation_step(self, batch, batch_idx):
        y,S_P,S_V,pairs,S_P1 = self._get_batch_data(batch)
        out = self.model(pairs, S_V, S_P,S_P1)
        loss = self._compute_and_log_loss(out, y_gt=y, pairs=pairs, mode='val')
        pred_final = out['pred_final']
        self.val_preds['final'].append(pred_final)
        self.val_preds['vibration'].append(out['pred_vibration'])
        self.val_preds['pressure'].append(out['pred_pressure'])
        self.val_preds['phy'].append(out['y_pred_phy'])
        self.val_pairs.append(pairs)
        self.val_labels.append(y)
        return self._compute_masked_pred_loss(pred_final, y, torch.ones(y.size(0), 1))
    def on_validation_epoch_end(self):
        for name in ['final','pressure','vibration','phy']:
            y_gt = torch.concat(self.val_labels, dim=0)
            preds = torch.concat(self.val_preds[name], dim=0)
            if name == 'vibration':
                pairs = torch.concat(self.val_pairs, dim=0)
                y_gt = y_gt[pairs==1]
                preds = preds[pairs==1, :]
            preds_classes = torch.argmax(preds, dim=1)
            accuracy = accuracy_score(y_gt.cpu().numpy(), preds_classes.cpu().numpy())
            if name == 'final':
                self.log('Val_Accuracy', accuracy, logger=False, prog_bar=True)
            log_dict = {
                 'step': float(self.current_epoch),
                 f'val_Accuracy_avg/{name}': accuracy,

            }
            for i in range(y_gt.max().item() + 1):
                accuracy_per_class = accuracy_score((y_gt == i).cpu().numpy(), (preds_classes == i).cpu().numpy())
                log_dict[f'val_Accuracy_per_class_{name}/{self.label_names[i]}'] = accuracy_per_class
                print(f'val_Accuracy_per_class_{name}/{self.label_names[i]}: {accuracy_per_class}')
            self.log_dict(log_dict)

        for k in self.val_preds:
            self.val_preds[k].clear()
        self.val_pairs.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        y, S_P, S_V, pairs,S_P1 = self._get_batch_data(batch)
        out = self.model(pairs,S_V,S_P,S_P1)
        pred_final = out['pred_final']
        self.test_preds.append(pred_final)
        self.test_labels.append(y)
        self.test_pairs.append(pairs)
        self.test_attns.append(out['attn_weights'])
        for k in self.test_feats:
            self.test_feats[k].append(out[k].cpu())

    def on_test_epoch_end(self):
        y_gt = torch.concat(self.test_labels, dim=0)
        preds = torch.concat(self.test_preds, dim=0)
        preds_classes = torch.argmax(preds, dim=1)
        accuracy = accuracy_score(y_gt.cpu().numpy(), preds_classes.cpu().numpy())
        self.test_results = {
             'accuracy': accuracy,
             'preds': preds,
             'preds_classes': preds_classes,
             'y_gt': y_gt
        }
        for k in self.test_feats:
            self.test_results[k] = torch.concat(self.test_feats[k], dim=0)
            self.test_feats[k].clear()
        self.test_labels.clear()
        self.test_preds.clear()
        self.test_pairs.clear()
        self.test_preds_phy.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer














