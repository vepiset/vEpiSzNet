# -*-coding:utf-8-*-
import copy
import time
import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from timm.loss.binary_cross_entropy import BinaryCrossEntropy

from lib.core.base_trainer.model import Net
from lib.core.base_trainer.metric import *
from lib.dataset.dataietr import AlaskaDataIter

from timm.utils.model_ema import ModelEmaV3

from train_config import config as cfg
from lib.utils.logger import logger


#
class Train(object):

    def __init__(self,
                 train_df,
                 val_df,
                 test_df=None,
                 fold=0):

        self.train_df=train_df
        self.val_df=val_df
        self.test_df=test_df

        self.train_generator = AlaskaDataIter(train_df, training_flag=True, shuffle=False,data_set_flag='TRAIN')
        self.train_ds = DataLoader(self.train_generator,
                                   cfg.TRAIN.batch_size,
                                   num_workers=cfg.TRAIN.process_num,
                                   shuffle=True,
                                   pin_memory=True,
                                   persistent_workers=True,
                                   prefetch_factor=2)

        self.val_generator = AlaskaDataIter(val_df, training_flag=False, shuffle=False,data_set_flag='VAL')

        self.val_ds = DataLoader(self.val_generator,
                                 cfg.TRAIN.validatiojn_batch_size,
                                 num_workers=cfg.TRAIN.process_num,
                                 shuffle=False,
                                 pin_memory=True,
                                 persistent_workers=True,
                                 prefetch_factor=2)

        if self.test_df is not None:
            self.test_generator = AlaskaDataIter(test_df, training_flag=False, shuffle=False,data_set_flag='TEST')

            self.test_ds = DataLoader(self.test_generator,
                                     cfg.TRAIN.validatiojn_batch_size,
                                     num_workers=cfg.TRAIN.process_num,
                                     shuffle=False,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     prefetch_factor=4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.fold = fold

        self.init_lr = cfg.TRAIN.init_lr
        self.warup_step = cfg.TRAIN.warmup_step
        self.epochs = cfg.TRAIN.epoch
        self.batch_size = cfg.TRAIN.batch_size
        self.l2_regularization = cfg.TRAIN.weight_decay_factor
        self.early_stop = cfg.MODEL.early_stop
        self.accumulation_step = cfg.TRAIN.accumulation_batch_size // cfg.TRAIN.batch_size
        self.gradient_clip = cfg.TRAIN.gradient_clip
        self.save_dir = cfg.MODEL.model_path
        self.fp16 = cfg.TRAIN.mix_precision

        self.model = Net().to(self.device)
        self.load_weight()

        if 'Adamw' in cfg.TRAIN.opt:
            self.optimizer = self.configure_optimizers()
        else:
            raise NotImplementedError

        self.model = nn.DataParallel(self.model)

        self.ema_model = ModelEmaV3(self.model, decay=0.999)

        self.iter_num = 0

        if cfg.TRAIN.lr_scheduler == 'cos':
            logger.info('lr_scheduler.CosineAnnealingLR')
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        self.epochs,
                                                                        eta_min=1.e-6)
        else:
            raise NotImplementedError

        self.scaler = torch.cuda.amp.GradScaler()

    def configure_optimizers(self, ):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay

        def get_wd_params(model: nn.Module):
            # 分别存储不同组件的参数
            transformer_decay = list()
            transformer_no_decay = list()
            other_decay = list()
            other_no_decay = list()

            for name, param in model.named_parameters():
                print('checking {}'.format(name))
                if hasattr(param, 'requires_grad') and not param.requires_grad:
                    continue

                # 判断是否为Qwen参数
                is_transformer_param = 'transformer' in name

                # 判断是否需要weight decay
                if 'weight' in name and 'norm' not in name and 'bn' not in name:
                    if is_transformer_param:
                        transformer_decay.append(param)
                        print('checking {}'.format(name), 'qwen with decay')
                    else:
                        other_decay.append(param)
                        print('checking {}'.format(name), 'other with decay')
                else:
                    if is_transformer_param:
                        transformer_no_decay.append(param)
                        print('checking {}'.format(name), 'qwen no decay')
                    else:
                        other_no_decay.append(param)
                        print('checking {}'.format(name), 'other no decay')

            return transformer_decay, transformer_no_decay, other_decay, other_no_decay

        # 修改后的优化器创建
        transformer_decay, transformer_no_decay, other_decay, other_no_decay = get_wd_params(self.model)

        optim_groups = [
            # Qwen参数 - 学习率 * 0.1
            {"params": transformer_decay, "weight_decay": self.l2_regularization, "lr": self.init_lr * 0.1},
            {"params": transformer_no_decay, "weight_decay": 0.0, "lr": self.init_lr * 0.1},
            # 其他参数 - 正常学习率
            {"params": other_decay, "weight_decay": self.l2_regularization, "lr": self.init_lr},
            {"params": other_no_decay, "weight_decay": 0.0, "lr": self.init_lr},
        ]

        # 过滤掉空的参数组
        optim_groups = [group for group in optim_groups if len(group["params"]) > 0]

        optimizer = torch.optim.AdamW(optim_groups)
        return optimizer
    def resample_train_data(self,):

        # 对 0样本随机采样一个，
        # 分离正常样本和异常样本
        train_data=copy.deepcopy( self.train_df)
        normal_samples = train_data[train_data['task1'] == 'NORMAL']
        abnormal_samples = train_data[train_data['task1'] != 'NORMAL']

        # 计算1/10的数量并进行采样
        sample_size = len(normal_samples) // 5
        normal_sampled = normal_samples.sample(n=sample_size,random_state=self.iter_num)

        # 合并采样后的正常样本和所有异常样本
        train_data = pd.concat([normal_sampled, abnormal_samples], ignore_index=True)

        self.train_generator = AlaskaDataIter(train_data, training_flag=True, shuffle=False, data_set_flag='TRAIN')
        self.train_ds = DataLoader(self.train_generator,
                                   cfg.TRAIN.batch_size,
                                   num_workers=cfg.TRAIN.process_num,
                                   shuffle=True,
                                   pin_memory=True,
                                   persistent_workers=True,
                                   prefetch_factor=2)
    def custom_loop(self):

        def distributed_train_epoch(epoch_num):

            self.resample_train_data()
            train_meters = {
                'task1': AUCMeter(),
                'task2': AUCMeter(),
                'task3': AUCMeter(),
                'loss': AverageMeter()
            }
            self.model.train()

            for waves, images, task1_label, task2_label, task3_label in self.train_ds:

                if epoch_num < 10:
                    # execute warm up in the first epochs
                    if self.warup_step > 0:
                        if self.iter_num < self.warup_step:
                            warmup_factor = self.iter_num / float(self.warup_step)

                            # 如果是第一次warmup，保存原始目标学习率
                            if not hasattr(self, 'original_lrs'):
                                self.original_lrs = []
                                for param_group in self.optimizer.param_groups:
                                    if 'lr' in param_group:
                                        self.original_lrs.append(param_group['lr'])
                                    else:
                                        self.original_lrs.append(self.init_lr)

                            # 应用warmup
                            lr_info = []
                            for i, (param_group, original_lr) in enumerate(
                                    zip(self.optimizer.param_groups, self.original_lrs)):
                                param_group['lr'] = warmup_factor * original_lr
                                lr_info.append(f"Group {i}: {param_group['lr']:.6f}")

                            logger.info('warm up with learning rates: [%s]' % (', '.join(lr_info)))

                start = time.time()

                data = waves.to(self.device).float()
                images = images.to(self.device).float()
                task1_label = task1_label.to(self.device).long()
                task2_label = task2_label.to(self.device).long()
                task3_label = task3_label.to(self.device).long()
                batch_size = data.shape[0]

                with torch.cuda.amp.autocast(enabled=self.fp16, dtype=torch.bfloat16):
                    outputs = self.model(data, images, [task1_label, task2_label, task3_label])

                    current_loss = outputs['loss']
                    train_meters['loss'].update(current_loss.detach().item(), batch_size)
                    train_meters['task1'].update(outputs['target1'].detach(), outputs['prediction1'].detach())
                    train_meters['task2'].update(outputs['target2'].detach(), outputs['prediction2'].detach())
                    train_meters['task3'].update(outputs['target3'].detach(), outputs['prediction3'].detach())

                self.scaler.scale(current_loss).backward()

                if ((self.iter_num + 1) % self.accumulation_step) == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip, norm_type=2)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.ema_model.update(self.model, self.iter_num)

                self.iter_num += 1
                time_cost_per_batch = time.time() - start

                images_per_sec = cfg.TRAIN.batch_size / time_cost_per_batch

                if self.iter_num % cfg.TRAIN.log_interval == 0:
                    log_message = '[fold %d], ' \
                                  'Train Step %d, ' \
                                  'summary_loss: %.6f, ' \
                                  'time: %.6f, ' \
                                  'speed %d images/persec' % (
                                      self.fold,
                                      self.iter_num,
                                      train_meters['loss'].avg,
                                      time.time() - start,
                                      images_per_sec)
                    logger.info(log_message)

            return train_meters

        def distributed_val_epoch(epoch_num):

            eval_meters = {
                'task1': AUCMeter(),
                'task2': AUCMeter(),
                'task3': AUCMeter(),
                'task2_pure': AUCMeter(),
                'task3_pure': AUCMeter(),
                'loss':AverageMeter()
            }

            self.model.eval()
            self.ema_model.module.eval()
            with torch.cuda.amp.autocast(enabled=self.fp16, dtype=torch.bfloat16):
                with torch.no_grad():
                    for (waves, images, task1_label, task2_label, task3_label) in tqdm(self.val_ds):
                        data = waves.to(self.device).float()
                        images = images.to(self.device).float()
                        task1_label = task1_label.to(self.device).long()
                        task2_label = task2_label.to(self.device).long()
                        task3_label = task3_label.to(self.device).long()
                        batch_size = data.shape[0]

                        outputs = self.ema_model(data, images, [task1_label,
                                                                task2_label,
                                                                task3_label])


                        if torch.any(torch.isnan(outputs['loss'])):
                            print(outputs['loss'],' there is nan')
                            print(task1_label)
                            print(task2_label)
                            print(task3_label)

                        eval_meters['task1'].update(outputs['target1'].detach(),
                                                   outputs['prediction1'].detach())

                        eval_meters['task2'].update(outputs['target2'].detach(),
                                                   outputs['prediction2'].detach())

                        eval_meters['task3'].update(outputs['target3'].detach(),
                                                   outputs['prediction3'].detach())

                        mask2 = task2_label != 0
                        mask3 = task3_label != 0


                        prediction2_pure = outputs['prediction2'].clone()
                        prediction3_pure = outputs['prediction3'].clone()
                        prediction2_pure[:, 0] *= 0
                        prediction3_pure[:, 0] *= 0

                        eval_meters['task2_pure'].update(outputs['target2'].detach()[mask2],
                                                        prediction2_pure.detach()[mask2])

                        eval_meters['task3_pure'].update(outputs['target3'].detach()[mask3],
                                                        prediction3_pure.detach()[mask3])
                        eval_meters['loss'].update(outputs['loss'].detach().item(), batch_size)


            return eval_meters
        def distributed_test_epoch(epoch_num):

            test_meters = {
                'task1': AUCMeter(),
                'task2': AUCMeter(),
                'task3': AUCMeter(),
                'task2_pure': AUCMeter(),
                'task3_pure': AUCMeter(),
                'loss':AverageMeter()
            }

            self.model.eval()
            self.ema_model.module.eval()
            with torch.cuda.amp.autocast(enabled=self.fp16, dtype=torch.bfloat16):
                with torch.no_grad():
                    for (waves, images, task1_label, task2_label, task3_label) in tqdm(self.test_ds):
                        data = waves.to(self.device).float()
                        images = images.to(self.device).float()
                        task1_label = task1_label.to(self.device).long()
                        task2_label = task2_label.to(self.device).long()
                        task3_label = task3_label.to(self.device).long()
                        batch_size = data.shape[0]

                        outputs = self.ema_model(data, images, [task1_label,
                                                                task2_label,
                                                                task3_label])


                        if torch.any(torch.isnan(outputs['loss'])):
                            print(outputs['loss'],' there is nan')
                            print(task1_label)
                            print(task2_label)
                            print(task3_label)

                        test_meters['task1'].update(outputs['target1'].detach(),
                                                   outputs['prediction1'].detach())

                        test_meters['task2'].update(outputs['target2'].detach(),
                                                   outputs['prediction2'].detach())

                        test_meters['task3'].update(outputs['target3'].detach(),
                                                   outputs['prediction3'].detach())

                        mask2 = task2_label != 0
                        mask3 = task3_label != 0


                        prediction2_pure = outputs['prediction2'].clone()
                        prediction3_pure = outputs['prediction3'].clone()
                        prediction2_pure[:, 0] *= 0
                        prediction3_pure[:, 0] *= 0

                        test_meters['task2_pure'].update(outputs['target2'].detach()[mask2],
                                                        prediction2_pure.detach()[mask2])

                        test_meters['task3_pure'].update(outputs['target3'].detach()[mask3],
                                                        prediction3_pure.detach()[mask3])
                        test_meters['loss'].update(outputs['loss'].detach().item(), batch_size)


            return test_meters
        best_distance = 0.
        not_improvement = 0
        for epoch in range(self.epochs):

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
            logger.info('learning rate: [%f]' % (lr))
            t = time.time()

            train_meters = distributed_train_epoch(epoch)
            train_epoch_log_message = '[fold %d], ' \
                                      '[RESULT]: TRAIN. Epoch: %d,' \
                                      ' summary_loss: %.5f,' \
                                      ' time:%.5f' % (
                                          self.fold,
                                          epoch,
                                          train_meters['loss'].avg,
                                          (time.time() - t))
            logger.info(train_epoch_log_message)
            train_meters['task1'].report()
            train_meters['task2'].report()
            train_meters['task3'].report()

            if epoch % cfg.TRAIN.test_interval == 0:
                eval_meters = distributed_val_epoch(epoch)

                val_epoch_log_message = ('[fold %d], ' \
                                         '[RESULT]: VAL. Epoch: %d,' \
                                         ' val_loss: %.5f,' \
                                         ' val_task1_pr_auc: %.5f,' \
                                         ' val_task2_pr_auc: %.5f,' \
                                         ' val_task3_pr_auc: %.5f,' \
                                         ' val_task1_acc: %.5f,' \
                                         ' val_task2_acc: %.5f,' \
                                         ' val_task3_acc: %.5f,' \
                                         ' time:%.5f' % (
                                             self.fold,
                                             epoch,
                                             eval_meters['loss'].avg,
                                             eval_meters['task1'].prauc,
                                             eval_meters['task2_pure'].prauc,
                                             eval_meters['task3_pure'].prauc,
                                             eval_meters['task1'].acc,
                                             eval_meters['task2_pure'].acc,
                                             eval_meters['task3_pure'].acc,
                                             (time.time() - t)))
                logger.info(val_epoch_log_message)
                logger.info('val: NORMAL, Seizure:')
                eval_meters['task1'].report()
                logger.info('val: NORMAL, GE, Focal:')
                eval_meters['task2'].report()
                logger.info('val: NORMAL, CP, FLE, OLE, TLE, Q1, Q2, Q3, Q4, Q5:')
                eval_meters['task3'].report()

                logger.info('val: GE, Focal:')
                eval_meters['task2_pure'].report()
                logger.info('val: CP, FLE, OLE, TLE, Q1, Q2, Q3, Q4, Q5:')
                eval_meters['task3_pure'].report()

                # eval_meters['task1'].plot_auc_curves(prefix='fold%d_val'%(self.fold),epoch=epoch)

                eval_meters['task1'].save_data(prefix='val_task1',fold=self.fold,epoch=epoch)
                eval_meters['task2'].save_data(prefix='val_task2',fold=self.fold,epoch=epoch)
                eval_meters['task3'].save_data(prefix='val_task3',fold=self.fold,epoch=epoch)
                if self.test_df is not None:
                    test_meters = distributed_test_epoch(epoch)

                    test_epoch_log_message = ('[fold %d], ' \
                                             '[RESULT]: TEST. Epoch: %d,' \
                                             ' test_loss: %.5f,' \
                                             ' test_task1_pr_auc: %.5f,' \
                                             ' test_task2_pr_auc: %.5f,' \
                                             ' test_task3_pr_auc: %.5f,' \
                                             ' test_task1_acc: %.5f,' \
                                             ' test_task2_acc: %.5f,' \
                                             ' test_task3_acc: %.5f,' \
                                             ' time:%.5f' % (
                                                 self.fold,
                                                 epoch,
                                                 test_meters['loss'].avg,
                                                 test_meters['task1'].prauc,
                                                 test_meters['task2_pure'].prauc,
                                                 test_meters['task3_pure'].prauc,
                                                 test_meters['task1'].acc,
                                                 test_meters['task2_pure'].acc,
                                                 test_meters['task3_pure'].acc,
                                                 (time.time() - t)))
                    logger.info(test_epoch_log_message)
                    logger.info('val: NORMAL, Seizure:')
                    test_meters['task1'].report()
                    logger.info('val: NORMAL, GE, Focal:')
                    test_meters['task2'].report()
                    logger.info('val: NORMAL, CP, FLE, OLE, TLE, Q1, Q2, Q3, Q4, Q5:')
                    test_meters['task3'].report()

                    logger.info('val: GE, Focal:')
                    test_meters['task2_pure'].report()
                    logger.info('val: CP, FLE, OLE, TLE, Q1, Q2, Q3, Q4, Q5:')
                    test_meters['task3_pure'].report()
                    test_meters['task1'].plot_auc_curves(prefix='fold%d_test'%(self.fold),epoch=epoch)

                    test_meters['task1'].save_data(prefix='test_task1', fold=self.fold, epoch=epoch)
                    test_meters['task2'].save_data(prefix='test_task2', fold=self.fold, epoch=epoch)
                    test_meters['task3'].save_data(prefix='test_task3', fold=self.fold, epoch=epoch)


            all_metric = (eval_meters['task1'].prauc + eval_meters['task2'].prauc + eval_meters['task3'].prauc) / 3.
            if cfg.TRAIN.lr_scheduler == 'cos':
                self.scheduler.step()
            else:
                self.scheduler.step(all_metric)

            # save model
            if not os.access(cfg.MODEL.model_path, os.F_OK):
                os.mkdir(cfg.MODEL.model_path)

            #### save the model every end of epoch
            current_model_saved_name = self.save_dir + '/fold%d_epoch_%d_prauc_%.6f_loss_%.6f.pth' % (self.fold,
                                                                                                      epoch,
                                                                                                      all_metric,
                                                                                                      eval_meters['loss'].avg)

            logger.info('A model saved to %s' % current_model_saved_name)
            torch.save(self.ema_model.module.module.state_dict(), current_model_saved_name)

            if all_metric > best_distance:
                best_distance = all_metric
                logger.info(' best prauc value update as %.6f' % (best_distance))
                logger.info(' bestmodel update as %s' % (current_model_saved_name))
                not_improvement = 0

            else:
                not_improvement += 1

            if not_improvement >= self.early_stop:
                logger.info(' best metric score not improvement for %d, break' % (self.early_stop))
                break

            torch.cuda.empty_cache()

    def load_weight(self):
        if cfg.MODEL.pretrained_model is not None:
            state_dict = torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
