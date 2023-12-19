import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset
from tqdm import tqdm
from itertools import chain

from core.model.replay.inc_net import FOSTERNet
from core.utils.utils import count_parameters, tensor2numpy, accuracy
from .finetune import Finetune
from ...data import DataManager

# Please refer to https://github.com/G-U-N/ECCV22-FOSTER for the full source code to reproduce foster.



class FOSTER(Finetune):
    def __init__(self, backbone,feat_dim, num_class, **args):
        super().__init__(backbone,feat_dim, num_class,**args)
        self.args = args
        self._network = FOSTERNet(args, False)
        self._snet = None
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        # todo (FOSTER) add methods to update per_cls_weights
        self.per_cls_weights = None
        self.is_teacher_wa = args["is_teacher_wa"]
        self.is_student_wa = args["is_student_wa"]
        self.lambda_okd = args["lambda_okd"]
        self.wa_value = args["wa_value"]
        self.init_cls_num = args["init_cls_num"]
        self.inc_cls_num = args["inc_cls_num"]
        self.oofc = args["oofc"].lower()

        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._old_network = None

        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"]
        self.samples_new_cls = args["samples_new_class"]

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        if task_idx == 0:
            self._memory_size = buffer.buffer_size
        self.buffer = buffer
        self.train_loader = train_loader

        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + (self.init_cls_num if task_idx == 0 else self.inc_cls_num)
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        print(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False
            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]

            if isinstance(self.beta1,list):
                beta1=self.beta1[self._cur_task-1]
            else:
                beta1=self.beta1
            effective_num = 1.0 - np.power(beta1, cls_num_list)
            per_cls_weights = (1.0 - beta1) / np.array(effective_num)
            per_cls_weights = (
                    per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            print("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                        self._known_classes:, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"

        self._network.to(self._device)



    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        if task_idx > 0:
            # feature compression
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes,
                    self._total_classes - self._known_classes,
                    self.wa_value,
                )
            else:
                print("do not weight align teacher!")

            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = (
                    per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            print("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            #todo (FOSTER) fix concat bugs
            merged_loaders = list(chain.from_iterable(test_loaders))
            self._feature_compression(train_loader, merged_loaders)
        # else:
        #     PATH = "./state_dict_model_200iniEpochs.pth"
        #     torch.save(self.state_dict(),PATH)

        self._known_classes = self._total_classes
        #print("Exemplar size: {}".format(self.exemplar_size))
        # self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def observe(self, data):

        if self._cur_task > 0:
            correct, total = 0, 0
            inputs, targets = data["image"], data["label"]
            inputs, targets = inputs.to(
                self._device, non_blocking=True
            ), targets.to(self._device, non_blocking=True)
            outputs = self._network(inputs)
            logits, fe_logits, old_logits = (
                outputs["logits"],
                outputs["fe_logits"],
                outputs["old_logits"].detach(),
            )
            loss_clf = F.cross_entropy(logits / self.per_cls_weights, targets)
            loss_fe = F.cross_entropy(fe_logits, targets)
            loss_kd = self.lambda_okd * _KD_loss(
                logits[:, : self._known_classes], old_logits, self.args["T"]
            )
            loss = loss_clf + loss_fe + loss_kd
            # optimizer.zero_grad()
            # loss.backward()
            # todo(FOSTER) 没有进行backward
            # if self.oofc == "az":
            #     for i, p in enumerate(self._network_module_ptr.fc.parameters()):
            #         if i == 0:
            #             p.grad.data[
            #             self._known_classes:,
            #             : self._network_module_ptr.out_dim,
            #             ] = torch.tensor(0.0)
            # elif self.oofc != "ft":
            #     assert 0, "not implemented"
            # optimizer.step()
            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)
            # scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            return outputs["logits"], train_acc, loss
        else:
            correct, total = 0, 0
            inputs, targets = data["image"], data["label"]
            inputs, targets = inputs.to(
                self._device, non_blocking=True
            ), targets.to(self._device, non_blocking=True)
            logits = self._network(inputs)["logits"]
            loss = F.cross_entropy(logits, targets)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            return logits, train_acc, loss

    def after_backward(self):
        if self._cur_task > 0:

            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.grad.data[
                        self._known_classes:,
                        : self._network_module_ptr.out_dim,
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
    def inference(self, data):
        inputs, targets = data["image"], data["label"]
        y_pred,y_true = [],[]
        inputs = inputs.to(self._device, non_blocking=True)
        with torch.no_grad():
            outputs = self._network(inputs)["logits"]
        predicts = torch.argmax(outputs, dim=1)
        y_pred.append(predicts.cpu().numpy())
        y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        acc = (y_pred == y_true).sum() * 100 / len(y_true)
        return outputs, acc

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()
    def eval(self):
        self._network.eval()

    def parameters(self) :
        return [p for p in self._network.parameters() if p.requires_grad == True]


    def _feature_compression(self, train_loader, test_loader):
        self._snet = FOSTERNet(self.args, False)
        self._snet.update_fc(self._total_classes)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        self._snet_module_ptr.convnets[0].load_state_dict(
            self._network_module_ptr.convnets[0].state_dict()
        )
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["compression_epochs"]
        )
        self._network.eval()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.0
            correct, total = 0, 0
            for i, batch in enumerate(train_loader):
                inputs, targets = batch["image"], batch["label"]
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                dark_logits = self._snet(inputs)["logits"]
                with torch.no_grad():
                    outputs = self._network(inputs)
                    logits, old_logits, fe_logits = (
                        outputs["logits"],
                        outputs["old_logits"],
                        outputs["fe_logits"],
                    )
                loss_dark = self.BKD(dark_logits, logits, self.args["T"])
                loss = loss_dark
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(dark_logits[: targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            print(info)
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value,
            )
        else:
            print("do not weight align student!")

        self._snet.eval()
        y_pred, y_true = [], []
        for _, batch in enumerate(test_loader):
            inputs,targets = batch["image"], batch["label"]
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)["logits"]
            predicts = torch.argmax(outputs,dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = (y_pred == y_true).sum() * 100 / len(y_true)
        print("darknet eval: ")
        print("CNN top1 curve: {}".format(cnn_accy))

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        return self.samples_new_cls
    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

    @property
    def backbone(self):
        return self._network.convnets

    # from Baselearner:
    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, batch in enumerate(loader):
            inputs, targets = batch["image"], batch["label"]
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
