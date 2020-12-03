import torch
import torch.nn as nn

class SVM_SVC(nn.modules.Module):
    def __init__(self, input_num, class_num, channels=3):
        super(SVM_SVC, self).__init__()

        self.class_num = class_num
        self.input_num = input_num * channels
        self.svm_model = nn.Linear(self.input_num, self.class_num)

    def forward(self, x):
        x = x.reshape(-1, self.input_num)
        return self.svm_model(x)

    def convert_from_one_hot(self, outputs):
        res = torch.ones(outputs.shape[0])
        outputs = outputs.tolist()
        for idx, item in enumerate(outputs):
            max_value, max_idx = -10000., -1
            for i_idx, i_item in enumerate(item):
                if max_value < i_item:
                    max_value = i_item
                    max_idx = i_idx
            res[idx] = max_idx
        return res

class SVM_Loss(nn.modules.Module):
    def __init__(self):
        super(SVM_Loss, self).__init__()

    def forward(self, output, label):
        N = output.shape[0]

        res = output.t() * label
        clamped_res = torch.clamp(1 - res, min=0)

        return torch.sum(clamped_res) / N
