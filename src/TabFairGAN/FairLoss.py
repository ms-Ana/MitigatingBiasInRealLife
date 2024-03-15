import torch
from torch import nn


def get_gen_loss(crit_fake_pred):
    gen_loss = -1.0 * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss


def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


class FairLossFunc(nn.Module):
    def __init__(
        self,
        S_start_index,
        Y_start_index,
        underpriv_index,
        priv_index,
        undesire_index,
        desire_index,
    ):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._S_start_index : self._S_start_index + 2]
        # print(x[0,64])
        I = x[:, self._Y_start_index : self._Y_start_index + 2]
        # disp = (torch.mean(G[:,1]*I[:,1])/(x[:,65].sum())) - (torch.mean(G[:,0]*I[:,0])/(x[:,64].sum()))
        # disp = -1.0 * torch.tanh(torch.mean(G[:,0]*I[:,1])/(x[:,64].sum()) - torch.mean(G[:,1]*I[:,1])/(x[:,65].sum()))
        # gen_loss = -1.0 * torch.mean(crit_fake_pred)
        disp = -1.0 * lamda * (
            torch.mean(G[:, self._underpriv_index] * I[:, self._desire_index])
            / (x[:, self._S_start_index + self._underpriv_index].sum())
            - torch.mean(G[:, self._priv_index] * I[:, self._desire_index])
            / (x[:, self._S_start_index + self._priv_index].sum())
        ) - 1.0 * torch.mean(crit_fake_pred)
        # print(disp)
        return disp
