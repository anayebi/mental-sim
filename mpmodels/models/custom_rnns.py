import numpy as np
import torch
from torch import nn

__all__ = [
    "CTRNN",
]

#################################################################################
# Pytorch implementations provided by Christopher J. Cueva
# continuous time recurrent neural network
# Tau * dah/dt = -ah + Wahh @ f(ah) + Wahx @ x + bah
#
# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)􏰩
# h[t] = f(ah[t]) + bhneverlearn[t], if t > 0
# y[t] = Wyh @ h[t] + by  output

# parameters to be learned: Wahh, Wahx, Wyh, bah, by, ah0(optional).
# In this implementation h[0] = f(ah[0]) with no noise added to h[0] except potentially through ah[0]
# constants that are not learned: dt, Tau, bhneverlearn
# Equation 1 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
class CTRNN(nn.Module):  # class CTRNN inherits from class torch.nn.Module
    def __init__(
        self,
        dim_input,
        dim_recurrent,
        dim_output,
        preset_random_init=True,
        Wahx=None,
        Wahh=None,
        Wyh=None,
        bah=None,
        by=None,
        nonlinearity="retanh",
        ah0=None,
        LEARN_ah0=False,
        LEARN_OUTPUTWEIGHT=True,
        LEARN_OUTPUTBIAS=True,
        dt=1,
        Tau=10,
    ):
        super().__init__()  # super allows you to call methods of the superclass in your subclass
        self.preset_random_init = preset_random_init
        self.fc_x2ah = nn.Linear(dim_input, dim_recurrent)  # Wahx @ x + bah
        self.fc_h2ah = nn.Linear(dim_recurrent, dim_recurrent, bias=False)  # Wahh @ h
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output)  # y = Wyh @ h + by
        self.numparameters = (
            dim_recurrent ** 2
            + dim_recurrent * dim_input
            + dim_recurrent
            + dim_output * dim_recurrent
            + dim_output
        )  # number of learned parameters in model
        self.dt = dt
        self.Tau = Tau

        if self.preset_random_init:
            Wahx = torch.randn(dim_recurrent, dim_input) / np.sqrt(dim_input)
            Wahh = (
                1.5 * torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
            )
            Wyh = torch.randn(dim_output, dim_recurrent)
            bah = torch.randn(dim_recurrent)
            by = torch.randn(dim_output)
            ah0 = torch.randn(dim_recurrent)

        # ------------------------------
        # initialize the biases bah and by
        if bah is not None:
            self.fc_x2ah.bias = torch.nn.Parameter(
                torch.squeeze(bah)
            )  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        if by is not None:
            self.fc_h2y.bias = torch.nn.Parameter(
                torch.squeeze(by)
            )  # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        # ------------------------------
        if (
            LEARN_OUTPUTBIAS == False
        ):  # this must go after the line self.fc_h2y.bias = torch.nn.Parameter(torch.squeeze(by)) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_OUTPUTBIAS = False then by does not change during gradient descent learning
            self.fc_h2y.bias.requires_grad = False  # y = Wyh @ h + by
            self.numparameters = (
                dim_recurrent ** 2
                + dim_recurrent * dim_input
                + dim_recurrent
                + dim_output * dim_recurrent
            )  # number of learned parameters in model
        # ------------------------------
        # initialize input(Wahx), recurrent(Wahh), output(Wyh) weights
        if Wahx is not None:
            self.fc_x2ah.weight = torch.nn.Parameter(Wahx)  # Wahx @ x + bah
        if Wahh is not None:
            self.fc_h2ah.weight = torch.nn.Parameter(Wahh)  # Wahh @ h
        if Wyh is not None:
            self.fc_h2y.weight = torch.nn.Parameter(Wyh)  # y = Wyh @ h + by
        # ------------------------------
        if (
            LEARN_OUTPUTWEIGHT == False
        ):  # this must go after the line self.fc_h2y.weight = torch.nn.Parameter(Wyh) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_OUTPUTWEIGHT = False then Wyh does not change during gradient descent learning
            self.fc_h2y.weight.requires_grad = False  # y = Wyh @ h + by
            self.numparameters = (
                self.numparameters - dim_output * dim_recurrent
            )  # number of learned parameters in model
        # ------------------------------
        # set the nonlinearity for h
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        # self.nonlinearity = lambda x: f(x, nonlinearity)
        self.nonlinearity = nonlinearity
        # ------------------------------
        # set the initial state ah0
        if ah0 is None:
            self.ah0 = torch.nn.Parameter(
                torch.zeros(dim_recurrent), requires_grad=False
            )  # (dim_recurrent,) tensor
        else:
            self.ah0 = torch.nn.Parameter(
                ah0, requires_grad=False
            )  # (dim_recurrent,) tensor
        if LEARN_ah0:
            # self.ah0 = self.ah0.requires_grad=True# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.ah0 = torch.nn.Parameter(
                self.ah0, requires_grad=True
            )  # (dim_recurrent,) tensor
            self.numparameters = (
                self.numparameters + dim_recurrent
            )  # number of learned parameters in model
        # ------------------------------

    # output y for all numT timesteps
    def forward(
        self, input, bhneverlearn=None
    ):  # nn.Linear expects inputs of size (numtrials, *, dim_input) where * is optional and could be numT
        if (
            len(input.shape) == 2
        ):  # if input has size (numT, dim_input) because there is only a single trial then add a singleton dimension
            input = input[None, :, :]  # (numtrials, numT, dim_input)
            if bhneverlearn is not None:
                bhneverlearn = bhneverlearn[
                    None, :, :
                ]  # (numtrials, numT, dim_recurrent)

        dt = self.dt
        Tau = self.Tau
        numtrials, numT, dim_input = input.shape
        ah = self.ah0.repeat(
            numtrials, 1
        )  # (numtrials, dim_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        h = computef(
            ah, self.nonlinearity
        )  # h0, this implementation doesn't add noise to h0
        hstore = []  # (numtrials, numT, dim_recurrent)
        for t in range(numT):
            ah = ah + (dt / Tau) * (
                -ah + self.fc_h2ah(h) + self.fc_x2ah(input[:, t])
            )  # ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)
            h = computef(ah, self.nonlinearity)
            if bhneverlearn is not None:
                h += bhneverlearn[
                    :, t, :
                ]  # bhneverlearn has shape (numtrials, numT, dim_recurrent)
            hstore.append(h)  # hstore += [h]
        hstore = torch.stack(
            hstore, dim=1
        )  # (numtrials, numT, dim_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (numtrials, *, dim_recurrent) where * means any number of additional dimensions
        return (
            self.fc_h2y(hstore),
            hstore,
        )  # (numtrials, numT, dim_output/dim_recurrent) tensor, y = Wyh @ h + by


#%%#############################################################################
#                      compute specified nonlinearity
# -----------------------------------------------------------------------------
def computef(IN, string, *args):  # ags[0] is the slope for string='tanhwithslope'
    if string == "linear":
        F = IN
        return F
    elif string == "logistic":
        F = 1 / (1 + torch.exp(-IN))
        return F
    elif string == "smoothReLU":  # smoothReLU or softplus
        F = torch.log(1 + torch.exp(IN))  # always greater than zero
        return F
    elif string == "ReLU":  # rectified linear units
        # F = torch.maximum(IN,torch.tensor(0))
        F = torch.clamp(IN, min=0)
        return F
    elif string == "swish":  # swish or SiLU (sigmoid linear unit)
        # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        # Elfwing et al. 2017 "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"
        # Ramachandran et al. 2017 "Searching for activation functions"
        sigmoid = 1 / (1 + torch.exp(-IN))
        F = torch.mul(
            IN, sigmoid
        )  # x*sigmoid(x), torch.mul performs elementwise multiplication
        return F
    elif (
        string == "mish"
    ):  # Misra 2019 "Mish: A Self Regularized Non-Monotonic Neural Activation Function
        F = torch.mul(
            IN, torch.tanh(torch.log(1 + torch.exp(IN)))
        )  # torch.mul performs elementwise multiplication
        return F
    elif (
        string == "GELU"
    ):  # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        F = 0.5 * torch.mul(
            IN,
            (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2 / np.pi)) * (IN + 0.044715 * IN ** 3)
                )
            ),
        )  # fast approximating used in original paper
        # F = x.*normcdf(x,0,1);% x.*normcdf(x,0,1)  =  x*0.5.*(1 + erf(x/sqrt(2)))
        # figure; hold on; x = linspace(-5,5,100); plot(x,x.*normcdf(x,0,1),'k-'); plot(x,0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3))),'r--')
        return F
    elif (
        string == "ELU"
    ):  # exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1
        inegativeIN = IN < 0
        F = IN.clone()
        F[inegativeIN] = alpha * (torch.exp(IN[inegativeIN]) - 1)
        return F
    elif string == "tanh":
        F = torch.tanh(IN)
        return F
    elif string == "tanhwithslope":
        a = args[0]
        F = torch.tanh(
            a * IN
        )  # F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)
        return F
    elif string == "tanhlecun":  # LeCun 1998 "Efficient Backprop"
        F = 1.7159 * torch.tanh(
            2 / 3 * IN
        )  # F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)
        return F
    elif string == "lineartanh":
        # F = torch.minimum(torch.maximum(IN,torch.tensor(-1)),torch.tensor(1))# -1(x<-1), x(-1<=x<=1), 1(x>1)
        F = torch.clamp(IN, min=-1, max=1)
        return F
    elif string == "retanh":  # rectified tanh
        F = torch.maximum(torch.tanh(IN), torch.tensor(0))
        return F
    elif string == "binarymeanzero":  # binary units with output values -1 and +1
        # F = (IN>=0) - (IN<0)# matlab code
        F = 1 * (IN >= 0) - 1 * (
            IN < 0
        )  # multiplying by 1 converts True to 1 and False to 0
        return F
    else:
        print("Unknown transfer function type")
