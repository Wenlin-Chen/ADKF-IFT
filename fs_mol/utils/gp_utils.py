import torch
import gpytorch

from gpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy


class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, kernel, ard_num_dims=None):
        #Set the likelihood noise and enable/disable learning
        likelihood.noise_covar.raw_noise.requires_grad = False
        likelihood.noise_covar.noise = torch.tensor(0.1)
        super().__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()

        ## Linear kernel
        if kernel=='linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif kernel=='rbf' or kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        ## Matern kernel (52)
        elif kernel=='matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims))
        ## Polynomial (p=1)
        elif kernel=='poli1':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif kernel=='poli2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        elif kernel=='cossim':# or kernel=='bncossim':
        ## Cosine distance and BatchNorm Cosine distance
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.covar_module.base_kernel.variance = 1.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPLayer(gpytorch.models.ApproximateGP):

    def __init__(self, train_x, train_y, kernel, ard_num_dims=None):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()

        ## Linear kernel
        if kernel=='linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif kernel=='rbf' or kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        ## Matern kernel (52)
        elif kernel=='matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims))
        ## Polynomial (p=1)
        elif kernel=='poli1':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif kernel=='poli2':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        elif kernel=='cossim':# or kernel=='bncossim':
        ## Cosine distance and BatchNorm Cosine distance
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.covar_module.base_kernel.variance = 1.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")

        #self.set_train_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def set_train_data(self, inputs, targets, strict=False):
        
        assert torch.allclose(inputs, self.variational_strategy.inducing_points, atol=1e-5, rtol=1e-5)

        if inputs is not None and torch.is_tensor(inputs):
            inputs = (inputs,)
        if inputs is not None and not all(torch.is_tensor(train_input) for train_input in inputs):
            raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")
            
        self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in inputs)
        self.train_targets = targets
