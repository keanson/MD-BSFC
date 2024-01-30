## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import warnings
from torch.distributions import MultivariateNormal
import warnings
# warnings.filterwarnings("error", category=UserWarning)

## Our packages
import gpytorch
from time import gmtime, strftime
import random
from configs import kernel_type
#Check if tensorboardx is installed
try:
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')

## Training CMD
#ATTENTION: to test each method use exaclty the same command but replace 'train.py' with 'test.py' or 'calibrate.py'
# CUB + data augmentation
#python3 train.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --train_aug --tau=1 --loss='ELBO' --steps=2 --seed=1
#python3 train.py --dataset="CUB" --method="CDKT" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug --tau=1 --loss='ELBO' --steps=2 --seed=1

class GD(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(GD, self).__init__(model_func, n_way, n_support)
        self.device = 'cuda:0'
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.kernel_type = kernel_type
        self.get_model_likelihood_mll() #Init model, likelihood
        
        self.normalize = True
        self.mu_q = []
        self.sigma_q = []
        
        # if(kernel_type=="cossim"):
        #     self.normalize=True
        # elif(kernel_type=="bncossim"):
        #     self.normalize=True
        #     latent_size = np.prod(self.feature_extractor.final_feat_dim)
        #     self.feature_extractor.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size))
        # else:
        #     # self.normalize=False
        #     pass
    
    def get_steps(self, steps):
        if steps == -1:
            self.STEPS = 'Annealing'
        else:
            self.STEPS = steps
    
    def get_temperature(self, temperature=1.):
        self.TEMPERATURE = temperature

    def get_negmean(self, mean=0.):
        if mean == 999:
            self.register_parameter("NEGMEAN", nn.Parameter(torch.zeros(1, device=self.device)))
            return True
        else:
            self.NEGMEAN = mean
            return False

    def get_loss(self, loss='ELBO'):
        self.LOSS = loss
        
    def get_kernel_type(self, kernel_type='bncossim'):
        self.kernel_type = kernel_type
        self.get_model_likelihood_mll()
        
    def init_summary(self):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M%S", gmtime())
            writer_path = "./log/" + time_string
            self.writer = SummaryWriter(log_dir=writer_path)
        
    def get_model_likelihood_mll(self, train_x_list=None, train_y_list=None):
        if(train_x_list is None): train_x_list=[torch.ones(100, 64).to(self.device)]*self.n_way
        if(train_y_list is None): train_y_list=[torch.ones(100).to(self.device)]*self.n_way
        model_list = list()
        for train_x, train_y in zip(train_x_list, train_y_list):
            model = Kernel(device=self.device, kernel=self.kernel_type)
            model_list.append(model)
        self.model = CombinedKernels(model_list)
        return self.model
    
    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).to(self.device)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).to(self.device)
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).to(self.device)
        y_query = np.repeat(range(self.n_way), self.n_query)

        with torch.no_grad():
            self.model.eval()
            self.feature_extractor.eval()
            
            z_support = self.feature_extractor.forward(x_support).detach()
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            support_outputs = self.model(z_support)
            
            support_mu, support_sigma = self.predict_gd(y_support, support_outputs, steps=30, rho=0.5)
            
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            
            q_posterior_list = []
            for c in range(len(self.model.kernels)):
                posterior = self.model.kernels[c].predict(z_query, z_support, support_mu[c], support_sigma[c])
                q_posterior_list.append(posterior)

            y_pred = self.montecarlo(q_posterior_list, times=10000, temperature=self.TEMPERATURE, return_logits=True) 
        return y_pred

    def train_loop(self, epoch, train_loader, optimizer, print_freq=10):
        if self.STEPS == 'Annealing':
            STEPS = 1 + epoch // 50
        else:
            STEPS = self.STEPS
        
        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way  = x.size(0)
            x_all = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]).to(self.device)
            y_all = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).to(self.device))
            x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).to(self.device)
            y_support = np.repeat(range(self.n_way), self.n_support)
            x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).to(self.device)
            y_query = np.repeat(range(self.n_way), self.n_query)
            x_train = x_all
            y_train = y_all

            self.model.train()
            self.feature_extractor.train()
            
            z_train = self.feature_extractor.forward(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
            output = self.model(z_train)

            lenghtscale = 0.0
            outputscale = 0.0
            meanscale = 0.0
            for idx, single_model in enumerate(self.model.kernels):
                if(single_model.covar_module.base_kernel.lengthscale is not None):
                    lenghtscale+=single_model.covar_module.base_kernel.lengthscale.mean().cpu().detach().numpy().squeeze()
                if(single_model.covar_module.outputscale is not None):
                    outputscale+=single_model.covar_module.outputscale.cpu().detach().numpy().squeeze()

            if(single_model.covar_module.base_kernel.lengthscale is not None): lenghtscale /= float(len(self.model.kernels))
            if(single_model.covar_module.outputscale is not None): outputscale /= float(len(self.model.kernels))
        
            if self.LOSS == 'ELBO':
                loss, logits = self.GradientDescentELBO(y=y_train, output=output, steps=STEPS, REQUIRES_GRAD=True, temperature=self.TEMPERATURE)
                bs = len(y_train)
            else:
                y_query = y_train.reshape(self.n_way, self.n_support + self.n_query)[:, :self.n_query].flatten()
                z_query = z_train.reshape(self.n_way, self.n_support + self.n_query, -1)[:, :self.n_query, :].reshape(self.n_query * self.n_way, -1)
                y_support = y_train.reshape(self.n_way, self.n_support + self.n_query)[:, self.n_query:].flatten()
                z_support = z_train.reshape(self.n_way, self.n_support + self.n_query, -1)[:, self.n_query:, :].reshape(self.n_support * self.n_way, -1)
                loss, logits = self.GDPredictiveLoglikelihood(y_support, z_support, y_query, z_query, steps=STEPS, REQUIRES_GRAD=False, times=10000, tau=self.TEMPERATURE)
                bs = len(y_train[:self.n_support * self.n_way])

            if optimizer.__class__.__name__ == "Adam" or optimizer.__class__.__name__ == "AdamW":
                try:
                    torch.nan_to_num(loss).backward() 
                    if not all([torch.isfinite(p.grad).all() for p in self.feature_extractor.parameters()]):
                        print("Nan in the gradients, skipping this iteration.")
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                except:
                    pass
            else:
                # Sophia
                torch.nan_to_num(loss).backward() 
                optimizer.step(bs=bs)
                optimizer.zero_grad(set_to_none=True)
                # setting k to 10 for hessian updates
                k = 20
                if i % k != k - 1:
                    pass
                else:
                    # update hessian EMA
                    z_train = self.feature_extractor.forward(x_train)
                    if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
                    output = self.model(z_train)
                    _, logits = self.MirrorPredictiveLoglikelihood(y_train[:self.n_support * self.n_way], z_train[:self.n_support * self.n_way], y_train[self.n_support * self.n_way:], z_train[self.n_support * self.n_way:], steps=STEPS, REQUIRES_GRAD=True, times=1000, tau=self.TEMPERATURE)
                    bs = len(y_train)
                    samp_dist = torch.distributions.Categorical(logits=logits)
                    y_sample = samp_dist.sample()
                    loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                    loss_sampled.backward()
                    optimizer.update_hessian()
                    optimizer.zero_grad(set_to_none=True)

            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss.item(), self.iteration)

            if i % print_freq==0:
                # if(self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                print('Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Lenghtscale {:f} | Loss {:f}'.format(epoch, i, len(train_loader), outputscale, lenghtscale, loss.item()))
            
            return loss.item()

    def correct(self, x):
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).to(self.device)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).to(self.device)
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).to(self.device)
        y_query = np.repeat(range(self.n_way), self.n_query)

        with torch.no_grad():
            self.model.eval()
            self.feature_extractor.eval()
            
            z_support = self.feature_extractor.forward(x_support).detach()
            if(self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
            support_outputs = self.model(z_support)
            
            support_mu, support_sigma = self.predict_gd(y_support, support_outputs, steps=30, rho=0.5)
            
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            
            q_posterior_list = []
            for c in range(len(self.model.kernels)):
                posterior = self.model.kernels[c].predict(z_query, z_support, support_mu[c], support_sigma[c])
                q_posterior_list.append(posterior)
            
            y_pred = self.montecarlo(q_posterior_list, times=10000, temperature=self.TEMPERATURE)     
            y_pred = y_pred.cpu().numpy() 
            top1_correct = np.sum(y_pred == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this

    def test_loop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            if(i % 100==0):
                acc_mean = np.mean(np.asarray(acc_all))
                print('Test | Batch {:d}/{:d} | Acc {:f}'.format(i, len(test_loader), acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)
        if(return_std): return acc_mean, acc_std
        else: return acc_mean

    def predict_gd(self, y, output, steps=10, rho=0.5):
        times = 1000
        REQUIRES_GRAD = False
        temperature = self.TEMPERATURE
        with torch.no_grad():
            y = torch.tensor(y).long()
            C = self.n_way
            # N, C
            Y = F.one_hot(y, num_classes = C)
            N = Y.shape[0]

            # variational inference params
            vi_mean = torch.zeros((C, N), requires_grad=REQUIRES_GRAD).to(self.device)
            vi_cov = torch.zeros((C, N, N), requires_grad=REQUIRES_GRAD).to(self.device)
            vi_cov += torch.eye(N).reshape(1, N, N).to(self.device)

            K_inv = torch.linalg.inv(output)
        
            # inner loop
            for step in range(steps):
                # get gradients (10) & (11)
                N = Y.shape[0]
                C = Y.shape[1]

                # draw samples
                samples_list = []
                for c in range(C):
                    samples = MultivariateNormal(vi_mean_ls[-1][c], scale_tril=psd_safe_cholesky(vi_cov_ls[-1][c])).rsample(torch.Size((times, )))
                    samples_list.append(samples)
                # times, N, C
                fn = torch.stack(samples_list).permute(1, 2, 0) / temperature
                # compute gradients
                grad_m = torch.mean(Y - torch.exp(fn) / torch.exp(fn).sum(dim=2, keepdim=True), dim=0)
                grad_v = 0.5 * torch.mean(torch.exp(2 * fn) / (torch.exp(fn).sum(dim=2, keepdim=True)) ** 2 - torch.exp(fn) / torch.exp(fn).sum(dim=2, keepdim=True), dim=0)

                grad_m = grad_m.T - (K_inv @ vi_mean_ls[-1].unsqueeze(2)).squeeze(2)
                grad_v = torch.diag_embed(grad_v.T) - 0.5 * (- torch.linalg.inv(vi_cov_ls[-1]) + K_inv)
                vi_mean = vi_mean_ls[-1] + rho * grad_m
                vi_cov = vi_cov_ls[-1] + rho * grad_v
            
                vi_mean_ls.append(vi_mean)
                vi_cov_ls.append(vi_cov)
        return vi_mean, vi_cov 
    
    def montecarlo(self, q_posterior_list, times=1000, temperature=1, return_logits=False):
        samples_list = []
        for posterior in q_posterior_list:
            samples = posterior.rsample(torch.Size((times, )))
            samples_list.append(samples)
        # classes, times, query points
        all_samples = torch.stack(samples_list)
        # times, classes, query points
        all_samples = all_samples.permute(1, 0, 2)
        if return_logits: return all_samples
        # compute logits
        C = all_samples.shape[1]
        all_samples = torch.exp(all_samples / temperature)
        all_samples = all_samples / all_samples.sum(dim=1, keepdim=True).repeat(1, C, 1)
        # classes, query points
        avg = all_samples.mean(dim=0)
        
        return torch.argmax(avg, dim=0)

    def GradientDescentELBO(self, y, output, steps=2, REQUIRES_GRAD=False, rho=0.01, times=10000, temperature=1):
        y = torch.tensor(y).long()
        N = (self.n_support + self.n_query) * self.n_way
        C = self.n_way
        # N, C
        Y = F.one_hot(y, num_classes = C)

        # variational inference params
        # vi_mean = torch.zeros((C, N), requires_grad=REQUIRES_GRAD).to(self.device)
        # vi_cov = torch.zeros((C, N, N), requires_grad=REQUIRES_GRAD).to(self.device)
        # vi_cov += torch.eye(N).reshape(1, N, N).to(self.device)
        vi_mean = torch.load('mean.pt')
        vi_cov = torch.load('cov.pt')

        K_inv = torch.linalg.inv(output)
        vi_mean_ls = [vi_mean]
        vi_cov_ls = [vi_cov]

        # gradient descent
        for step in range(steps):
            # get gradients (10) & (11)
            N = Y.shape[0]
            C = Y.shape[1]

            # draw samples
            samples_list = []
            for c in range(C):
                samples = MultivariateNormal(vi_mean_ls[-1][c], scale_tril=psd_safe_cholesky(vi_cov_ls[-1][c])).rsample(torch.Size((times, )))
                samples_list.append(samples)
            # times, N, C
            fn = torch.stack(samples_list).permute(1, 2, 0) / temperature
            # compute gradients
            grad_m = torch.mean(Y - torch.exp(fn) / torch.exp(fn).sum(dim=2, keepdim=True), dim=0)
            grad_v = 0.5 * torch.mean(torch.exp(2 * fn) / (torch.exp(fn).sum(dim=2, keepdim=True)) ** 2 - torch.exp(fn) / torch.exp(fn).sum(dim=2, keepdim=True), dim=0)

            grad_m = grad_m.T - (K_inv @ vi_mean_ls[-1].unsqueeze(2)).squeeze(2)
            grad_v = torch.diag_embed(grad_v.T) - 0.5 * (- torch.linalg.inv(vi_cov_ls[-1]) + K_inv)
            vi_mean = vi_mean_ls[-1] + rho * grad_m
            vi_cov = vi_cov_ls[-1] + rho * grad_v
        
            vi_mean_ls.append(vi_mean)
            vi_cov_ls.append(vi_cov)
        
        # compute ELBO
        samples_list = []
        for c in range(C):
            samples = MultivariateNormal(vi_mean_ls[-1][c], scale_tril=psd_safe_cholesky(vi_cov_ls[-1][c])).rsample(torch.Size((times, )))
            samples_list.append(samples)
        # times, N, C
        fn = torch.stack(samples_list).permute(1, 2, 0) / temperature
        ELBO = torch.sum(torch.mean(torch.sum(fn * Y, dim=2, keepdim=True) - torch.log(torch.sum(torch.exp(fn), dim=2, keepdim=True)), dim=0))
        ELBO2 = 0.5 * (torch.einsum('bii->b', K_inv @ vi_cov_ls[-1]) + (vi_mean_ls[-1].unsqueeze(1) @ K_inv @ vi_mean_ls[-1].unsqueeze(2)).squeeze() - N + torch.linalg.slogdet(output)[1] - torch.linalg.slogdet(vi_cov_ls[-1])[1] )
        ELBO -= torch.sum(ELBO2)
        return -ELBO, torch.mean(fn, dim=0)

    def GDPredictiveLoglikelihood(self, y_support, z_support, y_query, z_query, steps=2, REQUIRES_GRAD=False, times=1000, tau=1, rho=0.1):
        # with torch.no_grad():
        temperature = tau
        output = self.model(z_support)
        y = torch.tensor(y_support).long()
        y_query = torch.tensor(y_query).long().to(self.device)
        N = len(y_support)
        C = self.n_way

        Y = F.one_hot(y, num_classes = C)

        # variational inference params
        vi_mean = torch.zeros((C, N), requires_grad=REQUIRES_GRAD).to(self.device)
        vi_cov = torch.zeros((C, N, N), requires_grad=REQUIRES_GRAD).to(self.device)
        vi_cov += torch.eye(N).reshape(1, N, N).to(self.device)
        # natural params
        theta_tilde_1 = torch.zeros((N, C), requires_grad=REQUIRES_GRAD).to(self.device)
        theta_tilde_2 = torch.zeros((N, C), requires_grad=REQUIRES_GRAD).to(self.device)

        try:
            vi_mean = self.vi_mean.detach()
            vi_cov = self.vi_cov.detach()
        except:
            pass

        K_inv = torch.linalg.inv(output)

        vi_mean_ls = [vi_mean]
        vi_cov_ls = [vi_cov]
        
        # inner loop
        for step in range(steps):
            # get gradients (10) & (11)
            N = Y.shape[0]
            C = Y.shape[1]

            # draw samples
            samples_list = []
            for c in range(C):
                samples = MultivariateNormal(vi_mean_ls[-1][c], scale_tril=psd_safe_cholesky(vi_cov_ls[-1][c])).rsample(torch.Size((times, )))
                samples_list.append(samples)
            # times, N, C
            fn = torch.stack(samples_list).permute(1, 2, 0) / temperature
            # compute gradients
            grad_m = torch.mean(Y - torch.exp(fn) / torch.exp(fn).sum(dim=2, keepdim=True), dim=0)
            grad_v = 0.5 * torch.mean(torch.exp(2 * fn) / (torch.exp(fn).sum(dim=2, keepdim=True)) ** 2 - torch.exp(fn) / torch.exp(fn).sum(dim=2, keepdim=True), dim=0)

            grad_m = grad_m.T - (K_inv @ vi_mean_ls[-1].unsqueeze(2)).squeeze(2)
            grad_v = torch.diag_embed(grad_v.T) - 0.5 * (- torch.linalg.inv(vi_cov_ls[-1]) + K_inv)
            vi_mean = vi_mean_ls[-1] + rho * grad_m
            vi_cov = vi_cov_ls[-1] + rho * grad_v
        
            vi_mean_ls.append(vi_mean)
            vi_cov_ls.append(vi_cov)
        
        self.vi_mean = vi_mean_ls[-1].detach()
        self.vi_cov = vi_cov_ls[-1].detach()

        q_posterior_list = []
        for c in range(len(self.model.kernels)):
            posterior = self.model.kernels[c].predict(z_query, z_support, vi_mean_ls[-1][c], vi_cov_ls[-1][c])
            q_posterior_list.append(posterior)
        samples_list = []
        for posterior in q_posterior_list:
            samples = posterior.rsample(torch.Size((times, )))
            samples_list.append(samples)
        # classes, times, query points
        all_samples = torch.stack(samples_list).to(self.device)
        # times, classes, query points
        all_samples = all_samples.permute(1, 0, 2)
        # compute logits
        # classes, query points
        logits = (all_samples / temperature).mean(0)
        return nn.CrossEntropyLoss()(logits.T, y_query), logits
            
class Kernel(nn.Module):
    '''
    Parameters learned by the model:
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, device, kernel='rbf'):
        super().__init__()
        self.device = device
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = None
        
        ## Linear kernel
        if(kernel=='linear'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Matern kernel
        elif(kernel=='matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        ## Polynomial (p=1)
        elif(kernel=='poli1'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif(kernel=='poli2'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        elif(kernel=='cossim' or kernel=='bncossim'):
        ## Cosine distance and BatchNorm Cosine distancec
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.covar_module.base_kernel.variance = 1.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")
        self.covar_module = self.covar_module.to(self.device
                                                )

    def forward(self, x):
        covar_x = self.covar_module(x).evaluate()
        while not torch.all(torch.linalg.eigvals(covar_x).real > 1e-6).item():
            covar_x += 1e-6 * torch.eye(covar_x.shape[0], device=self.device)
        return covar_x
    
    def predict(self, z_query, z_support, support_mu, support_sigma, noise=0.1):
        K_lt = self.covar_module(z_support, z_query).evaluate()
        K_tt = self.covar_module(z_query).evaluate()
        covar_x = self.covar_module(z_support).evaluate()

        L = psd_safe_cholesky(covar_x)
        mean = K_lt.T @ torch.cholesky_solve(support_mu.unsqueeze(1), L).squeeze()
        covar = K_tt - K_lt.T @ torch.cholesky_solve(K_lt, L) + K_lt.T @ torch.cholesky_solve(support_sigma, L) @ torch.cholesky_solve(K_lt, L)
        try:
            return MultivariateNormal(mean, scale_tril=psd_safe_cholesky(covar))
        except:
            return MultivariateNormal(mean, scale_tril=psd_safe_cholesky(torch.eye(covar.shape[0], device=self.device)))
    
class CombinedKernels(nn.Module):
    def __init__(self, kernel_list) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(kernel_list)
    
    def forward(self, x):
        covar = []
        mean = []
        for kernel in self.kernels:
            covar_x = kernel(x)
            # mean.append(mean_x)
            covar.append(covar_x)
        return torch.stack(covar, dim=0)
    
    
def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        if A.dim() == 2:
            L = torch.linalg.cholesky(A, upper=upper, out=out)
            return L
        else:
            L_list = []
            for idx in range(A.shape[0]):
                L = torch.linalg.cholesky(A[idx], upper=upper, out=out)
                L_list.append(L)
            return torch.stack(L_list, dim=0)
    except:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(8):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                if Aprime.dim() == 2:
                    L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
                    warnings.warn(
                        f"A not p.d., added jitter of {jitter_new} to the diagonal",
                        RuntimeWarning,
                    )
                    return L
                else:
                    L_list = []
                    for idx in range(Aprime.shape[0]):
                        L = torch.linalg.cholesky(Aprime[idx], upper=upper, out=out)
                        L_list.append(L)
                    warnings.warn(
                        f"A not p.d., added jitter of {jitter_new} to the diagonal",
                        RuntimeWarning,
                    )
                    return torch.stack(L_list, dim=0)
            except:
                continue