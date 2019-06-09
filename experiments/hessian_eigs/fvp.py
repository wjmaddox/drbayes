import torch
import gpytorch
from gpytorch.lazy import LazyTensor
import copy
import pdb

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim,keepdim=True))

class FVP_FD(LazyTensor):
    def __init__(self, model, data, epsilon=1e-4):
        super(FVP_FD, self).__init__(data)
        self.model = model
        self.data = data
        self.epsilon = epsilon

        #compute number of paraemters
        self.num_params = 0
        for p in self.model.parameters():
            self.num_params += p.numel()

    def _size(self, val=None):
        if val==0 or val==1:
            return self.num_params
        else:
            return (self.num_params, self.num_params)

    def KL_logits(self, p_logits, q_logits):
        #computes KL divergence between two tensors of logits
        #KL(p || q) = \sum p log(p/q) \propto -\sum p log(q)
        #when we differentiate this, we really only need the cross-entropy?

        #this is the standard version (float/double independent)
        SM = torch.nn.Softmax(dim=1)
        p = SM(p_logits)

        #\sum_{i=1}^N \left(\sum_{k=1}^K p(y = k|x_i, \theta) (logit_p(y=k| x_i, \theta) - logit_q(y=k| x_i, \theta))\right)
        part1 = (p*(p_logits - q_logits)).sum(1).mean(0)
        #normalization constants
        
        #\log{ \sum_{k=1}^K exp{logit_p(y=k| x_i, \theta)} }
        #apparently implementations of LogSumExp are slower?
        r1 = torch.log(torch.exp(q_logits).sum(1))
        r2 = torch.log(torch.exp(p_logits).sum(1))
        #mean of difference of normalization constants
        part2 = (r1 - r2).mean(0)
        kl = part1 + part2
        return kl

    def _matmul(self, rhs):
        rhs_norm = torch.norm(rhs, dim=0)
        vec = rhs.t() / rhs_norm #transpose and normalize
        

        #check if all norms are zeros and return a zero matrix if so
        if torch.norm(vec,dim=0).eq(0.0).all():
            return torch.zeros(self.num_params, rhs.size(1), device=rhs.device, dtype=rhs.dtype)  

        grad_list = []
        
        #forwards pass with current parameters
        #return logit(y_k | \theta, x_i)
        with torch.no_grad():
            output = self.model(self.data).detach()

        #copy model state dict
        model_state_dict = copy.deepcopy(self.model.state_dict())

        for v in vec:
            #update model with \theta + \epsilon v
            i = 0
            for param_val in self.model.parameters():
                n = param_val.numel()
                param_val.data.add_(self.epsilon * v[i:i+n].view_as(param_val))
                i+=n

            with torch.autograd.enable_grad():
                #forwards pass with updated parameters
                #logit(y_k | \theta + \epsilon v, x_i)
                output_prime = self.model(self.data)

                #compute kl divergence loss
                #KL(p(y|\theta, x_i) || p(y|\theta + \epsilon v, x_i))
                kl = self.KL_logits(output.double(), output_prime.double())
                #kl = torch.tensor(kl, dtype=self.data.dtype, device=self.data.device)
                kl = kl.type(self.data.dtype).to(self.data.device)

                #compute gradient of kl divergence loss
                kl_grad = torch.autograd.grad(kl, self.model.parameters(), retain_graph=True)
                grad_i = flatten(kl_grad)
            grad_list.append(grad_i)

            #restore model dict now -> model(\theta) 
            self.model.load_state_dict(model_state_dict)

        #stack vector and turn divide by epsilon
        res = torch.stack(grad_list)/self.epsilon
        return res.t() * rhs_norm #de-normalize at the end

    def _size(self, val=None):
        if val==0 or val==1:
            return self.num_params
        else:
            return (self.num_params, self.num_params)

    def _approx_diag(self):
        #empirical fisher version of this
        #diag(F) \approx (\nabla_\theta \log{p(y_i | x_i, \theta)})^2
        grad_vec = flatten([param.grad for param in self.model.parameters()])
        #print('max element of grad_vec squared', grad_vec.max().pow(2))
        return grad_vec.pow(2.0)
    
    def __getitem__(self, index):
        # Will not do anything except get a single row correctly
        row_id = index[0].item()
        e_i = torch.zeros(self.size(0), 1, device=list(self.model.parameters())[0].device)
        e_i[row_id] = 1
        return self._matmul(e_i).squeeze()

        

