import torch
import gpytorch
from gpytorch.lazy import LazyTensor
import copy
import time

def flatten(lst):
    #for this we will compute gradient of loss, then pass into fvp as rhs
    #then forwards back through again model(data), model(data)|parameters + epsilon*gradient
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i=0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i:i+n].view(tensor.shape))
        i+=n
    return outList

class FVP_AG(LazyTensor):
    def __init__(self, model, data, **kwargs):
        super(FVP_AG, self).__init__(data)
        self.model = model
        self.data = data

        #compute number of paraemters
        self.num_params = 0
        for p in self.model.parameters():
            self.num_params += p.numel()

    def _size(self, val=None):
        if val==0 or val==1:
            return self.num_params
        else:
            return (self.num_params, self.num_params)

    # A loss whose 2nd derivative is the Fisher information matrix
    def detached_entropy(self, logits, y=None):
        # -1*\frac{1}{m}\sum_{i,k} [f_k(x_i)] \log f_k(x_i), where [] is detach
        log_probs = torch.nn.LogSoftmax(dim=1)(logits)
        probs = torch.nn.Softmax(dim=1)(logits)
        return -1*(probs.detach() * log_probs).sum(1).mean(0)

    def _matmul(self, rhs):
        orig_dtype = rhs.dtype
        rhs = rhs.float()
        vec = rhs.t() #transpose

        #check if all norms are zeros and return a zero matrix otherwise
        if torch.norm(vec,dim=0).eq(0.0).all():
            return torch.zeros(self.num_params, rhs.size(1), device=rhs.device, dtype=rhs.dtype)

        #form list of all vectors
        with torch.autograd.no_grad():
            vec_list = []
            for v in vec:
                vec_list.append(unflatten_like(v, self.model.parameters()))

        with torch.autograd.enable_grad():
            #start = time.time()
            #compute batch loss with detached entropy
            batch_loss = self.detached_entropy(self.model(self.data))
            #print('Batch loss time: ', time.time() - start)

            #first gradient wrt parameters
            #start = time.time()
            grad_bl_list = torch.autograd.grad(batch_loss, self.model.parameters(), create_graph=True,only_inputs=True)
            #print('Grad time: ', time.time() - start)

            res = []
            for vec_sublist in vec_list:
                deriv=0
                #print(len(vec_sublist), len(vec_list))
                for vec_part, grad_part in zip(vec_sublist, grad_bl_list):
                    #start = time.time()
                    #print(vec_part.norm(), grad_part.norm())
                    deriv += torch.sum(vec_part.detach().double()*grad_part.double())
                
                #print(deriv)
                #fast implicit hvp product
                hvp_list = torch.autograd.grad(deriv.float(), self.model.parameters(), only_inputs=True, retain_graph=True)
                #print('Hvp time: ', time.time() - start)

                res.append(flatten(hvp_list))

        res_matrix = torch.stack(res).detach()
        return res_matrix.t().type(orig_dtype)
