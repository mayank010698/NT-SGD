import torch
import copy


class NormTruncate():
    def __init__(self,model,eps2):
        self.eps2 = eps2

    def step(self, model):
        kappa = self.cut_rate(model)

        layer_sparsity = {}

        for param in model.named_parameters():
            mask = torch.where(torch.abs(param[1].grad)>kappa,1,0)
            param[1].grad = mask*param[1].grad
            layer_sparsity[param[0]] = (torch.sum(mask)/mask.numel())

        return layer_sparsity
    
    def cut_rate(self,model):
        gradients = torch.cat([param.grad.flatten() for param in model.parameters()])
        sorted_grad = torch.sort(torch.abs(gradients),descending=True).values
        grad_norm = torch.norm(gradients).pow(2)
        cumsum = torch.cumsum(sorted_grad.pow(2), dim=0)

        idx = torch.where(cumsum>(1-self.eps2)*grad_norm)[0]

        if idx.numel():
            return sorted_grad[idx[0]]
        else:
            return torch.Tensor([0]).to("cuda")
    
    def reset(self):
        pass

class ETC():
    def __init__(self, model, p, thres) -> None:
        self.count   = 0
        self.thres  = thres
        self.model  = copy.deepcopy(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        for param in self.model.parameters():
            param.grad = torch.zeros(param.shape).to(self.device)
        self.kappa = 0
        self.p = p
    
    def step(self,model):
        layer_sparsity = {}
        if self.count < self.thres:
            for net2,net1 in zip(self.model.named_parameters(),model.named_parameters()):
                mask = torch.ones(net1[1].grad.shape)
                net2[1].grad += torch.abs(net1[1].grad.clone().detach())
                layer_sparsity[net1[0]]=(torch.sum(mask)/mask.numel())
        elif self.count == self.thres:
            # based on the accumulated gradients choose top-k arms
            gradients   = torch.cat([param.grad.flatten() for param in self.model.parameters()])
            sorted_grad = torch.sort(torch.abs(gradients),descending=True).values

            self.kappa = sorted_grad[int(self.p*gradients.shape[0])]
            print(" Setting kappa as: {}",self.kappa.detach().cpu())

            for net1,net2 in zip(self.model.named_parameters(),model.named_parameters()):
                mask = torch.where(net1[1].grad>=self.kappa,1,0)
                net2[1].grad = mask*net2[1].grad
                layer_sparsity[net2[0]]=torch.sum(mask)/mask.numel()
        else:
            for net1,net2 in zip(self.model.named_parameters(),model.named_parameters()):
                mask = torch.where(net1[1].grad>=self.kappa,1,0)
                net2[1].grad = mask*net2[1].grad
                layer_sparsity[net2[0]]=torch.sum(mask)/mask.numel()
        
        
        self.count+=1
        return layer_sparsity

    def reset(self):
        self.count = 0
        self.model.zero_grad()

        


