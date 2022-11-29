from util import *

def euclidean_metric_cal(a, b):
    euc_metric = torch.empty(0, dtype=torch.long).cuda()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            euc = torch.sqrt(((a[i][j].repeat(b.shape[1], 1) - b[i]) ** 2).sum(dim=1))
            euc_metric = torch.cat([euc_metric, euc.unsqueeze(0)])
    return euc_metric.view(-1, a.shape[1], b.shape[1])


def function_cal(x, y, temperature, kccl_euc=0, ks=1, km=0):
    if kccl_euc == 1:
        euc_dis = euclidean_metric_cal(x, y)
        return torch.exp(torch.div(1, euc_dis * temperature))        
    else:
        return torch.exp((torch.div(torch.bmm(x, y.permute(0, 2, 1)), temperature) - km) * ks)


def kccl_loss(pooled_output, labels, k, temperature, neg_num=1, weight=None, loss_metric=0, neg_method=0, centroids=None, kccl_euc=0, ks=1, km=0):
    features = torch.cat((pooled_output, labels.unsqueeze(1)), 1)
    B, H = pooled_output.shape
    if neg_method in [3, 6]:
        pooled_output = pooled_output.view(-1, 1 + k + neg_num, H)  
        labels = labels.view(-1, 1 + k + neg_num)                   
        neg_examples = pooled_output[:, (1 + k):, :]                
        pooled_output = pooled_output[:, :(1 + k), :]               
            
    pos = function_cal(pooled_output, pooled_output, temperature, kccl_euc, ks, km)  
    neg1 = function_cal(pooled_output, neg_examples, temperature, kccl_euc, ks, 0)  
    if neg_num > 1 or neg_examples.shape[1] > 1:
        neg1 = torch.sum(neg1, dim=2).unsqueeze(2)  
    neg2 = neg1.permute(0, 2, 1).repeat(1, k + 1, 1)    
    pos_neg_mask = 1 - torch.eye(pos.shape[-1], device=neg2.device).unsqueeze(0).repeat(pos.shape[0], 1, 1)
    pos_neg_mask = pos_neg_mask.float()
    pos_neg = torch.sum(torch.sum(pos * pos_neg_mask, dim=-1), dim=-1).unsqueeze(1).unsqueeze(1).repeat(1, *pos.shape[1:])
    pos_neg = pos_neg - pos * 2
    neg = pos + neg2 + neg2.permute(0, 2, 1) + pos_neg
    loss_a = - torch.log(torch.div(pos, neg))
    if loss_metric == 0:
        for i in range(loss_a.shape[1]):
            loss_a[:, i, i] = 0
        loss_b = torch.sum(torch.sum(loss_a, dim=1))
        loss = loss_b / (pooled_output.shape[0] * k * (k + 1))
    elif loss_metric == 1:
        for i in range(loss_a.shape[1]):
            for j in range(loss_a.shape[2]):
                if i >= j:
                    loss_a[:, i, j] = 0
        loss_b = torch.sum(torch.sum(loss_a, dim=1))
        loss = 2 * loss_b / (pooled_output.shape[0] * k * (k + 1))
    return loss


class BertForModel(BertPreTrainedModel):
    def __init__(self, config, num_labels, model_type='3'):
        super(BertForModel, self).__init__(config)
        print(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.model_type = model_type
        print('self.model_type: ', self.model_type)
        self.weight = nn.Parameter(torch.rand(self.num_labels, self.bert.config.hidden_size))
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, feature_ext=False, mode=None, args=None, return_dict=None, centroids=None):
        encoded_layer_12, pooled_output_ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)

        pooled_output = torch.sum(encoded_layer_12[-1] * attention_mask.unsqueeze(-1).repeat(1, 1, encoded_layer_12[-1].shape[-1]), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.normalize(pooled_output)  
        weight = F.normalize(self.weight)  
        logits = torch.mm(pooled_output, weight.T)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                if labels is not None:
                    loss_1 = kccl_loss(pooled_output, labels, args.kccl_k, args.temperature, args.neg_num, self.weight,
                                        args.loss_metric, args.neg_method, centroids, args.kccl_euc, args.ks, args.km)
                    loss_2 = nn.CrossEntropyLoss()(logits, labels)
                    loss = loss_1 * args.KCCL_LOSS_LAMBDA + loss_2 * args.CE_LOSS_LAMBDA
                return loss, pooled_output, logits
            else:
                return pooled_output, logits
