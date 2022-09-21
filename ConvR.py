import torch.nn as nn
from torch.nn import functional as F, init
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_, xavier_normal_
from ResNet50 import resnet50

class ConvR(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations, device):
        super(ConvR, self).__init__()

        
        self.emb_e = nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        # 建立卷积核表
        self.conv_r = [nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias) for i in range(num_relations)]
        self.inp_drop = nn.Dropout(args.input_drop)
        self.hidden_drop = nn.Dropout(args.hidden_drop)
        self.feature_map_drop = nn.Dropout2d(args.feat_drop)
        self.loss = nn.BCELoss()
        
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1
        
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = nn.Linear(args.hidden_size,args.embedding_dim)
        print(num_entities, num_relations)
        
        #resnet50
        self.resnet = resnet50(args.embedding_dim,ibn=True) 
        self.relu = nn.Sigmoid()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)

    def image_embed(self, image_path='../input/ccks2022-task3/images/images'):
        image_en = os.listdir(image_path)

        entity2id = dict(self.entity2id)
        for each in image_en:
            index = int(entity2id[each])
            img_p = os.path.join(image_path, '{}/image_0.jpg'.format(each))
            if(os.path.exists(img_p)):
                img = Image.open(img_p, mode='r')
                img = np.array(img, dtype=np.float32)
                img/=255.0
                img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img)
                img = img.unsqueeze(0)
                image_em = self.resnet.forward(img)
                image_em = image_em.reshape(-1)
                self.emb_e.weight.data[index].copy_(image_em)
            else:
                continue

    def forward(self, e1, rel):
        # learning weight
        e1 = Variable(e1)
        # shape [batch size, 1, 20, embeding_dim // 20]
        e1_embedded= Variable(self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2))
        stacked_inputs = self.bn0(e1_embedded)
        x= self.inp_drop(stacked_inputs)
        # shape [batch size, 32, 16, 8]
        for i in range(x.shape[0]):
            if i == 0:
                input_x = self.conv_r[rel[i]].to(device)(x[i]).unsqueeze(dim=0)
            else:
                input_x = torch.cat([input_x, self.conv_r[rel[i]].to(device)(x[i]).unsqueeze(dim=0)], dim=0)
        x= self.bn1(input_x.to(device))
        x= F.relu(x)
        x = self.feature_map_drop(x)
        # shape [batch size, -1]
        x = x.view(x.shape[0], -1)
        # shape [batch size, embeding dim]
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        # shape [batch size, entity nums]
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred