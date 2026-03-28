import torch
import torch.nn as nn
from monai.networks.layers import trunc_normal_
from torch_geometric.nn import SAGPooling, TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from stage1.Src.unest_base_patch_classes import Dual_UNseT
from torch_geometric.data import Data, Batch
import yaml
from Attention import FusionAttentionBlock, Attention_ori
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Graph_Transformer(nn.Module):
    def __init__(self, input_dim, head_num, hidden_dim):
        super(Graph_Transformer, self).__init__()
        #  multi-head self-attention
        self.graph_conv = TransformerConv(input_dim, input_dim//head_num, head_num)
        self.lin_out = nn.Linear(input_dim, input_dim)

        # feed forward network
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        #  multi-head self-attention
        out1 = self.lin_out(self.graph_conv(x, edge_index, edge_attr))

        # feed forward network
        out2 = self.ln1(out1 + x)
        out3 = self.lin2(self.act(self.lin1(out2)))
        out4 = self.ln2(out3 + out2)

        return out4


class GraphNet(nn.Module):
    def __init__(self, input_dim, num_classes, head_num=4, hidden_dim=256, ratio=0.8):
        super(GraphNet, self).__init__()
        self.conv1 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.pool1 = SAGPooling(input_dim, ratio)
        self.conv2 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.pool2 = SAGPooling(input_dim, ratio)
        self.conv3 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.pool3 = SAGPooling(input_dim, ratio)
        self.conv4 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.pool4 = SAGPooling(input_dim, ratio)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr.unsqueeze(1))
        
        x, edge_index, edge_attr, batch, perm, score = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index, edge_attr.unsqueeze(1))

        x, edge_index, edge_attr, batch, perm, score = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index, edge_attr.unsqueeze(1))

        x, edge_index, edge_attr, batch, perm, score = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv4(x, edge_index, edge_attr.unsqueeze(1))

        x, edge_index, edge_attr, batch, perm, score = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # multi-level features from read out layers
        x_feature = x1 + x2 + x3 + x4
        # x_out = self.mlp(x_feature)

        return x_feature


class gcn_net(nn.Module):
    def __init__(self, num_classes, changel):
        super().__init__()
        yaml_file = '/Src/yaml/mulorganseg_base.yaml'
        with open(yaml_file, 'r') as f:
            cfig = yaml.safe_load(f)
        self.device = device
        self.PT_gcn = GraphNet(input_dim=changel, num_classes=num_classes).to(device)
        self.MLN_gcn = GraphNet(input_dim=changel, num_classes=num_classes).to(device)
        self.Merge_gcn = GraphNet(input_dim=changel, num_classes=num_classes).to(device)
        self.feature_extractor = Dual_UNseT(in_channels=1, out_channels=2, cfig=cfig).to(device)
        self.feature_extractor.load_state_dict(
            torch.load('fig_128_fold1/model.pt')['state_dict'])
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False

        self.atten_PT = Attention_ori(dim=512, num_heads=8)
        self.atten_LN = Attention_ori(dim=512, num_heads=8)
        self.cross_atten = FusionAttentionBlock(embedding_dim=512, num_heads=8)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)

        init_parameter = [3. / 6, 2. / 6, 1. / 6]
        self.learning_parameter1 = nn.Parameter(torch.FloatTensor(init_parameter))
        self.learning_parameter2 = nn.Parameter(torch.FloatTensor(init_parameter))
        self.learning_parameter3 = nn.Parameter(torch.FloatTensor(init_parameter))

        self.learning_parameter = nn.Parameter(torch.FloatTensor([1./3, 1./3, 1./3]))
        self.linear = nn.Sequential(nn.Linear(512 * 3, 512),
                                    nn.LayerNorm(512),
                                    nn.ReLU()
                                    )

        input_dim = changel
        self.PT_head = nn.Sequential(nn.Linear(input_dim * 2, input_dim),
                                     nn.LayerNorm(input_dim),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(input_dim, input_dim // 4),
                                     nn.LayerNorm(input_dim // 4),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(input_dim // 4, num_classes)
                                     )
        self.MLN_head = nn.Sequential(nn.Linear(input_dim * 2, input_dim),
                                      nn.LayerNorm(input_dim),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(input_dim, input_dim // 4),
                                      nn.LayerNorm(input_dim // 4),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(input_dim // 4, num_classes)
                                      )
        self.Merge_head = nn.Sequential(nn.Linear(input_dim * 2, input_dim),
                                        nn.LayerNorm(input_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(input_dim, input_dim // 4),
                                        nn.LayerNorm(input_dim // 4),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(input_dim // 4, num_classes)
                                      )
        self.position_embeddings = nn.Parameter(torch.zeros(1, 512, 512))
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_edge(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            _, _, PT_classes, MLN_classes, PT_project, MLN_project = self.feature_extractor(x)
        
        PT_split_image = PT_project.permute(0, 2, 1)
        MLN_split_image = MLN_project.permute(0, 2, 1)
        PT_edge = torch.argmax(torch.softmax(PT_classes, dim=1), dim=1)
        MLN_edge = torch.argmax(torch.softmax(MLN_classes, dim=1), dim=1)
        PT_edge = PT_edge.view(PT_split_image.shape[0], PT_split_image.shape[1])
        MLN_edge = MLN_edge.view(MLN_split_image.shape[0], MLN_split_image.shape[1])
        return PT_edge, MLN_edge, PT_split_image, MLN_split_image

    def torch_adj(self, image, edge, indexs, learning_parameter):
        
        B, N, C = image.shape
        image = image + self.position_embeddings
        total = N
        indexs = indexs.to(self.device)
        adj_s = torch.zeros((B, total, total)).to(self.device)
        grap_list = []
        for z in range(B):
            d = indexs[z, :, 0].unsqueeze(1) - indexs[z, :, 0].unsqueeze(0)
            x = indexs[z, :, 1].unsqueeze(1) - indexs[z, :, 1].unsqueeze(0)
            y = indexs[z, :, 2].unsqueeze(1) - indexs[z, :, 2].unsqueeze(0)
            mask = (torch.abs(d) <= 1) & (torch.abs(x) <= 1) & (torch.abs(y) <= 1)
            prod = edge[z].unsqueeze(1) * edge[z].unsqueeze(0)
            sum_ = edge[z].unsqueeze(1) + edge[z].unsqueeze(0)
            adj_s[z] = torch.where((mask & (prod == 1)), learning_parameter[0], adj_s[z])
            adj_s[z] = torch.where(mask & (sum_ == 1), learning_parameter[1], adj_s[z])
            adj_s[z] = torch.where(mask & (sum_ == 0), learning_parameter[2], adj_s[z])
            for i in range(total):
                adj_s[z][i, i] = 0
            edge_index = torch.where(adj_s[z] > 0)
            edge_index = torch.stack(edge_index, dim=1)
            edge_attr = adj_s[z][edge_index[:, 0], edge_index[:, 1]]
            edge_index = edge_index.transpose(0, 1)
            edge_attr = edge_attr[:, None]
            graph_data = Data(x=image[z], edge_index=edge_index, edge_attr=edge_attr)
            
            grap_list.append(graph_data)
        batch = Batch.from_data_list(grap_list)
        return batch

    def forward(self, x, indexs_matrix):
        PT_label1, MLN_label1, PT_split_image, MLN_split_image = self.generate_edge(x)
        PT_edge, MLN_edge = PT_label1, MLN_label1
        
        PT_batch = self.torch_adj(PT_split_image, PT_edge, indexs_matrix, self.learning_parameter1)
        PT_gcn = self.PT_gcn(PT_batch.x, PT_batch.edge_index, PT_batch.edge_attr, PT_batch.batch)
        
        MLN_batch = self.torch_adj(MLN_split_image, MLN_edge, indexs_matrix, self.learning_parameter2)
        MLN_gcn = self.MLN_gcn(MLN_batch.x, MLN_batch.edge_index, MLN_batch.edge_attr, MLN_batch.batch)

        
        merge_edge = PT_edge + MLN_edge
        merge_edge[merge_edge > 1] = 1
        
        PT_atten = self.atten_PT(PT_split_image)
        LN_atten = self.atten_LN(MLN_split_image)
        x1_out = self.norm1(PT_split_image + PT_atten)
        x2_out = self.norm2(MLN_split_image + LN_atten)
        cross_out = self.norm3(self.cross_atten(x1_out, x2_out))
        merge_split_image = self.linear(torch.cat([x1_out, cross_out, x2_out], dim=-1))
        
        Merge_batch = self.torch_adj(merge_split_image, merge_edge, indexs_matrix, self.learning_parameter3)
        Merge_gcn = self.Merge_gcn(Merge_batch.x, Merge_batch.edge_index, Merge_batch.edge_attr, Merge_batch.batch)

        
        PT_out = self.PT_head(PT_gcn)
        MLN_out = self.MLN_head(MLN_gcn)
        Merge_out = self.Merge_head(Merge_gcn)
        delta_logits = (PT_out * self.learning_parameter[0] +
                        MLN_out * self.learning_parameter[1] +
                        Merge_out * self.learning_parameter[2])
        return delta_logits


if __name__ == '__main__':
    from thop import profile, clever_format
    x = torch.randn([1, 1, 128, 128, 128]).cuda()
    index = torch.randint(0, 1, [1, 512, 3]).cuda()
    net = gcn_net(num_classes=2, changel=512).cuda()
    y = net(x, index)
    # input = torch.randn(1, 1, 128, 128, 128).to(device)
    flops, params = profile(net, inputs=(x, index,))
    
    flops, params = clever_format([flops, params], '%.3f')

    print(f"运算量：{flops}, 参数量：{params}")
