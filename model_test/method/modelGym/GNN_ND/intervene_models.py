import torch_geometric.nn as gnn
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from loss_functions.GeneralizedCELoss.GCE import GeneralizedCELoss
from loss_functions.negentropy_loss.negentroy_loss import NG_loss
from ..layer.gnn_conv import gnn_conv
from ..act.gnn_act import gnn_act


class ND_GNN_intervene(nn.Module):     #定义图网络模型
    def __init__(self, net_param):
        super(ND_GNN_intervene, self).__init__()


        #图节点原始特征信息维度
        self.num_features = net_param["num_features"]
        #
        self.prelayers_num=net_param["prelayers_num"]
        self.prelayers_dim = net_param["prelayers_dim"]
        self.prelayers_gnn =net_param["prelayers_gnn"]
        self.prelayers_gnn_param = net_param["prelayers_gnn_param"]
        self.prelayers_skip=net_param["prelayers_skip"]
        self.prelayers_act=net_param["prelayers_act"]

        self.enclayers_num = net_param["enclayers_num"]
        self.enclayers_dim = net_param["enclayers_dim"]
        self.enclayers_gnn = net_param["enclayers_gnn"]
        self.enclayers_gnn_param = net_param["enclayers_gnn_param"]
        self.enclayers_skip = net_param["enclayers_skip"]
        self.enclayers_act = net_param["enclayers_act"]
        self.encoder_graph_state =net_param["encoder_graph_state"]

        self.declayers_num = net_param["declayers_num"]
        self.declayers_dim = net_param["declayers_dim"]
        self.declayers_act =net_param["declayers_act"]
        self.device=net_param["device"]

        self.node_prefeatures = None
        self.node_embfeatures=None
        self.graph_state_list=None


        # Call super
        self.pre_conv_layers=torch.nn.ModuleList()
        self.pre_linear_layers = torch.nn.ModuleList()
        self.pre_act=gnn_act(act_name=self.prelayers_act)
        self.encoder_conv_layers = torch.nn.ModuleList()
        self.encoder_linear_layers = torch.nn.ModuleList()
        self.enc_act = gnn_act(act_name=self.enclayers_act)
        self.decoder_layers = torch.nn.ModuleList()
        self.dec_act = gnn_act(act_name=self.declayers_act)
        #图结构信息提取层
        for i in range(self.prelayers_num):
            pin_channels=self.prelayers_dim if i>0 else self.num_features
            plout_channels=pout_channels=self.prelayers_dim
            if self.prelayers_gnn=="GATConv" :
                pin_channels=pin_channels*self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param["concat"]==True and i>0 else pin_channels
                plout_channels = pout_channels*self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param["concat"]==True else pout_channels
            self.pre_conv_layers.append(
                gnn_conv(gnn_name=self.prelayers_gnn,in_channels=pin_channels,out_channels=pout_channels,gnn_param=self.prelayers_gnn_param)
            )
            if self.prelayers_skip!="none":
                self.pre_linear_layers.append(
                    torch.nn.Linear(in_features=pin_channels, out_features=plout_channels)
                )
        if self.prelayers_num==0:
            pout_channels=self.num_features
        #节点特征信息编码层
        for i in range(self.enclayers_num):
            ein_channels=self.enclayers_dim if i>0 else plout_channels
            elout_channels=eout_channels=self.enclayers_dim
            if self.enclayers_gnn=="GATConv" :
                ein_channels=ein_channels*self.enclayers_gnn_param["heads"] if self.enclayers_gnn_param["concat"]==True and i>0 else ein_channels
                elout_channels = eout_channels*self.enclayers_gnn_param["heads"] if self.enclayers_gnn_param["concat"]==True else eout_channels

            self.encoder_conv_layers.append(
                gnn_conv(gnn_name=self.enclayers_gnn,in_channels=ein_channels,out_channels=eout_channels,gnn_param=self.enclayers_gnn_param)
            )
            if self.enclayers_skip != "none":
                self.encoder_linear_layers.append(
                    torch.nn.Linear(in_features=ein_channels, out_features=elout_channels)
                )
        if self.enclayers_num == 0:
            eout_channels = pout_channels
        embedding_size=elout_channels
        # 节点特征信息编码层——考虑全局图信息
        if self.encoder_graph_state==True:
            self.encoder_W = nn.Parameter( torch.FloatTensor(embedding_size,embedding_size))
            self.decoder_W1 = nn.Parameter(torch.FloatTensor(embedding_size, 1))

        self.register_buffer('pre_features', torch.zeros(net_param["batch_nodes_train"], embedding_size))
        self.register_buffer('pre_weight1', torch.ones(net_param["batch_nodes_train"], 1))

        #解码层 Regressor
        for i in range(self.declayers_num):
            din_channels = self.declayers_dim if i > 0 else embedding_size
            dout_channels = self.declayers_dim if i<(self.declayers_num-1) else 1
            self.decoder_layers.append(
                torch.nn.Linear(in_features=din_channels, out_features=dout_channels)
                       )
        self.reset_parameters(self.decoder_layers)


    def reset_parameters(self,layers):
        for i in range(len(layers)):
            weight=layers[i].weight
            bias=layers[i].bias
            torch.nn.init.kaiming_normal_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.normal_(bias , -bound, bound)
        torch.nn.init.kaiming_normal_(self.encoder_W, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.decoder_W1, a=math.sqrt(5))


    def forward(self, data,subgraph_edge_index):
        #x：[batch_size*node_number, features_num]
        # edge_index:[2,edge_num]( directed edge)
        self.graph_state_list=data
        x=data.x
        edge_index=subgraph_edge_index


        #prefeatures :x:[batch_size*node_number,prelayers_dim]
        for i in range(self.prelayers_num):
            x = self.pre_act(self.pre_conv_layers[i](x, edge_index) + self.pre_linear_layers[i](x)) if self.prelayers_skip !="none" else self.pre_act(self.pre_conv_layers[i](x, edge_index) )
            x = F.normalize(input=x,p=2.,dim=-1)
        self.node_prefeatures=x

        #encoder features:x[batch_size*node_number,enclayers_dim]
        for i in range(self.enclayers_num):
            x = self.enc_act(self.encoder_conv_layers[i](x, edge_index) + self.encoder_linear_layers[i](x)) if self.enclayers_skip !="none" else self.enc_act(self.encoder_conv_layers[i](x, edge_index))
            x = F.normalize(input=x, p=2., dim=-1)

        self.node_prefeatures = x
        node_emb=x
        if self.encoder_graph_state == True:
            if isinstance(data,Batch):
                # state_emb calculate
                # rep_global:[node_cnt,batch_size] node_cnt 是整个batch中所有图的节点数
                v = torch.ones(data.num_nodes).to(self.device)
                i = torch.cat((torch.arange(data.num_nodes).unsqueeze(0).to(self.device), data.batch.unsqueeze(0)), 0)
                rep_global = torch.sparse_coo_tensor(indices=i, values=v, size=(data.num_nodes, data.num_graphs))

                # [batch_size emb_size]
                state_agg = gnn.global_add_pool(x=node_emb, batch=data.batch, size=data.num_graphs)
                # [node_size batch_size]*[batch emb_size]=[node_zise emb_size]
                state_agg = torch.spmm(rep_global, state_agg)
                # state_emb:[batch_size emb_size]*[emb_size emb_size]=[batch_size emb_size]
                state = F.normalize(F.relu(state_agg @ self.encoder_W), p=2, dim=-1)

                size = node_emb.size()
                #结合图状态，（内积）
                node_emb=node_emb.unsqueeze(2) @ state.unsqueeze(1)@self.decoder_W1.repeat(size[0],1).reshape(size[0],size[1],1)
            else:

                state_agg = torch.sum(node_emb, dim=0).unsqueeze(0)  # [1,emb_size]
                state = F.normalize(F.relu(state_agg @ self.encoder_W), p=2, dim=-1)
                size = node_emb.size()
                node_emb = node_emb.unsqueeze(2) @ state @ self.decoder_W1.repeat(size[0], 1).reshape(size[0],size[1],1)
        # [batch_size*node_number enclayers_dim]
        x=node_emb.squeeze()
        self.node_embfeatures=x

        return x

    def classifier(self,feature):
        x=feature
        for i in range(self.declayers_num):
            x = self.dec_act( self.decoder_layers[i](x) )
        #[batch_size*node_number ]
        x_out = x.view(x.size(0))
        x = torch.sigmoid(x_out)
        return x

    def loss(self, pred, label,reduction="mean"):
        criterion = nn.BCELoss(reduction=reduction).to(self.device)
        loss = criterion(pred, label)
        return loss

    def GCE(self,pred,label):
        criterion = GeneralizedCELoss().to(self.device)
        loss = criterion(pred, label)
        return loss



class stablegraph_GNN(nn.Module):     #定义图网络模型
    def __init__(self, net_param):
        super(stablegraph_GNN, self).__init__()


        #图节点原始特征信息维度
        self.num_features = net_param["num_features"]
        #
        self.prelayers_num=net_param["prelayers_num"]
        self.prelayers_dim = net_param["prelayers_dim"]
        self.prelayers_gnn =net_param["prelayers_gnn"]
        self.prelayers_gnn_param = net_param["prelayers_gnn_param"]
        self.prelayers_skip=net_param["prelayers_skip"]
        self.prelayers_act=net_param["prelayers_act"]



        self.declayers_num = net_param["declayers_num"]
        self.declayers_dim = net_param["declayers_dim"]
        self.declayers_act =net_param["declayers_act"]
        self.device=net_param["device"]

        self.threshold=net_param["edge_subgraph_threshold"]#0.5
        self.node_prefeatures = None


        # Call super
        self.pre_conv_layers=torch.nn.ModuleList()
        self.pre_linear_layers = torch.nn.ModuleList()
        self.pre_act=gnn_act(act_name=self.prelayers_act)

        self.decoder_layers = torch.nn.ModuleList()
        self.dec_act = gnn_act(act_name=self.declayers_act)
        #图结构信息提取层
        for i in range(self.prelayers_num):
            pin_channels=self.prelayers_dim if i>0 else self.num_features
            plout_channels=pout_channels=self.prelayers_dim
            if self.prelayers_gnn=="GATConv" :
                pin_channels=pin_channels*self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param["concat"]==True and i>0 else pin_channels
                plout_channels = pout_channels*self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param["concat"]==True else pout_channels
            self.pre_conv_layers.append(
                gnn_conv(gnn_name=self.prelayers_gnn,in_channels=pin_channels,out_channels=pout_channels,gnn_param=self.prelayers_gnn_param)
            )
            if self.prelayers_skip!="none":
                self.pre_linear_layers.append(
                    torch.nn.Linear(in_features=pin_channels, out_features=plout_channels)
                )
        if self.prelayers_num==0:
            pout_channels=self.num_features

        #解码层 Regressor
        for i in range(self.declayers_num):
            din_channels = self.declayers_dim if i > 0 else 2*pout_channels
            dout_channels = self.declayers_dim if i<(self.declayers_num-1) else 1
            self.decoder_layers.append(
                torch.nn.Linear(in_features=din_channels, out_features=dout_channels)
                       )
       # self.reset_parameters(self.decoder_layers)


    def reset_parameters(self,layers):
        for i in range(len(layers)):
            weight=layers[i].weight
            bias=layers[i].bias
            torch.nn.init.kaiming_normal_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.normal_(bias , -bound, bound)

    def forward(self, data):
        #x：[batch_size*node_number, features_num]
        # edge_index:[2,edge_num]( directed edge)
        x,edge_index=data.x,data.edge_index
        row,col =data.edge_index

        #prefeatures :x:[batch_size*node_number,prelayers_dim]
        for i in range(self.prelayers_num):
            x = self.pre_act(self.pre_conv_layers[i](x, edge_index) + self.pre_linear_layers[i](x)) if self.prelayers_skip !="none" else self.pre_act(self.pre_conv_layers[i](x, edge_index) )
            x = F.normalize(input=x,p=2.,dim=-1)
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        #decoder
        for i in range(self.declayers_num):
            edge_score = self.dec_act( self.decoder_layers[i](edge_rep) )

        edge_score = torch.sigmoid(edge_score).view(-1)
        causal_edge_index, causal_edge_weight, conf_edge_index, conf_edge_weight =  self.split_graph(data,edge_score)
        return causal_edge_index, causal_edge_weight,conf_edge_index, conf_edge_weight,edge_score

    def split_batch(self,g):
        split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
        edge_indices = torch.split(g.edge_index, split, dim=1)
        num_nodes = degree(g.batch, dtype=torch.long)
        cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
        num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
        cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

        return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

    def split_graph(self,data,edge_score):
        causal_edge_index = torch.LongTensor([[], []]).to(data.x.device)
        causal_edge_weight = torch.tensor([]).to(data.x.device)
        conf_edge_index = torch.LongTensor([[], []]).to(data.x.device)
        conf_edge_weight = torch.tensor([]).to(data.x.device)

        edge_indices, _, _, num_edges, cum_edges = self.split_batch(data)
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):

            single_mask = edge_score[C:C + N]
            single_mask_detach = edge_score[C:C + N].detach().cpu()
              # 此处应该改成 分类阈值
            idx_reserve = torch.nonzero(single_mask_detach>self.threshold).view(-1)
            idx_drop =torch.nonzero(single_mask_detach<=self.threshold).view(-1)

            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)

            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            conf_edge_weight = torch.cat([conf_edge_weight, 1-single_mask[idx_drop]])  # ?? 1-single_mask[idx_drop]

        return causal_edge_index , causal_edge_weight,conf_edge_index, conf_edge_weight

    def edge_loss(self,input):
        criterion=NG_loss().to(self.device)
        return criterion(input)

