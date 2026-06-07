import torch_geometric.nn as gnn
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch_geometric.data import Batch
from ..layer.gnn_conv import gnn_conv
from ..act.gnn_act import gnn_act
from ..norm.gnn_norm import gnn_norm
from ..pool.gnn_pool import gnn_pool


class ND_GNN_vanilla(nn.Module):     #定义图网络模型
    def __init__(self, net_param):
        super(ND_GNN_vanilla, self).__init__()


        #图节点原始特征信息维度
        self.num_features = net_param["num_features"]
        #节点结构信息图神经网络参数设置
        self.prelayers_num=net_param["prelayers_num"]
        self.prelayers_dim = net_param["prelayers_dim"]
        self.prelayers_gnn =net_param["prelayers_gnn"]
        self.prelayers_gnn_param = net_param["prelayers_gnn_param"]
        self.prelayers_skip=net_param["prelayers_skip"]
        self.prelayers_act=net_param["prelayers_act"]
        self.prelayers_norm=net_param["prelayers_norm"]
        self.gstate_pool_name = net_param["gstate_pool_name"]
        #拓扑相关性图神经网络参数设置
        self.encoder_graph_state = net_param["encoder_graph_state"]
        self.enclayers_num = net_param["enclayers_num"]
        self.enclayers_dim = net_param["enclayers_dim"]
        self.enclayers_gnn = net_param["enclayers_gnn"]
        self.enclayers_gnn_param = net_param["enclayers_gnn_param"]
        self.enclayers_skip = net_param["enclayers_skip"]
        self.enclayers_act = net_param["enclayers_act"]
        self.enclayers_norm = net_param["enclayers_norm"]
        #特征选择学习层参数设置
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
        self.pre_norms = torch.nn.ModuleList()
        self.gstate_pool=gnn_pool(pool_name=self.gstate_pool_name)
        self.encoder_conv_layers = torch.nn.ModuleList()
        self.encoder_linear_layers = torch.nn.ModuleList()
        self.enc_act = gnn_act(act_name=self.enclayers_act)
        self.enc_norms =torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        self.dec_act = gnn_act(act_name=self.declayers_act)

        #图神经网络层 学习图结构信息
        for i in range(self.prelayers_num):
            pin_channels=self.prelayers_dim if i>0 else self.num_features
            plout_channels=pout_channels=self.prelayers_dim
            if self.prelayers_gnn=="GATConv" :
                pin_channels=pin_channels*self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param["concat"]==True and i>0 else pin_channels
                plout_channels = pout_channels*self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param["concat"]==True else pout_channels
            self.pre_conv_layers.append(
                gnn_conv(gnn_name=self.prelayers_gnn,in_channels=pin_channels,out_channels=pout_channels,gnn_param=self.prelayers_gnn_param)

            )
            self.pre_norms.append(
                gnn_norm(norm_name=self.prelayers_norm, in_channerls=plout_channels)
            )
            if self.prelayers_skip!="none":
                self.pre_linear_layers.append(
                    torch.nn.Linear(in_features=pin_channels, out_features=plout_channels)
                )
        if self.prelayers_num==0:
            plout_channels=self.num_features

        # 节点结构信息编码层——融合全局图信息
        if self.encoder_graph_state == True:
            self.encoder_W = nn.Parameter(torch.FloatTensor(plout_channels, plout_channels))
            self.decoder_W1 = nn.Parameter(torch.FloatTensor(plout_channels, 1))

        #图神经网络层 学习节点拓扑相关性
        for i in range(self.enclayers_num):
            ein_channels=self.enclayers_dim if i>0 else plout_channels
            elout_channels=eout_channels=self.enclayers_dim
            if self.enclayers_gnn=="GATConv" :
                ein_channels=ein_channels*self.enclayers_gnn_param["heads"] if self.enclayers_gnn_param["concat"]==True and i>0 else ein_channels
                elout_channels = eout_channels*self.enclayers_gnn_param["heads"] if self.enclayers_gnn_param["concat"]==True else eout_channels

            self.encoder_conv_layers.append(
                gnn_conv(gnn_name=self.enclayers_gnn,in_channels=ein_channels,out_channels=eout_channels,gnn_param=self.enclayers_gnn_param)
            )
            self.enc_norms.append(
                gnn_norm(norm_name=self.enclayers_norm, in_channerls=elout_channels)
            )
            if self.enclayers_skip != "none":
                self.encoder_linear_layers.append(
                    torch.nn.Linear(in_features=ein_channels, out_features=elout_channels)
                )
        if self.enclayers_num == 0:
            elout_channels = plout_channels
        embedding_size=elout_channels

        #稳定学习框架所需相关变量
        self.register_buffer('pre_features', torch.zeros(net_param["batch_nodes_train"], embedding_size))
        self.register_buffer('pre_weight1', torch.ones(net_param["batch_nodes_train"], 1))

        #特征选择算子层 Regressor
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
        if self.encoder_graph_state == True:
            torch.nn.init.kaiming_normal_(self.encoder_W, a=math.sqrt(5))
            torch.nn.init.kaiming_normal_(self.decoder_W1, a=math.sqrt(5))


    def forward(self, data):
        #x：[batch_size*node_number, features_num]
        # edge_index:[2,edge_num]( directed edge)
        self.graph_state_list=data
        x,edge_index=data.x,data.edge_index
        #图神经网络层 学习图结构信息
        #prefeatures :x:[batch_size*node_number,prelayers_dim]
        for i in range(self.prelayers_num):
            x = self.pre_act(self.pre_conv_layers[i](x, edge_index) + self.pre_linear_layers[i](x)) \
                if self.prelayers_skip !="none" \
                else self.pre_act(self.pre_conv_layers[i](x, edge_index) )
            #x = F.normalize(input=x,p=2.,dim=-1)
            x=self.pre_norms[i](x,batch=data.batch)

        #节点结构信息编码层——融合全局图信息
        node_emb = x
        if self.encoder_graph_state == True:
            if isinstance(data, Batch):
                # state_emb calculate
                # rep_global:[node_cnt,batch_size] node_cnt 是整个batch中所有图的节点数
                v = torch.ones(data.num_nodes).to(self.device)
                i = torch.cat((torch.arange(data.num_nodes).unsqueeze(0).to(self.device), data.batch.unsqueeze(0)), 0)
                rep_global = torch.sparse_coo_tensor(indices=i, values=v, size=(data.num_nodes, data.num_graphs))
                # [batch_size emb_size]
                state_agg = self.gstate_pool(x=node_emb, batch=data.batch, size=data.num_graphs)
                # [node_size batch_size]*[batch emb_size]=[node_zise emb_size]
                state_agg = torch.spmm(rep_global, state_agg)
                # state_emb:[node_zise emb_size]*[emb_size emb_size]=[node_zise emb_size]
                state = F.normalize(F.relu(state_agg @ self.encoder_W), p=2, dim=-1)
                size = node_emb.size()
                # 结合全图信息
                node_emb = node_emb.unsqueeze(2) @ state.unsqueeze(1) @ self.decoder_W1.repeat(size[0], 1).reshape(
                    size[0], size[1], 1)
            else:
                state_agg = self.gstate_pool(x=node_emb,batch=None).unsqueeze(0)  # [1,emb_size]
                state = F.normalize(F.relu(state_agg @ self.encoder_W), p=2, dim=-1)
                size = node_emb.size()
                node_emb = node_emb.unsqueeze(2)@state@self.decoder_W1.repeat(size[0], 1).reshape(size[0], size[1], 1)
        # [batch_size*node_number enclayers_dim]
        x = node_emb
        if x.dim() != 2:
            x = node_emb.squeeze()
        self.node_prefeatures = x #融合全局信息的节点结构信息向量

        #学习节点拓扑相关性
        # encoder features:x[batch_size*node_number,enclayers_dim]
        for i in range(self.enclayers_num):
            
            x = self.enc_act(self.encoder_conv_layers[i](x, edge_index) + self.encoder_linear_layers[i](x)) \
                if self.enclayers_skip != "none" \
                else self.enc_act(self.encoder_conv_layers[i](x, edge_index))
            x=self.enc_norms[i](x,batch=data.batch)
        self.node_embfeatures=x #融合节点拓扑相关性的节点嵌入向量
       #特征选择学习层-回归算子
        for i in range(self.declayers_num):
            x = self.dec_act( self.decoder_layers[i](x) )
        #[batch_size*node_number ]
        x_out = x.view(x.size(0))
        x = torch.sigmoid(x_out).view(-1)
        return x


class robustenss_metircs_model(nn.Module):  # 定义图网络模型
    def __init__(self, net_param):
        super(robustenss_metircs_model, self).__init__()

        # 图节点原始特征信息维度
        self.num_features = net_param["num_features"]
        # 节点结构信息图神经网络参数设置
        self.prelayers_num = net_param["prelayers_num"]
        self.prelayers_dim = net_param["prelayers_dim"]
        self.prelayers_gnn = net_param["prelayers_gnn"]
        self.prelayers_gnn_param = net_param["prelayers_gnn_param"]
        self.prelayers_skip = net_param["prelayers_skip"]
        self.prelayers_act = net_param["prelayers_act"]
        self.prelayers_norm = net_param["prelayers_norm"]
        self.gstate_pool_name = net_param["gstate_pool_name"]
        # 拓扑相关性图神经网络参数设置
        self.enclayers_num = net_param["enclayers_num"]
        self.enclayers_dim = net_param["enclayers_dim"]
        self.enclayers_gnn = net_param["enclayers_gnn"]
        self.enclayers_gnn_param = net_param["enclayers_gnn_param"]
        self.enclayers_skip = net_param["enclayers_skip"]
        self.enclayers_act = net_param["enclayers_act"]
        self.enclayers_norm = net_param["enclayers_norm"]
        # 特征选择学习层参数设置
        self.declayers_num = net_param["declayers_num"]
        self.declayers_dim = net_param["declayers_dim"]
        self.declayers_act = net_param["declayers_act"]

        self.node_prefeatures = None
        self.node_embfeatures = None
        self.graph_state_list = None

        # Call super
        self.pre_conv_layers = torch.nn.ModuleList()
        self.pre_linear_layers = torch.nn.ModuleList()
        self.pre_act = gnn_act(act_name=self.prelayers_act)
        self.pre_norms = torch.nn.ModuleList()
        self.gstate_pool = gnn_pool(pool_name=self.gstate_pool_name)
        self.encoder_conv_layers = torch.nn.ModuleList()
        self.encoder_linear_layers = torch.nn.ModuleList()
        self.enc_act = gnn_act(act_name=self.enclayers_act)
        self.enc_norms = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        self.dec_act = gnn_act(act_name=self.declayers_act)

        # 图神经网络层 学习图结构信息
        for i in range(self.prelayers_num):
            pin_channels = self.prelayers_dim if i > 0 else self.num_features
            plout_channels = pout_channels = self.prelayers_dim
            if self.prelayers_gnn == "GATConv":
                pin_channels = pin_channels * self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param[
                                                                                       "concat"] == True and i > 0 else pin_channels
                plout_channels = pout_channels * self.prelayers_gnn_param["heads"] if self.prelayers_gnn_param[
                                                                                          "concat"] == True else pout_channels
            self.pre_conv_layers.append(
                gnn_conv(gnn_name=self.prelayers_gnn, in_channels=pin_channels, out_channels=pout_channels,
                         gnn_param=self.prelayers_gnn_param)

            )
            self.pre_norms.append(
                gnn_norm(norm_name=self.prelayers_norm, in_channerls=plout_channels)
            )
            if self.prelayers_skip != "none":
                self.pre_linear_layers.append(
                    torch.nn.Linear(in_features=pin_channels, out_features=plout_channels)
                )
        if self.prelayers_num == 0:
            plout_channels = self.num_features


        # 图神经网络层 学习节点拓扑相关性
        for i in range(self.enclayers_num):
            ein_channels = self.enclayers_dim if i > 0 else plout_channels
            elout_channels = eout_channels = self.enclayers_dim
            if self.enclayers_gnn == "GATConv":
                ein_channels = ein_channels * self.enclayers_gnn_param["heads"] if self.enclayers_gnn_param[
                                                                                       "concat"] == True and i > 0 else ein_channels
                elout_channels = eout_channels * self.enclayers_gnn_param["heads"] if self.enclayers_gnn_param[
                                                                                          "concat"] == True else eout_channels

            self.encoder_conv_layers.append(
                gnn_conv(gnn_name=self.enclayers_gnn, in_channels=ein_channels, out_channels=eout_channels,
                         gnn_param=self.enclayers_gnn_param)
            )
            self.enc_norms.append(
                gnn_norm(norm_name=self.enclayers_norm, in_channerls=elout_channels)
            )
            if self.enclayers_skip != "none":
                self.encoder_linear_layers.append(
                    torch.nn.Linear(in_features=ein_channels, out_features=elout_channels)
                )
        if self.enclayers_num == 0:
            elout_channels = pout_channels
        embedding_size = elout_channels



        # 特征选择算子层 Regressor
        for i in range(self.declayers_num):
            din_channels = self.declayers_dim if i > 0 else embedding_size
            dout_channels = self.declayers_dim if i < (self.declayers_num - 1) else 1
            self.decoder_layers.append(
                torch.nn.Linear(in_features=din_channels, out_features=dout_channels)
            )
        self.reset_parameters(self.decoder_layers)

    def reset_parameters(self, layers):
        for i in range(len(layers)):
            weight = layers[i].weight
            bias = layers[i].bias
            torch.nn.init.kaiming_normal_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.normal_(bias, -bound, bound)

    def forward(self, data):
        # x：[batch_size*node_number, features_num]
        # edge_index:[2,edge_num]( directed edge)

        x, edge_index = data.x.reshape(-1,1), data.edge_index
        # 图神经网络层 学习图结构信息
        # prefeatures :x:[batch_size*node_number,prelayers_dim]
        for i in range(self.prelayers_num):
            x = self.pre_act(self.pre_conv_layers[i](x, edge_index) + self.pre_linear_layers[i](x)) \
                if self.prelayers_skip != "none" \
                else self.pre_act(self.pre_conv_layers[i](x, edge_index))
            # x = F.normalize(input=x,p=2.,dim=-1)
            x = self.pre_norms[i](x, batch=data.batch)
        # 学习节点拓扑相关性
        # encoder features:x[batch_size*node_number,enclayers_dim]
        for i in range(self.enclayers_num):
            x = self.enc_act(self.encoder_conv_layers[i](x, edge_index) + self.encoder_linear_layers[i](x)) \
                if self.enclayers_skip != "none" \
                else self.enc_act(self.encoder_conv_layers[i](x, edge_index))
            x = self.enc_norms[i](x, batch=data.batch)

        # 节点结构信息编码层——融合全局图信息
        node_emb = x
        if isinstance(data, Batch):
            graph_state = self.gstate_pool(x=node_emb, batch=data.batch, size=data.num_graphs)
        else:
            graph_state = self.gstate_pool(x=node_emb, batch=None).unsqueeze(0)  # [1,emb_size]
        x=graph_state

        # 特征选择学习层-回归算子
        for i in range(self.declayers_num-1):
            x = self.dec_act(self.decoder_layers[i](x))
            # x = F.normalize(input=x,p=2.,dim=-1)
        x = self.decoder_layers[-1](x)
        # [batch_size,]

        x_out = x.reshape(-1)

        return x_out



