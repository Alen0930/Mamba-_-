"""
FINDER – pure-Python rewrite of the inference pipeline.

Reference:
> Fan, C., Zeng, L., Sun, Y., Liu, Y.-Y. Finding key players in complex networks through deep reinforcement learning. Nature Machine Intelligence 2, 317–324 (2020). DOI: 10.1038/s42256-020-0177-2
"""
"""
Pure-Python rewrite of FINDER_ND inference pipeline.
Removes all Cython and graph_tool dependencies.
Compatible with TensorFlow 1.x.
"""
import numpy as np
import networkx as nx
import random
import time
import sys
from tqdm import tqdm
import tensorflow as tf

# Hyper Parameters (same as FINDER.pyx)
GAMMA = 1
UPDATE_TIME = 1000
EMBEDDING_SIZE = 64
MAX_ITERATION = 1000000
LEARNING_RATE = 0.0001
MEMORY_SIZE = 500000
Alpha = 0.0001
epsilon = 0.0000001
alpha = 0.6
beta = 0.4
beta_increment_per_sampling = 0.001
TD_err_upper = 1.
N_STEP = 5
NUM_MIN = 30
NUM_MAX = 50
REG_HIDDEN = 32
BATCH_SIZE = 64
initialization_stddev = 0.01
n_valid = 200
aux_dim = 4
num_env = 1
inf = 2147483647 / 2
max_bp_iter = 3
aggregatorID = 0  # 0:sum; 1:mean; 2:GCN
embeddingMethod = 1  # 0:structure2vec; 1:graphsage


# ------------------------------------------------------------------
# 1. Graph data structures (replaces graph.pyx)
# ------------------------------------------------------------------
class Graph:
    def __init__(self, num_nodes=0, num_edges=0, edges_from=None, edges_to=None):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edge_list = []
        self.adj_list = [[] for _ in range(num_nodes)]
        if edges_from is not None and edges_to is not None:
            for i in range(num_edges):
                a, b = int(edges_from[i]), int(edges_to[i])
                self.edge_list.append((a, b))
                self.adj_list[a].append(b)
                self.adj_list[b].append(a)


class GSet:
    def __init__(self):
        self.graph_pool = {}

    def InsertGraph(self, gid, graph):
        self.graph_pool[gid] = graph

    def Sample(self):
        return random.choice(list(self.graph_pool.values()))

    def Get(self, gid):
        return self.graph_pool[gid]

    def Clear(self):
        self.graph_pool.clear()


# ------------------------------------------------------------------
# 2. MvcEnv (replaces mvc_env.pyx)
# ------------------------------------------------------------------
class MvcEnv:
    def __init__(self, norm):
        self.norm = norm
        self.graph = None
        self.numCoveredEdges = 0
        self.CcNum = 1.0
        self.action_list = []
        self.covered_set = set()
        self.state_seq = []
        self.act_seq = []
        self.reward_seq = []
        self.sum_rewards = []
        self.avail_list = []

    def s0(self, _g):
        self.graph = _g
        self.covered_set = set()
        self.action_list = []
        self.numCoveredEdges = 0
        self.CcNum = 1.0
        self.state_seq = []
        self.act_seq = []
        self.reward_seq = []
        self.sum_rewards = []
        self.avail_list = []

    def stepWithoutReward(self, a):
        assert self.graph is not None
        assert a not in self.covered_set
        self.covered_set.add(a)
        self.action_list.append(a)
        for neigh in self.graph.adj_list[a]:
            if neigh not in self.covered_set:
                self.numCoveredEdges += 1

    def isTerminal(self):
        assert self.graph is not None
        return self.graph.num_edges == self.numCoveredEdges

    def randomAction(self):
        assert self.graph is not None
        self.avail_list = []
        for i in range(self.graph.num_nodes):
            if i in self.covered_set:
                continue
            useful = False
            for neigh in self.graph.adj_list[i]:
                if neigh not in self.covered_set:
                    useful = True
                    break
            if useful:
                self.avail_list.append(i)
        return random.choice(self.avail_list)


# ------------------------------------------------------------------
# 3. PrepareBatchGraph (replaces PrepareBatchGraph.pyx)
# ------------------------------------------------------------------
class PrepareBatchGraph:
    def __init__(self, aggregatorID=0):
        self.aggregatorID = aggregatorID
        self.act_select = None
        self.rep_global = None
        self.n2nsum_param = None
        self.laplacian_param = None
        self.subgsum_param = None
        self.idx_map_list = []
        self.subgraph_id_span = []
        self.aux_feat = []
        self.avail_act_cnt = []

    @staticmethod
    def _to_sparse_tensor_value(row_idx, col_idx, values, row_num, col_num):
        if len(values) == 0:
            indices = np.zeros((0, 2), dtype=np.int64)
            vals = np.zeros(0, dtype=np.float32)
        else:
            indices = np.column_stack((row_idx, col_idx)).astype(np.int64)
            vals = np.array(values, dtype=np.float32)
        return tf.SparseTensorValue(indices, vals, (row_num, col_num))

    def GetStatusInfo(self, g, covered):
        idx_map = [-1] * g.num_nodes
        counter = 0
        twohop_number = 0
        node_twohop_counter = {}
        c = set(covered)
        n = 0

        for p in g.edge_list:
            if p[0] in c or p[1] in c:
                counter += 1
            else:
                if idx_map[p[0]] < 0:
                    n += 1
                if idx_map[p[1]] < 0:
                    n += 1
                idx_map[p[0]] = 0
                idx_map[p[1]] = 0

                twohop_number += node_twohop_counter.get(p[0], 0)
                node_twohop_counter[p[0]] = node_twohop_counter.get(p[0], 0) + 1
                twohop_number += node_twohop_counter.get(p[1], 0)
                node_twohop_counter[p[1]] = node_twohop_counter.get(p[1], 0) + 1

        return n, counter, twohop_number, idx_map

    def SetupGraphInput(self, idxes, g_list, covered, actions=None):
        self.idx_map_list = []
        self.subgraph_id_span = []
        self.aux_feat = []
        self.avail_act_cnt = []

        node_cnt = 0

        for i in range(len(idxes)):
            g = g_list[idxes[i]]
            cov = covered[idxes[i]]

            temp_feat = []
            if g.num_nodes > 0:
                temp_feat.append(len(cov) / g.num_nodes)

            n, counter, twohop_number, idx_map = self.GetStatusInfo(g, cov)
            self.avail_act_cnt.append(n)

            if len(g.edge_list) > 0:
                temp_feat.append(counter / len(g.edge_list))
            else:
                temp_feat.append(0.0)

            temp_feat.append(twohop_number / max(1.0, g.num_nodes * g.num_nodes))
            temp_feat.append(1.0)

            self.aux_feat.append(temp_feat)
            self.idx_map_list.append(idx_map)
            node_cnt += n

        # Build big graph
        total_nodes = node_cnt
        big_edge_list = []
        node_offset = 0

        for i in range(len(idxes)):
            g = g_list[idxes[i]]
            idx_map = self.idx_map_list[i]

            t = 0
            for j in range(g.num_nodes):
                if idx_map[j] < 0:
                    continue
                idx_map[j] = t
                t += 1

            for p in g.edge_list:
                if idx_map[p[0]] < 0 or idx_map[p[1]] < 0:
                    continue
                x = idx_map[p[0]] + node_offset
                y = idx_map[p[1]] + node_offset
                big_edge_list.append((x, y))
                big_edge_list.append((y, x))

            node_offset += self.avail_act_cnt[i]

        # n2nsum_param & laplacian_param
        in_edges = [[] for _ in range(total_nodes)]
        for x, y in big_edge_list:
            in_edges[y].append(x)

        n2n_rows, n2n_cols, n2n_vals = [], [], []
        lap_rows, lap_cols, lap_vals = [], [], []

        for i in range(total_nodes):
            degree = len(in_edges[i])
            if degree > 0:
                lap_vals.append(float(degree))
                lap_rows.append(i)
                lap_cols.append(i)

            for j in in_edges[i]:
                if self.aggregatorID == 0:
                    val = 1.0
                elif self.aggregatorID == 1:
                    val = 1.0 / degree
                elif self.aggregatorID == 2:
                    neighbor_degree = len(in_edges[j])
                    norm = np.sqrt(neighbor_degree + 1) * np.sqrt(degree + 1)
                    val = 1.0 / norm
                else:
                    val = 1.0

                n2n_vals.append(val)
                n2n_rows.append(i)
                n2n_cols.append(j)

                lap_vals.append(-1.0)
                lap_rows.append(i)
                lap_cols.append(j)

        self.n2nsum_param = self._to_sparse_tensor_value(
            n2n_rows, n2n_cols, n2n_vals, total_nodes, total_nodes)
        self.laplacian_param = self._to_sparse_tensor_value(
            lap_rows, lap_cols, lap_vals, total_nodes, total_nodes)

        # subgsum_param
        subg_rows, subg_cols, subg_vals = [], [], []
        start = 0
        for i in range(len(idxes)):
            t = self.avail_act_cnt[i]
            for j in range(t):
                subg_rows.append(i)
                subg_cols.append(start + j)
                subg_vals.append(1.0)
            if t > 0:
                self.subgraph_id_span.append((start, start + t - 1))
            else:
                self.subgraph_id_span.append((total_nodes, total_nodes))
            start += t

        self.subgsum_param = self._to_sparse_tensor_value(
            subg_rows, subg_cols, subg_vals, len(idxes), total_nodes)

        # rep_global or act_select
        if actions is not None:
            act_rows, act_cols, act_vals = [], [], []
            act_rows_num = len(idxes)
            act_cols_num = total_nodes
            node_offset = 0
            for i in range(len(idxes)):
                g = g_list[idxes[i]]
                idx_map = self.idx_map_list[i]
                act = actions[idxes[i]]
                if idx_map[act] >= 0 and 0 <= act < g.num_nodes:
                    act_rows.append(i)
                    act_cols.append(node_offset + idx_map[act])
                    act_vals.append(1.0)
                node_offset += self.avail_act_cnt[i]
            self.act_select = self._to_sparse_tensor_value(
                act_rows, act_cols, act_vals, act_rows_num, act_cols_num)
            self.rep_global = None
        else:
            rep_rows, rep_cols, rep_vals = [], [], []
            rep_rows_num = total_nodes
            rep_cols_num = len(idxes)
            node_offset = 0
            for i in range(len(idxes)):
                g = g_list[idxes[i]]
                idx_map = self.idx_map_list[i]
                for j in range(g.num_nodes):
                    if idx_map[j] < 0:
                        continue
                    rep_rows.append(node_offset + idx_map[j])
                    rep_cols.append(i)
                    rep_vals.append(1.0)
                node_offset += self.avail_act_cnt[i]
            self.rep_global = self._to_sparse_tensor_value(
                rep_rows, rep_cols, rep_vals, rep_rows_num, rep_cols_num)
            self.act_select = None

    def SetupTrain(self, idxes, g_list, covered, actions):
        self.SetupGraphInput(idxes, g_list, covered, actions)

    def SetupPredAll(self, idxes, g_list, covered):
        self.SetupGraphInput(idxes, g_list, covered, None)


# ------------------------------------------------------------------
# 4. Utils (replaces utils.pyx) - only inference helpers
# ------------------------------------------------------------------
class Utils:
    def __init__(self):
        self.MaxWccSzList = []

    def getRobustness(self, graph, solution):
        self.MaxWccSzList = []
        n = graph.num_nodes
        adj = [set(nei) for nei in graph.adj_list]
        removed = set()
        parent = list(range(n))
        size = [1] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]

        # Add back nodes in reverse order of removal
        for node in reversed(solution):
            removed.discard(node)
            for nei in adj[node]:
                if nei not in removed:
                    union(node, nei)
            max_cc = max((size[find(i)] for i in range(n) if i not in removed), default=1)
            self.MaxWccSzList.append(max_cc / n)

        return sum(self.MaxWccSzList) / n

    def reInsert(self, graph, solution, allVex, strategyID, reinsertEachStep):
        # Simplified reinsertion: try adding back nodes that don't increase max CC
        # This is a heuristic approximation of the original C++ implementation
        n = graph.num_nodes
        adj = [set(nei) for nei in graph.adj_list]
        removed = set(solution)
        keep = []

        # Work with a copy
        current_removed = set(solution)
        for node in solution:
            current_removed.discard(node)
            # Check if adding back is OK
            # Use DSU to find max CC
            parent = list(range(n))
            size = [1] * n

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    if size[ra] < size[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    size[ra] += size[rb]

            for i in range(n):
                if i in current_removed:
                    continue
                for j in adj[i]:
                    if j not in current_removed:
                        union(i, j)

            max_cc = max((size[find(i)] for i in range(n) if i not in current_removed), default=0)
            # Allow if max CC is small enough (heuristic)
            if max_cc <= 1:
                # Keep it out (reinserted)
                pass
            else:
                current_removed.add(node)
                keep.append(node)

        final_sol = keep + list(allVex)
        return final_sol


# ------------------------------------------------------------------
# 5. FINDER_Pure (replaces FINDER.pyx, inference only)
# ------------------------------------------------------------------
class FINDER_Pure:
    def __init__(self):
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = 'barabasi_albert'
        self.TrainSet = GSet()
        self.TestSet = GSet()
        self.inputs = {}
        self.reg_hidden = REG_HIDDEN
        self.utils = Utils()

        self.IsHuberloss = False
        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False
        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False

        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list = []
        self.g_list = []
        self.pred = []

        self.nStepReplayMem = None  # not needed for inference

        for i in range(num_env):
            self.env_list.append(MvcEnv(NUM_MAX))
            self.g_list.append(Graph())

        self.test_env = MvcEnv(NUM_MAX)

        self.action_select = tf.sparse_placeholder(tf.float32, name="action_select")
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global")
        self.n2nsum_param = tf.sparse_placeholder(tf.float32, name="n2nsum_param")
        self.laplacian_param = tf.sparse_placeholder(tf.float32, name="laplacian_param")
        self.subgsum_param = tf.sparse_placeholder(tf.float32, name="subgsum_param")
        self.target = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name="target")
        self.aux_input = tf.placeholder(tf.float32, name="aux_input")

        if self.IsPrioritizedSampling:
            self.ISWeights = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name='IS_weights')

        self.loss, self.trainStep, self.q_pred, self.q_on_all, self.Q_param_list = self.BuildNet()
        self.lossT, self.trainStepT, self.q_predT, self.q_on_allT, self.Q_param_listT = self.BuildNet()
        self.copyTargetQNetworkOperation = [a.assign(b) for a, b in zip(self.Q_param_listT, self.Q_param_list)]
        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        self.saver = tf.train.Saver(max_to_keep=None)
        config = tf.ConfigProto(device_count={"CPU": 8},
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

    def BuildNet(self):
        w_n2l = tf.Variable(tf.truncated_normal([2, self.embedding_size], stddev=initialization_stddev), tf.float32)
        p_node_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
        if embeddingMethod == 1:
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            p_node_conv3 = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)

        if self.reg_hidden > 0:
            h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            last_w = h2_weight
        else:
            h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            last_w = h1_weight

        cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)

        nodes_size = tf.shape(self.n2nsum_param)[0]
        node_input = tf.ones((nodes_size, 2))

        y_nodes_size = tf.shape(self.subgsum_param)[0]
        y_node_input = tf.ones((y_nodes_size, 2))

        input_message = tf.matmul(tf.cast(node_input, tf.float32), w_n2l)
        input_potential_layer = tf.nn.relu(input_message)

        y_input_message = tf.matmul(tf.cast(y_node_input, tf.float32), w_n2l)
        y_input_potential_layer = tf.nn.relu(y_input_message)

        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        y_cur_message_layer = y_input_potential_layer
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        lv = 0
        while lv < max_bp_iter:
            lv = lv + 1
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param, tf.float32), cur_message_layer)
            node_linear = tf.matmul(n2npool, p_node_conv)
            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param, tf.float32), cur_message_layer)
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)

            if embeddingMethod == 0:
                merged_linear = tf.add(node_linear, input_message)
                cur_message_layer = tf.nn.relu(merged_linear)
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))

                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        self.node_embedding = cur_message_layer
        y_potential = y_cur_message_layer
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)

        temp = tf.matmul(tf.expand_dims(action_embed, axis=2), tf.expand_dims(y_potential, axis=1))
        Shape = tf.shape(action_embed)
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product, [Shape[0], 1]), [Shape[0], Shape[1], 1])), Shape)

        last_output = embed_s_a
        if self.reg_hidden > 0:
            hidden = tf.matmul(embed_s_a, h1_weight)
            last_output = tf.nn.relu(hidden)

        last_output = tf.concat([last_output, self.aux_input], 1)
        q_pred = tf.matmul(last_output, last_w)

        loss_recons = 2 * tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param, tf.float32), cur_message_layer)))
        edge_num = tf.sparse_reduce_sum(tf.cast(self.n2nsum_param, tf.float32))
        loss_recons = tf.divide(loss_recons, edge_num)

        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred)

        loss = loss_rl + Alpha * loss_recons
        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)

        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2), tf.expand_dims(rep_y, axis=1))
        Shape1 = tf.shape(cur_message_layer)
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product, [Shape1[0], 1]), [Shape1[0], Shape1[1], 1])), Shape1)

        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            last_output = tf.nn.relu(hidden)

        rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)
        last_output = tf.concat([last_output, rep_aux], 1)
        q_on_all = tf.matmul(last_output, last_w)

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables()

    def InsertGraph(self, g, is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        return prepareBatchGraph.idx_map_list

    def SetupTrain(self, idxes, g_list, covered, actions):
        self.m_y = None  # placeholder
        self.inputs['target'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat

    def Predict(self, g_list, covered, isSnapSnot):
        n_graphs = len(g_list)
        pred = []
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            my_dict = {}
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])

            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict=my_dict)
            else:
                result = self.session.run([self.q_on_all], feed_dict=my_dict)
            raw_output = result[0]
            pos = 0
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j - i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithCurrentQNet(self, g_list, covered):
        return self.Predict(g_list, covered, False)

    def PredictWithSnapshot(self, g_list, covered):
        return self.Predict(g_list, covered, True)

    def TakeSnapShot(self):
        self.session.run(self.UpdateTargetQNetwork)

    def SaveModel(self, model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    def LoadModel(self, model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def GenNetwork(self, g):
        edges = list(g.edges())
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return Graph(len(g.nodes()), len(edges), A, B)

    def argMax(self, scores):
        pos = -1
        best = -10000000
        for i in range(len(scores)):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos

    def Max(self, scores):
        best = -10000000
        for i in range(len(scores)):
            if scores[i] > best:
                best = scores[i]
        return best

    def EvaluateRealData(self, g, stepRatio=0.0025):
        solution_time = 0.0
        print('testing')
        print('number of nodes:%d' % (nx.number_of_nodes(g)))
        print('number of edges:%d' % (nx.number_of_edges(g)))
        print('stepRatio:%f' % stepRatio)

        if stepRatio > 0:
            step = max([int(stepRatio * nx.number_of_nodes(g)), 1])
        else:
            step = 1
        print('step:%f' % stepRatio)

        self.InsertGraph(g, is_test=True)
        t1 = time.time()
        solution = self.GetSolution(0, step)
        t2 = time.time()
        solution_time = (t2 - t1)
        self.ClearTestGraphs()
        return solution, solution_time

    def GetSolution(self, gid, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        iter = 0
        while not self.test_env.isTerminal():
            print('Iteration:%d' % iter)
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    continue
        return sol

    def EvaluateSol(self, g, solution, strategyID=0, reInsertStep=20):
        g_inner = self.GenNetwork(g)
        print('number of nodes:%d' % nx.number_of_nodes(g))
        print('number of edges:%d' % nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = solution
        print('number of sol nodes:%d' % len(sol))
        sol_left = list(set(nodes) ^ set(sol))
        if strategyID > 0:
            start = time.time()
            if reInsertStep > 0 and reInsertStep < 1:
                step = max([int(reInsertStep * nx.number_of_nodes(g)), 1])
            else:
                step = reInsertStep
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, step)
            end = time.time()
            print('reInsert time:%.6f' % (end - start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        print('number of solution nodes:%d' % len(solution))
        Robustness = self.utils.getRobustness(g_inner, solution)
        MaxCCList = self.utils.MaxWccSzList
        return solution, Robustness, MaxCCList

    def GetSol(self, gid, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        while not self.test_env.isTerminal():
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes) ^ set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness, sol
