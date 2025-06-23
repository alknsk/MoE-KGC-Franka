# utils/kg_visualization.py
from pyvis.network import Network
import networkx as nx
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


class KnowledgeGraphVisualizer:
    """Knowledge graph visualization using pyvis"""

    def __init__(self, config):
        self.config = config

        # 节点类型到颜色的映射（基于您的配色方案）
        self.node_colors = {
            'action': '#4CAF50',  # 绿色 - 动作/物理组件
            'object': '#2196F3',  # 蓝色 - 对象/本体类
            'task': '#FF9800',  # 橙色 - 任务
            'constraint': '#9C27B0',  # 紫色 - 约束/参数
            'safety': '#F44336',  # 红色 - 安全限制
            'spatial': '#00BCD4',  # 青色 - 空间关系
            'temporal': '#FFC107',  # 黄色 - 时间关系
            'semantic': '#795548',  # 棕色 - 语义关系
        }

        # 节点形状映射
        self.node_shapes = {
            'action': 'box',
            'object': 'ellipse',
            'task': 'diamond',
            'constraint': 'triangle',
            'safety': 'star',
            'spatial': 'dot',
            'temporal': 'square',
            'semantic': 'database'
        }

    def create_from_model_output(self,
                                 graph: nx.Graph,
                                 entities: Dict[str, List],
                                 relations: List[Dict],
                                 predictions: Optional[Dict] = None) -> Network:
        """从模型输出创建可视化"""

        # 创建pyvis网络
        net = Network(
            notebook=True,
            height="1200px",
            width="100%",
            bgcolor="#222222",
            font_color="white"
        )

        # 添加实体节点
        for entity_type, entity_list in entities.items():
            color = self.node_colors.get(entity_type, '#888888')
            shape = self.node_shapes.get(entity_type, 'ellipse')

            for entity in entity_list:
                node_id = entity.get('id', str(entity))
                node_name = entity.get('name', entity.get('text', str(entity)))

                # 添加预测置信度信息
                title = f"{node_name}"
                if predictions and node_id in predictions:
                    confidence = predictions[node_id].get('confidence', 0)
                    title += f"\n置信度: {confidence:.3f}"

                net.add_node(
                    node_id,
                    label=node_name[:20] + '...' if len(node_name) > 20 else node_name,
                    title=title,
                    color=color,
                    shape=shape,
                    group=entity_type
                )

        # 添加关系边
        edge_colors = {
            'follows': '#8BC34A',
            'interacts_with': '#FF5722',
            'subtask_of': '#3F51B5',
            'depends_on': '#E91E63',
            'contextual': '#9E9E9E',
            'physical_connection': '#4CAF50',
            'controls': '#FF9800',
            'limits': '#F44336'
        }

        for relation in relations:
            head = relation.get('head')
            tail = relation.get('tail')
            rel_type = relation.get('type', 'unknown')

            if head and tail:
                edge_color = edge_colors.get(rel_type, '#666666')

                # 添加关系预测信息
                edge_title = rel_type
                if predictions and f"{head}_{tail}" in predictions:
                    confidence = predictions[f"{head}_{tail}"].get('confidence', 0)
                    edge_title += f" (置信度: {confidence:.3f})"

                net.add_edge(
                    str(head),
                    str(tail),
                    label=rel_type,
                    title=edge_title,
                    color=edge_color,
                    arrows={'to': {'enabled': True}}
                )

        # 设置物理布局
        self._configure_physics(net)

        return net

    def _configure_physics(self, net: Network):
        """配置物理引擎参数"""
        net.set_options("""
        {
          "nodes": {
            "font": {"size": 18},
            "scaling": {"min": 20, "max": 40}
          },
          "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}}
          },
          "physics": {
            "hierarchicalRepulsion": {
              "springLength": 200,
              "nodeDistance": 300,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "solver": "hierarchicalRepulsion"
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "zoomView": true,
            "dragView": true
          }
        }
        """)

    def add_franka_specific_components(self, net: Network):
        """添加Franka特定的组件（基于您的代码）"""
        # 物理结构
        physical_nodes = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
            "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7"
        ]

        for node in physical_nodes:
            net.add_node(
                node,
                label=node,
                title=f"Franka {node}",
                color=self.node_colors['action'],
                shape='box',
                group='PhysicalComponent'
            )

        # 添加物理连接
        physical_edges = [
            ("link0", "joint1"), ("joint1", "link1"),
            ("link1", "joint2"), ("joint2", "link2"),
            ("link2", "joint3"), ("joint3", "link3"),
            ("link3", "joint4"), ("joint4", "link4"),
            ("link4", "joint5"), ("joint5", "link5"),
            ("link5", "joint6"), ("joint6", "link6"),
            ("link6", "joint7"), ("joint7", "link7")
        ]

        for src, dst in physical_edges:
            net.add_edge(src, dst,
                         label="connected",
                         color="#8BC34A",
                         title="Physical Connection")

    def save_with_legend(self, net: Network, output_path: str):
        """保存带图例的HTML"""
        # 保存基础HTML
        net.save_graph(output_path)

        # 创建图例HTML
        legend_html = self._create_legend_html()

        # 读取并修改HTML
        with open(output_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 插入图例和额外的控制面板
        control_panel = self._create_control_panel()
        html_content = html_content.replace("</body>",
                                            legend_html + control_panel + "</body>")

        # 保存修改后的HTML
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _create_legend_html(self) -> str:
        """创建图例HTML"""
        legend_items = []

        # 节点类型图例
        for entity_type, color in self.node_colors.items():
            legend_items.append(
                f'<p><span style="display: inline-block; width: 20px; '
                f'height: 20px; background: {color};"></span> '
                f'{entity_type.capitalize()}</p>'
            )

        legend_html = f"""
        <div style="position: absolute; top: 10px; left: 10px; 
                    background: rgba(255,255,255,0.9); padding: 15px; 
                    border-radius: 5px; z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
          <h3 style="margin-top: 0;">图例 / Legend</h3>
          <h4>节点类型 / Node Types</h4>
          {''.join(legend_items)}
          <h4>专家激活 / Expert Activation</h4>
          <div id="expert-activation"></div>
        </div>
        """

        return legend_html

    def _create_control_panel(self) -> str:
        """创建控制面板"""
        return """
        <div style="position: absolute; top: 10px; right: 10px; 
                    background: rgba(255,255,255,0.9); padding: 15px; 
                    border-radius: 5px; z-index: 100;">
          <h3>控制面板 / Control Panel</h3>
          <button onclick="network.fit()">重置视图</button>
          <button onclick="highlightExpertNodes('action')">突出动作专家</button>
          <button onclick="highlightExpertNodes('safety')">突出安全专家</button>
          <button onclick="showAllNodes()">显示全部</button>
        </div>

        <script>
        function highlightExpertNodes(expertType) {
            var nodes = network.body.data.nodes;
            var updateArray = [];
            nodes.forEach(function(node) {
                if (node.group === expertType) {
                    updateArray.push({id: node.id, size: 40, borderWidth: 3});
                } else {
                    updateArray.push({id: node.id, size: 20, borderWidth: 1});
                }
            });
            nodes.update(updateArray);
        }

        function showAllNodes() {
            network.fit();
        }
        </script>
        """