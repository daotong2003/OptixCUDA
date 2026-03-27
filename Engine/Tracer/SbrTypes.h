#pragma once
#include <vector_types.h>
#include <cstdint>

namespace Engine {
	namespace Tracer {
		// 【配置参数】定义 SBR 允许的最大弹跳深度
		constexpr int MAX_BOUNCE_DEPTH = 5;

		// =========================================================
		// 1. 射线路径节点 (记录单次弹跳的微观/宏观物理快照)
		// =========================================================
#pragma pack(push, 1)
		struct RayPathNode {
			float3 position;      // 宏观坐标：射线在 3D 空间中的实际交点
			float3 normal;        // 微观物理：通过你的校准算法提取的高精度法线
			float3 inDirection;   // 几何物理：入射射线方向
			float3 outDirection;  // 几何物理：反射出的射线方向
			float  distance;      // 空间物理：从上一节点到当前节点的传播距离
			uint8_t material_id;  // 材质标签：决定衰减或反射系数
			bool isValid;         // 标志位：当前节点是否记录了有效碰撞
		};
#pragma pack(pop)

		// =========================================================
		// 2. 完整射线路径 (包含多次弹跳记录的完整拓扑结构)
		// =========================================================
		struct RayPath {
			RayPathNode nodes[MAX_BOUNCE_DEPTH]; // 静态数组，确保显存分配是连续且安全的
			int nodeCount;                       // 当前射线实际发生的弹跳次数
			bool isEscaped;                      // 标志位：射线最终是否射向太空未命中
		};

		// 拓扑节点：不记录坐标，只记录“身份证号”
		struct TopologyNode {
			int32_t instance_id; // 击中物体的 ID (例如车辆 1，大楼 2)
			// [必须是这行！] 击中的平面标签，取代原先的 prim_idx
			int32_t plane_label;
		};

		// 完整拓扑路径：一条射线一生的碰撞记录
		struct PathTopology {
			TopologyNode nodes[MAX_BOUNCE_DEPTH];
			int nodeCount; // 发生碰撞的次数
			bool hitRx;    // 最终是否成功落入 Rx 捕获球！
		};
		// =========================================================
		// 3. [阶段二专属新增] 镜像法精确路径解析
		// =========================================================

		// 理想平面方程：Ax + By + Cz + D = 0
		struct PlaneEquation {
			float3 normal; // 法向量 (A, B, C)，必须是单位向量
			float d;       // 距离常数 D
		};

		// 绝对精确的物理路径 (包含 Tx, 所有反射点, Rx)
		struct ExactPath {
			// 顶点数组：最大容量 = 最大弹跳数 + 起点 + 终点
			float3 vertices[MAX_BOUNCE_DEPTH + 2];
			int vertexCount; // 实际顶点数量 (如 1次弹跳 = 3个顶点)
			bool isValid;    // 路径是否有效 (防平行未相交等数学异常)
		};

		// GPU 端用于快速二分查找的平面字典条目
		struct PlaneDictEntry {
			int32_t label;          // 平面标签
			PlaneEquation eq;       // 平面方程
		};
	} // namespace Tracer
} // namespace Engine