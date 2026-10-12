#include "scancontext/Scancontext.h"

// namespace SC2
// {

void coreImportTest (void)
{
    cout << "scancontext lib is successfully imported." << endl;
} // coreImportTest


float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}


float xy2theta( const float & _x, const float & _y )
{
    if ( (_x >= 0) & (_y >= 0)) 
        return (180/M_PI) * atan(_y / _x);

    if ( (_x < 0) & (_y >= 0)) 
        return 180 - ( (180/M_PI) * atan(_y / (-_x)) );

    if ( (_x < 0) & (_y < 0)) 
        return 180 + ( (180/M_PI) * atan(_y / _x) );

    if ( (_x >= 0) & (_y < 0))
        return 360 - ( (180/M_PI) * atan((-_y) / _x) );
} // xy2theta


MatrixXd circshift( MatrixXd &_mat, int _num_shift )
{
    // shift columns to right direction 
    assert(_num_shift >= 0);

    if( _num_shift == 0 )
    {
        MatrixXd shifted_mat( _mat );
        return shifted_mat; // Early return 
    }

    MatrixXd shifted_mat = MatrixXd::Zero( _mat.rows(), _mat.cols() );
    for ( int col_idx = 0; col_idx < _mat.cols(); col_idx++ )
    {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }

    return shifted_mat;

} // circshift


std::vector<float> eig2stdvec( MatrixXd _eigmat )
{
    std::vector<float> vec( _eigmat.data(), _eigmat.data() + _eigmat.size() );
    return vec;
} // eig2stdvec


double SCManager::distDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    // 计算两帧描述子的"逐列余弦距离"（已对齐后调用）
    // 返回值越小 = 越相似
    int num_eff_cols = 0; // 有效列计数（排除"空列"）
    double sum_sector_similarity = 0;
    for ( int col_idx = 0; col_idx < _sc1.cols(); col_idx++ )
    {
        VectorXd col_sc1 = _sc1.col(col_idx);   // 第1帧的第 col_idx 个扇区（一列）
        VectorXd col_sc2 = _sc2.col(col_idx);   // 第2帧对应的扇区

        // 某一列为空（norm==0，即该扇区没点）则跳过，不计入相似度
        if( (col_sc1.norm() == 0) | (col_sc2.norm() == 0) )
            continue; // don't count this sector pair.

        // 单列余弦相似度 = 两向量夹角的余弦值（1=完全一样，0=正交，-1=相反）
        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;  // 累加
        num_eff_cols = num_eff_cols + 1;
    }

    double sc_sim = sum_sector_similarity / num_eff_cols;  // 所有有效列的平均相似度
    return 1.0 - sc_sim;  // 转成"距离"：相似度1→距离0（完全一样），相似度0→距离1

} // distDirectSC


int SCManager::fastAlignUsingVkey( MatrixXd & _vkey1, MatrixXd & _vkey2)
{
    // 用 sector key（1×60 行向量）快速找"两帧的最佳角度对齐量"
    // 原理：把 vkey2 循环移位，找与 vkey1 最接近的移位量（即角度差）
    int argmin_vkey_shift = 0;         // 最佳移位量
    double min_veky_diff_norm = 10000000;  // 初始化为很大的数
    for ( int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++ )  // 遍历所有可能移位
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);  // vkey2 右移 shift_idx 个扇区

        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;            // 差向量

        double cur_diff_norm = vkey_diff.norm();                // 差的模长（越小越对齐）
        if( cur_diff_norm < min_veky_diff_norm )
        {
            argmin_vkey_shift = shift_idx;      // 记录更优的移位
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;   // 返回让两帧 sector key 最接近的移位量

} // fastAlignUsingVkey


std::pair<double, int> SCManager::distanceBtnScanContext( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    // 比较两帧 ScanContext 描述子的相似度，并找最优旋转对齐
    // 返回 pair: (最小距离, 最佳扇区移位量)
    // 距离越小 = 越相似；扇区移位量可换算成偏航差

    // ---- 第1步：用 sector key 快速猜"大致对齐角度" ----
    // sector key（每列均值，会随旋转变）比完整描述子便宜，
    // 用它循环移位找出让两帧最接近的角度，缩小精搜范围
    MatrixXd vkey_sc1 = makeSectorkeyFromScancontext( _sc1 );
    MatrixXd vkey_sc2 = makeSectorkeyFromScancontext( _sc2 );
    int argmin_vkey_shift = fastAlignUsingVkey( vkey_sc1, vkey_sc2 );  // 猜测的最优移位量

    // 只在大致角度 ±SEARCH_RADIUS 内精搜（不穷举全部 60 个扇区，省计算）
    const int SEARCH_RADIUS = round( 0.5 * SEARCH_RATIO * _sc1.cols() ); // a half of search range
    std::vector<int> shift_idx_search_space { argmin_vkey_shift };  // 候选移位列表，从猜测值开始
    for ( int ii = 1; ii < SEARCH_RADIUS + 1; ii++ )
    {
        // 往左右各扩展 ii 个移位量（% cols 保证循环不越界）
        shift_idx_search_space.push_back( (argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols() );
        shift_idx_search_space.push_back( (argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols() );
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    // ---- 第2步：对每个候选移位，做精确的逐列余弦距离 ----
    int argmin_shift = 0;         // 最优移位量
    double min_sc_dist = 10000000;  // 最小距离（初始化为很大）
    for ( int num_shift: shift_idx_search_space )
    {
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);  // 把第2帧按此移位量旋转对齐
        double cur_sc_dist = distDirectSC( _sc1, sc2_shifted );  // 逐列余弦距离
        if( cur_sc_dist < min_sc_dist )
        {
            argmin_shift = num_shift;   // 记录更优的对齐量
            min_sc_dist = cur_sc_dist;  // 记录更小的距离
        }
    }

    return make_pair(min_sc_dist, argmin_shift);  // (相似度距离, 最优扇区对齐量)

} // distanceBtnScanContext


MatrixXd SCManager::makeScancontext( pcl::PointCloud<SCPointType> & _scan_down )
{
    TicTocV2 t_making_desc;

    int num_pts_scan_down = _scan_down.points.size();

    // 初始化描述子：20环 × 60扇区 的矩阵，全部置为 NO_POINT(-1000) 作为"空格子"占位
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

    SCPointType pt;
    float azim_angle, azim_range; // within 2d plane（极坐标：角度+半径）
    int ring_idx, sctor_idx;      // 该点所在的环号、扇区号（1 开头）
    for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        pt.x = _scan_down.points[pt_idx].x;
        pt.y = _scan_down.points[pt_idx].y;
        pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0). 抬高 z 使高度非负

        // 把 xy 直角坐标转成极坐标：水平距离 + 方位角(0~360°)
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        azim_angle = xy2theta(pt.x, pt.y);

        // 超出最大半径范围的点直接跳过（不纳入描述子）
        if( azim_range > PC_MAX_RADIUS )
            continue;

        // 由距离算出环号（每 PC_MAX_RADIUS/PC_NUM_RING 米一环），ceil 向上取整、min/max 夹在 [1, PC_NUM_RING]
        ring_idx = std::max( std::min( PC_NUM_RING, int(ceil( (azim_range / PC_MAX_RADIUS) * PC_NUM_RING )) ), 1 );
        // 由方位角算出扇区号（每 360/PC_NUM_SECTOR 度一扇区），同理夹在 [1, PC_NUM_SECTOR]
        sctor_idx = std::max( std::min( PC_NUM_SECTOR, int(ceil( (azim_angle / 360.0) * PC_NUM_SECTOR )) ), 1 );

        // 每个格子只保留"最高点的高度"：若当前点的 z 比格子里的值大，就覆盖（-1 是因为 C++ 下标从 0 开始）
        if ( desc(ring_idx-1, sctor_idx-1) < pt.z )
            desc(ring_idx-1, sctor_idx-1) = pt.z; // 更新为最大 z 值
    }

    // 把"没有点的空格子"从 -1000 改成 0（后面算余弦相似度时，0 才不会干扰）
    for ( int row_idx = 0; row_idx < desc.rows(); row_idx++ )
        for ( int col_idx = 0; col_idx < desc.cols(); col_idx++ )
            if( desc(row_idx, col_idx) == NO_POINT )
                desc(row_idx, col_idx) = 0;

    t_making_desc.toc("PolarContext making");

    return desc;
} // SCManager::makeScancontext


MatrixXd SCManager::makeRingkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: rowwise mean vector
    */
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for ( int row_idx = 0; row_idx < _desc.rows(); row_idx++ )
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();
    }

    return invariant_key;
} // SCManager::makeRingkeyFromScancontext


MatrixXd SCManager::makeSectorkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* columnwise mean vector
     * 按列（角度扇区）取均值，得到 1×PC_NUM_SECTOR 的 sector key，
     * 会随旋转变化，用于快速估计两帧间的偏航差。
    */
    Eigen::MatrixXd variant_key(1, _desc.cols());
    for ( int col_idx = 0; col_idx < _desc.cols(); col_idx++ )
    {
        Eigen::MatrixXd curr_col = _desc.col(col_idx);
        variant_key(0, col_idx) = curr_col.mean();
    }

    return variant_key;
} // SCManager::makeSectorkeyFromScancontext


const Eigen::MatrixXd& SCManager::getConstRefRecentSCD(void)
{
    return polarcontexts_.back();
}


void SCManager::saveScancontextAndKeys( Eigen::MatrixXd _scd )
{
    Eigen::MatrixXd ringkey = makeRingkeyFromScancontext( _scd );
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext( _scd );
    std::vector<float> polarcontext_invkey_vec = eig2stdvec( ringkey );

    polarcontexts_.push_back( _scd ); 
    polarcontext_invkeys_.push_back( ringkey );
    polarcontext_vkeys_.push_back( sectorkey );
    polarcontext_invkeys_mat_.push_back( polarcontext_invkey_vec );
} // SCManager::makeAndSaveScancontextAndKeys


void SCManager::makeAndSaveScancontextAndKeys( pcl::PointCloud<SCPointType> & _scan_down )
{
    // 1. 把当前这帧点云压缩成 ScanContext 描述子（20环×60扇区的高度图）
    //    sc（描述子矩阵）里的值是"每个格子的最高点高度"
    Eigen::MatrixXd sc = makeScancontext(_scan_down); // v1

    // 2. 从描述子生成 ring key：每一行(环)取平均值 → 20×1 向量
    //    ring key 对"整帧旋转"不敏感（环是同心圆，旋转不改变环内点的分布），用于快速粗筛
    Eigen::MatrixXd ringkey = makeRingkeyFromScancontext( sc );

    // 3. 从描述子生成 sector key：每一列(扇区)取平均值 → 1×60 向量
    //    sector key 会随旋转而整体移位，用于细匹配时找对齐角度
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext( sc );

    // 4. 把 ring key 从 Eigen 矩阵转成 std::vector<float>（K-D 树搜索需要这种格式）
    std::vector<float> polarcontext_invkey_vec = eig2stdvec( ringkey );

    // 5. 把当前帧的 描述子 + 两种 key 全部存入数据库（供以后回环检测用）
    polarcontexts_.push_back( sc );                     // 完整描述子（存原始，用于精确相似度比较）
    polarcontext_invkeys_.push_back( ringkey );         // ring key（旋转不变，用于粗筛）
    polarcontext_vkeys_.push_back( sectorkey );         // sector key（用于旋转对齐）
    polarcontext_invkeys_mat_.push_back( polarcontext_invkey_vec ); // ring key 的 vector 版（建 K-D 树用）
} // SCManager::makeAndSaveScancontextAndKeys

void SCManager::setSCdistThres(double _new_thres)
{
    SC_DIST_THRES = _new_thres;
} // SCManager::setThres

void SCManager::setMaximumRadius(double _max_r)
{
    PC_MAX_RADIUS = _max_r;
} // SCManager::setMaximumRadius

std::pair<int, float> SCManager::detectLoopClosureIDBetweenSession (std::vector<float>& _curr_key, Eigen::MatrixXd& _curr_desc)
{
    int loop_id { -1 }; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")

    auto& curr_key = _curr_key;
    auto& curr_desc = _curr_desc; // current observation (query)

    // step 0: if first, construct the tree in batch
    if( ! is_tree_batch_made ) // run only once
    {
        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.assign( polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end() ) ;

        polarcontext_tree_batch_.reset(); 
        polarcontext_tree_batch_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */ );

        is_tree_batch_made = true; // for running this block only once
    }
        
    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    // step 1: knn search
    std::vector<size_t> candidate_indexes( NUM_CANDIDATES_FROM_TREE ); 
    std::vector<float> out_dists_sqr( NUM_CANDIDATES_FROM_TREE );

    nanoflann::KNNResultSet<float> knnsearch_result( NUM_CANDIDATES_FROM_TREE );
    knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );
    polarcontext_tree_batch_->index->findNeighbors( knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10) ); // error here

    // step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
    TicTocV2 t_calc_dist;   
    for ( int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++ )
    {
        MatrixXd polarcontext_candidate = polarcontexts_[ candidate_indexes[candidate_iter_idx] ];
        std::pair<double, int> sc_dist_result = distanceBtnScanContext( curr_desc, polarcontext_candidate ); 
        
        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if( candidate_dist < min_dist )
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }
    t_calc_dist.toc("Distance calc");

    // step 3: similarity threshold
    if( min_dist < SC_DIST_THRES )
        loop_id = nn_idx; 

    // To do: return also nn_align (i.e., yaw diff)
    float yaw_diff_rad = deg2rad(nn_align * PC_UNIT_SECTORANGLE);
    std::pair<int, float> result {loop_id, yaw_diff_rad};

    return result;

} // SCManager::detectLoopClosureIDBetweenSession


std::pair<int, float> SCManager::detectLoopClosureID ( void )
{
    int loop_id { -1 }; // 初始化为 -1，-1 表示没找到回环（等同 LeGO-LOAM 的 closestHistoryFrameID）

    auto curr_key = polarcontext_invkeys_mat_.back(); // 当前帧的 ring key（用作查询）
    auto curr_desc = polarcontexts_.back();           // 当前帧的完整描述子（用作查询）

    /*
     * 第1步：用 ring key 在 K-D 树里粗筛候选
     */
    // 历史帧太少（不足 NUM_EXCLUDE_RECENT+1 帧）时直接返回"无回环"：
    // 早期还没有足够"久远"的帧，检了也白检，还容易误判
    if( (int)polarcontext_invkeys_mat_.size() < NUM_EXCLUDE_RECENT + 1)
    {
        std::pair<int, float> result {loop_id, 0.0};
        return result; // Early return
    }

    // K-D 树重建（不需要每次都重建，按周期来以省计算）
    if( tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        TicTocV2 t_tree_construction;

        // 搜索集 = 全部历史帧，去掉最近的 NUM_EXCLUDE_RECENT 帧
        //（排除"太近、本来就像"的帧，防止把连续前进的相邻帧误判成回环）
        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.assign( polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end() - NUM_EXCLUDE_RECENT ) ;

        // 用搜索集重建 K-D 树（nanoflann 的构造函数内部会建索引）
        polarcontext_tree_.reset();
        polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */ );
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
        t_tree_construction.toc("Tree construction");
    }
    tree_making_period_conter = tree_making_period_conter + 1;

    double min_dist = 10000000; // 初始化为一个很大的数（当前最佳距离）
    int nn_align = 0;           // 最佳候选的扇区对齐量（求偏航差用）
    int nn_idx = 0;             // 最佳候选的帧编号

    // 在 K-D 树里做 KNN 搜索，找最相似的 NUM_CANDIDATES_FROM_TREE 个历史帧
    std::vector<size_t> candidate_indexes( NUM_CANDIDATES_FROM_TREE );
    std::vector<float> out_dists_sqr( NUM_CANDIDATES_FROM_TREE );

    TicTocV2 t_tree_search;
    nanoflann::KNNResultSet<float> knnsearch_result( NUM_CANDIDATES_FROM_TREE );
    knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );
    polarcontext_tree_->index->findNeighbors( knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10) );
    t_tree_search.toc("Tree search");

    /*
     * 第2步：对粗筛出的候选，用完整描述子做精确的逐列余弦距离
     */
    TicTocV2 t_calc_dist;
    for ( int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++ )
    {
        MatrixXd polarcontext_candidate = polarcontexts_[ candidate_indexes[candidate_iter_idx] ];
        // 精确距离 + 最优扇区对齐量（distanceBtnScanContext 内部会做旋转对齐）
        std::pair<double, int> sc_dist_result = distanceBtnScanContext( curr_desc, polarcontext_candidate );

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        // 保留距离最小的那个候选
        if( candidate_dist < min_dist )
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }
    t_calc_dist.toc("Distance calc");

    /*
     * 回环阈值判定：距离小于阈值才认定为真回环
     */
    if( min_dist < SC_DIST_THRES )
    {
        loop_id = nn_idx;

        // std::cout.precision(3);
        cout << "[Loop found] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
        // cout << "[Loop found] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg." << endl;
    }
    else
    {
        std::cout.precision(3);
        cout << "[Not loop] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
        // cout << "[Not loop] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg." << endl;
    }

    // 把扇区对齐量换算成偏航角差值（弧度）返回，供回环约束用
    float yaw_diff_rad = deg2rad(nn_align * PC_UNIT_SECTORANGLE);
    std::pair<int, float> result {loop_id, yaw_diff_rad};

    return result;

} // SCManager::detectLoopClosureID

// } // namespace SC2