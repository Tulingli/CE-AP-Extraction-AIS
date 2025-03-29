import sys
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPolygon

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
'锚地区域划分'
def to_region(data):
    geometry = [Point(x, y) for x, y in zip(data['LON'], data['LAT'])]
    gdf_points = gpd.GeoDataFrame(data, geometry=geometry)
    buffers = [point.buffer(0.001) for point in gdf_points['geometry']]
    # 合并相连但不相重叠的缓冲区
    merged_buffer = unary_union(buffers)
    individual_buffers = [geom for geom in merged_buffer.geoms]
    # df=pd.DataFrame(individual_buffers,columns=['11'])
    # df.to_csv('11.csv',mode='w',index=False)
    # sys.exit()
    groups = gdf_points.groupby('T')
    data_all=pd.DataFrame()
    points1 = [(-118.23903719027486, 33.74235205122199), (-118.19825253038783, 33.74346233135964),
               (-118.19752423288985, 33.7239798757773), (-118.23891580735854, 33.723273175088494)]
    polygon1 = Polygon(points1)
    points2 = [(-118.18550732151598, 33.73801172717929), (-118.14095979122271, 33.737708905689445),
               (-118.13998872789206, 33.72771519689133), (-118.18575008734864, 33.72751329178986)]
    polygon2 = Polygon(points2)
    points3 = [(-118.18686242682018, 33.754714546167115), (-118.16751960350305, 33.754714546167115),
               (-118.16419956666502, 33.74067157483337), (-118.18527458311505, 33.73935117721836)]
    polygon3 = Polygon(points3)
    points4 = [(-118.23016725686253, 33.714259760716864), (-118.19480164706624, 33.716901307812684),
               (-118.18902766995663, 33.68195427311284), (-118.22626982231354, 33.67859092220169)]
    polygon4 = Polygon(points4)
    points5 = [(-118.170319,33.72092701), (-118.117104, 33.72092701),
               (-118.0916981, 33.67093961), (-118.1184773, 33.62549729),(-118.183022,33.62892773)]
    polygon5 = Polygon(points5)
    points6 = [(-118.0951314, 33.63007118), (-118.051186, 33.65322282),
               (-118.0117039, 33.59719089), (-118.0707554, 33.57688462)]
    polygon6 = Polygon(points6)
    polygons = {'A':[polygon1],'B':[polygon2],'C':[polygon3],'D':[polygon4],'E':[polygon5],'F':[polygon6]}
    for i,buffer in enumerate(individual_buffers):
        for j, (key, polygon) in enumerate(polygons.items()):
            intersection_result = individual_buffers[i].intersection(polygon[0]).area / min(individual_buffers[i].area,
                                                                                  polygon[0].area)
            if intersection_result>0.2:
                polygons[key].extend([buffer])

    for key, group in groups:
        for i,(key,buffers) in enumerate(polygons.items()):
            for buffer in buffers:
                if buffer.contains(group['geometry'].iloc[0]):
                    df=group[['MMSI','LON','LAT', 'Length', 'Width', 'type_code', 'VesselType','T','firsttime','endtime','time_interval', 'YYMM']].copy()
                    df['Q']=key
                    data_all=pd.concat([data_all,df])
    return data_all
'计算重叠度'
#第一重叠度
def convex_hull_overlap(polygon1, polygon2):
    polygon1=polygon1.convex_hull
    polygon2 = polygon2.convex_hull

    intersection = polygon2.intersection(polygon1)
    overlap_ratio = intersection.area / polygon1.area

    return overlap_ratio
#最小重叠度
def convex_hull_overlap_min(polygon1, polygon2):
    polygon1=polygon1.convex_hull
    polygon2 = polygon2.convex_hull

    intersection = polygon2.intersection(polygon1)
    overlap_ratio = intersection.area / min(polygon1.area,polygon2.area)
    return overlap_ratio
#最大重叠度
def convex_hull_overlap_max(polygon1, polygon2):
    polygon1=polygon1.convex_hull
    polygon2 = polygon2.convex_hull

    intersection = polygon2.intersection(polygon1)
    overlap_ratio = intersection.area / max(polygon1.area,polygon2.area)
    return overlap_ratio
'凸包融合'
def hull_contact(groups_dic):
    groups_behalf = []
    for key, value in groups_dic.items():
        values = [dictionary for i,dictionary in value]
        merged_polygon = MultiPolygon(values).buffer(0)  # buffer(0) 用于修复可能出现的无效几何对象

        # Step 2: 计算大的 Polygon 的凸包
        convex_hulls = merged_polygon.convex_hull

        groups_behalf.append(convex_hulls)
    return groups_behalf

'转换为df'
def to_df(group_dict,data_Q):
    all_df = pd.DataFrame()
    c = 1
    for key, value in group_dict.items():

        for hull in value:
            hull_df = hull_to_df(hull)
            hull_df['cluster'] = c
            merged_df = pd.merge(data_Q, hull_df, on=['LON', 'LAT'], how='inner')
            all_df=pd.concat([all_df,merged_df])

        c += 1
    all_df=all_df.drop_duplicates(subset=['LON', 'LAT'])
    all_df.reset_index(inplace=True, drop=True)
    return all_df

'凸包转df'
def hull_to_df(polygon):
    hull_points = list(polygon.exterior.coords)
    hull_df = pd.DataFrame(hull_points, columns=['LON', 'LAT'])
    return hull_df

'对象存储'
def dic_to_df(dic,data_all):
    df_all=pd.DataFrame()
    for key,values in dic.items():
        for T,value in values:
            df=data_all[data_all['T']==T]
            df=df.copy()
            df.loc[:, 'D'] = key
            df_all=pd.concat([df_all,df])
    return df_all
'最优匹配位置'
def find_matching_positions(input_list, segment_length):
    matching_positions = []

    # 将列表分为相同长度的几段
    segmented_lists = [input_list[i:i + segment_length] for i in range(0, len(input_list), segment_length)]
    for position,segment in enumerate(segmented_lists):
        if all(value == 0.0 for value in segment):
            return position
    # # 检查每个位置上的值是否相同
    # for position in range(segment_length):
    #     values_at_position = [segment[position] for segment in segmented_lists]
    #
    #     # 如果所有段相同位置的数值都相同，则获取到这个位置序号
    #     if all(value == 0.0 for value in values_at_position):
    #         return position
'''
p:建组率，保障组的中心性
q:扩张率，确定组初步扩张的程度
m:匹配率，有度匹配程度参数，保证锚位的完整性
l：剔除度，剔除小数量锚泊点，保证数据质量
'''
def to_place(data_all,outfile,p=0.8,q=0.4,l=30,m=0.05):
    hull_list = {}
    #构建凸包对象
    datas = data_all.groupby('T')
    for da_key, data in datas:
        if len(data)<l:
            continue
        cluster_data = data[['LON', 'LAT']]
        points = cluster_data.values
        hull = Polygon(points)
        hull_list[da_key] = hull
    '建组'
    group_dict = {}
    match = []
    n = 0
    hu = []
    for key, hull1 in list(hull_list.items()):
        q=q
        if key in hu:
            continue
        try:
            overlap_ratio_list = []
            for j, (key1, hull2) in enumerate(hull_list.items()):
                overlap_ratio = convex_hull_overlap(hull1, hull2)
                overlap_ratio_list.append(overlap_ratio)
            # 如果列表中除了 1 以外的元素都为 0，获得 1 的索引
            if all(x <p for i, x in enumerate(overlap_ratio_list) if i != 0):
                match.extend([(key,hull1)])
                hu.append(key)
                hull_list.pop(key)
            # 如果列表中除了1,包含大于等于p的元素
            elif len([i for i, x in enumerate(overlap_ratio_list) if i != 0 if x >= p]) >= 1:
                indices_above_0_8 = [i for i, x in enumerate(overlap_ratio_list) if x >= p]
                if len(indices_above_0_8)<=3:
                    match.extend([(key, hull1)])
                    hu.append(key)
                    hull_list.pop(key)
                else:
                    group_dict[n] = [list(hull_list.items())[index] for index in indices_above_0_8]

                    for index in sorted(indices_above_0_8, reverse=True):
                        key2 = list(hull_list.keys())[index]
                        hu.append(key2)
                        hull_list.pop(key2)
                    n += 1
                    group_dict1 = {}
                    #贪性算法

                    overlap = True
                    while overlap:
                        group_dict1[1] = group_dict[n - 1]
                        groups_behalf = hull_contact(group_dict1)
                        behalf = groups_behalf[0]
                        overlap_ratio_list = []
                        for hull in list(hull_list.values()):
                            overlap_ratio = convex_hull_overlap(behalf, hull)
                            overlap_ratio_list.append(overlap_ratio)
                        try:
                            print(min([x for x in overlap_ratio_list if x != 0.0]))
                        except ValueError:
                            overlap = False
                            continue
                        if len([x for x in overlap_ratio_list if  x > q]) >= 1:
                            indices_above_0_8 = [i for i, x in enumerate(overlap_ratio_list) if x >= q]
                            group_dict[n - 1].extend([list(hull_list.items())[index] for index in indices_above_0_8])
                            for index in sorted(indices_above_0_8, reverse=True):
                                key = list(hull_list.keys())[index]
                                hu.append(key)
                                hull_list.pop(key)
                            overlap = True
                        elif len([x for x in overlap_ratio_list if x != 0.0 and x < q])>=1:
                            indices_above_0_8 = [i for i, x in enumerate(overlap_ratio_list) if x != 0.0 and x < q]
                            for index in sorted(indices_above_0_8, reverse=True):
                                key2 = list(hull_list.keys())[index]
                                match.extend([(key2, list(hull_list.values())[index])])
                                hu.append(key2)
                                hull_list.pop(key2)
                            overlap = True
                        else:
                            overlap = False
        except IndexError:
            break
    '精细提取'
    status=True
    while status:
        if len(match)==0:
            break
        else:
            '计算重叠度'
            groups_behalf = hull_contact(group_dict)
            overlap_ratio_list = []
            for key, hull1 in match:
                for hull2 in groups_behalf:
                    overlap_ratio = convex_hull_overlap(hull2, hull1)
                    overlap_ratio_list.append(overlap_ratio)

            k=len(groups_behalf)
            removes = []
            '有组情况'
            if k!=0:
                '匹配'
                z = 0
                for i in range(0, len(overlap_ratio_list), k):
                    group = overlap_ratio_list[i:i + k]
                    max_value = max(group)
                    if max_value>m:
                        max_index = group.index(max_value)
                        group_dict[max_index].extend([match[z]])
                        removes.append(z)
                    z += 1
                if len(removes)>0:
                    for index in sorted(removes, reverse=True):
                        match.pop(index)
                    continue
                '建组'
                index_f=find_matching_positions(overlap_ratio_list, k)
                if index_f!=None:
                    print(f'index_f:{index_f}')
                    group_dict[n]=[match[index_f]]
                    removes.append(index_f)
                    n+=1
                else:
                    z = 0
                    for i in range(0, len(overlap_ratio_list), k):
                        group = overlap_ratio_list[i:i + k]
                        max_value = max(group)
                        if max_value > 0:
                            max_index = group.index(max_value)
                            group_dict[max_index].extend([match[z]])
                            removes.append(z)
                        z += 1
            else:
                '无组情况'
                group_dict[n] = [match[0]]
                removes.append(0)
                n += 1

            '数据剔除'
            for index in sorted(removes, reverse=True):
                match.pop(index)



    df_all=dic_to_df(group_dict,data_all)
    if os.path.exists(outfile):
        df_all.to_csv(outfile,mode='a',index=False,header=False)
    else:
        df_all.to_csv(outfile, mode='a', index=False, header=True)



def main(input_folder,outfolder):


    filelist = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
    for filename in filelist:
        # 读取CSV文件，包含点的坐标信息
        file_path =os.path.join(input_folder,filename)
        data = pd.read_csv(file_path)
        data = to_region(data)

        print('完成分区')
        outfile=os.path.join(outfolder,filename)
        if os.path.exists(outfile):
            os.remove(outfile)
        data_groups=data.groupby('Q')
        for key,da_group in data_groups:
            to_place(da_group,outfile)
            print(f'完成{key}')





if __name__=='__main__':
    input_folder=r'data'
    outfolder=r'result'
    os.makedirs(outfolder,exist_ok=True)


    main(input_folder,outfolder)