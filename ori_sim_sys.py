from utils import *
# import dxfgrabber

# 定义折纸系统，包含各刚度和单元信息
class OrigamiSimulationSystem:
    def __init__(self, unit_edge_max, spring_k=5000., bending_k=0.2, face_k=100., material_density=1.24e-9, controller_mass=5e-4, split_unit_list=[0]) -> None:
        self.unit_edge_max = unit_edge_max
        self.unit_list = []
        self.kps = []
        self.line_indices = []
        self.indices = []
        self.tri_indices = []
        self.tri_indices_ref = []
        self.indices_crease_type = []
        self.connection_matrix = None
        self.spring_k = spring_k
        self.bending_k = bending_k
        self.face_k = face_k
        self.material_density = material_density
        self.bending_pairs = []
        self.crease_pairs = []
        self.facet_bending_pairs = []
        self.facet_crease_pairs = []
        self.mass_list = []
        self.consistent_mass_list = []
        self.dup_time_list = []

        self.controller_mass = controller_mass
        self.special_num = 0

        self.split_origami_num = 1
        self.split_unit_list = split_unit_list
        
        self.problem_id_list = []

        self.new_kp_origami_id = []

    def getNewLines(self):
        new_lines = []
        for ele in self.line_indices:
            if ele[1] != 3:
                new_lines.append(Crease(self.kps[ele[0][START]], self.kps[ele[0][END]], ele[1], upper=ele[2], lower=ele[3]))
        return new_lines

    def getNewLineIndices(self):
        new_line_indices = []
        for ele in self.line_indices:
            if len(ele) == 4:
                if ele[1] != 3:
                    new_line_indices.append(ele)
            elif len(ele) == 5:
                if ele[2] != 3:
                    new_line_indices.append(ele)
        return new_line_indices
    
    def addToLineIndices(self, pair, linetype, folding_angle_upper_bound, folding_angle_lower_bound):
        duplicated = False
        for ele in self.line_indices:
            if ele[0][0] == pair[0] and ele[0][1] == pair[1]:
                duplicated = True
                break
            if ele[0][1] == pair[0] and ele[0][0] == pair[1]:
                duplicated = True
                break
        if not duplicated:
            self.line_indices.append([pair, linetype, folding_angle_upper_bound, folding_angle_lower_bound])

    def distance(self, kp1, kp2):
        return ((kp1[0] - kp2[0]) ** 2 + (kp1[1] - kp2[1]) ** 2 + (kp1[2] - kp2[2]) ** 2) ** 0.5
        
    def pointInList(self, kp, tolerance=2): # strict or loose
        candidate_points_id = []
        for i in range(len(self.kps)):
            if self.distance(kp, self.kps[i]) < tolerance:
                candidate_points_id.append(i)
        if len(candidate_points_id) == 1:
            return candidate_points_id[0]
        elif len(candidate_points_id) > 1:
            return candidate_points_id[0]   
        return -1
    
    def optimizeIndices(self):
        self.unit_edge_max = max([len(self.indices[i]) for i in range(len(self.indices))])
        preserved_id = [-1, -1]
        preserved_id_accompany = [-1, -1]
        preserved_crease_type = [-1, -1]
        
        created_id = [-1, -1]
        created_id_accompany = [-1, -1]
        created_crease_type = [-1, -1]
        
        created_kps = []
        created_mass_ref = []
        
        replacement_buffer = [] #[id, position, other_id]
        
        for i in range(len(self.indices)):
            current_indice = self.indices[i]
            current_crease_type_list = self.indices_crease_type[i] 
            kp_num = len(current_indice)
            for j in range(kp_num):
                current_kp_id = current_indice[j]
                next_kp_id = current_indice[(j + 1) % kp_num]
                current_crease_type = current_crease_type_list[j]
                if current_crease_type == BORDER:
                    indice_position = -1
                    for i2 in range(i + 1, len(self.indices)):
                        other_indice = self.indices[i2]
                        kp_num2 = len(other_indice)
                        other_crease_type_list = self.indices_crease_type[i2] 
                        if current_kp_id in other_indice and next_kp_id in other_indice:
                            indice_position = other_indice.index(next_kp_id)
                            break
                    if indice_position >= 0:
                        # sharing border happens, removing the duplicated points
                        preserved_id[0] = current_kp_id
                        preserved_id[1] = next_kp_id
                        preserved_id_accompany[0] = current_indice[(j - 1 + kp_num) % kp_num]
                        preserved_id_accompany[1] = current_indice[(j + 2 + kp_num) % kp_num]
                        preserved_crease_type[0] = current_crease_type_list[(j - 1 + kp_num) % kp_num]
                        preserved_crease_type[1] = current_crease_type_list[(j + 1 + kp_num) % kp_num]
                        
                        # created_id[0] = all_kp_num + len(created_kps)
                        # created_id[1] = all_kp_num + len(created_kps) + 1
                        
                        # created_kps.append(deepcopy(self.kps[preserved_id[0]]))
                        # created_kps.append(deepcopy(self.kps[preserved_id[1]]))
                        
                        created_id_accompany[0] = other_indice[(indice_position + 2 + kp_num2) % kp_num2]
                        created_id_accompany[1] = other_indice[(indice_position - 1 + kp_num2) % kp_num2]
                        created_crease_type[0] = other_crease_type_list[(indice_position + 1 + kp_num2) % kp_num2]
                        created_crease_type[1] = other_crease_type_list[(indice_position - 1 + kp_num2) % kp_num2]
                        
                        backup_end_flag = [0, 0, 0, 0]
                        end_flag = [0, 0, 0, 0]
                        
                        preserved_ids_1 = [i] #previous
                        preserved_ids_2 = [i] #next
                        replace_ids_1 = [i2] #previous
                        replace_ids_2 = [i2] #next
                        
                        max_iterations = 1000  # 防止死循环的最大迭代次数
                        iteration_count = 0
                        steps = [0, 0, 0, 0]  # 初始化steps变量
                        
                        while (0 in end_flag):
                            # con-stop condition
                            if preserved_id_accompany[0] == created_id_accompany[0] and created_crease_type[0] != BORDER and preserved_crease_type[0] != BORDER:
                                replace_ids_1.clear()
                                end_flag[0] = end_flag[2] = 1
                            if preserved_id_accompany[1] == created_id_accompany[1] and created_crease_type[1] != BORDER and preserved_crease_type[1] != BORDER:
                                replace_ids_2.clear()
                                end_flag[1] = end_flag[3] = 1
                                
                            exist_full_in_1 = False
                            for i3 in range(len(self.indices)):
                                if (preserved_id_accompany[0] in self.indices[i3]) and \
                                    (created_id_accompany[0] in self.indices[i3]) and \
                                    (preserved_id[0] in self.indices[i3]):
                                        position = self.indices[i3].index(preserved_id[0])
                                        if self.indices_crease_type[i3][position] != BORDER and self.indices_crease_type[i3][(position - 1 + len(self.indices[i3])) % len(self.indices[i3])] != BORDER:
                                            exist_full_in_1 = True
                                        break
                            if exist_full_in_1:
                                replace_ids_1.clear()
                                end_flag[0] = end_flag[2] = 1
                            
                            exist_full_in_2 = False
                            for i3 in range(len(self.indices)):
                                if (preserved_id_accompany[1] in self.indices[i3]) and \
                                    (created_id_accompany[1] in self.indices[i3]) and \
                                    (preserved_id[1] in self.indices[i3]):
                                        position = self.indices[i3].index(preserved_id[1])
                                        if self.indices_crease_type[i3][position] != BORDER and self.indices_crease_type[i3][(position - 1 + len(self.indices[i3])) % len(self.indices[i3])] != BORDER:
                                            exist_full_in_2 = True
                                        break
                            if exist_full_in_2:
                                replace_ids_2.clear()
                                end_flag[1] = end_flag[3] = 1
                                
                            #single stop
                            if preserved_crease_type[0] == BORDER:
                                end_flag[0] = 1
                            if preserved_crease_type[1] == BORDER:
                                end_flag[1] = 1
                            if created_crease_type[0] == BORDER:
                                end_flag[2] = 1
                            if created_crease_type[1] == BORDER:
                                end_flag[3] = 1
                                
                            #roll
                            if end_flag == backup_end_flag:
                                steps = [0, 0, 0, 0]
                                for i3 in range(len(self.indices)):
                                    if not end_flag[0] and not steps[0] and (i3 not in preserved_ids_1 and preserved_id[0] in self.indices[i3] and preserved_id_accompany[0] in self.indices[i3]):
                                        kp_num3 = len(self.indices[i3])
                                        preserved_ids_1.append(i3)
                                        position = self.indices[i3].index(preserved_id[0])
                                        previous_id = self.indices[i3][((position - 1) + kp_num3) % kp_num3]
                                        next_id = self.indices[i3][((position + 1) + kp_num3) % kp_num3]
                                        if previous_id == preserved_id_accompany[0]:
                                            preserved_id_accompany[0] = next_id
                                            preserved_crease_type[0] = self.indices_crease_type[i3][((position) + kp_num3) % kp_num3]
                                        else:
                                            preserved_id_accompany[0] = previous_id
                                            preserved_crease_type[0] = self.indices_crease_type[i3][((position - 1) + kp_num3) % kp_num3]
                                        if preserved_crease_type[0] == BORDER:
                                            end_flag[0] = 1
                                        steps[0] = 1
                                    
                                    if not end_flag[1] and not steps[1] and (i3 not in preserved_ids_2 and preserved_id[1] in self.indices[i3] and preserved_id_accompany[1] in self.indices[i3]):
                                        kp_num3 = len(self.indices[i3])
                                        preserved_ids_2.append(i3)
                                        position = self.indices[i3].index(preserved_id[1])
                                        previous_id = self.indices[i3][((position - 1) + kp_num3) % kp_num3]
                                        next_id = self.indices[i3][((position + 1) + kp_num3) % kp_num3]
                                        if previous_id == preserved_id_accompany[1]:
                                            preserved_id_accompany[1] = next_id
                                            preserved_crease_type[1] = self.indices_crease_type[i3][((position) + kp_num3) % kp_num3]
                                        else:
                                            preserved_id_accompany[1] = previous_id
                                            preserved_crease_type[1] = self.indices_crease_type[i3][((position - 1) + kp_num3) % kp_num3]
                                        if preserved_crease_type[1] == BORDER:
                                            end_flag[1] = 1
                                        steps[1] = 1
                                                
                                    if not end_flag[2] and not steps[2] and (i3 not in replace_ids_1 and preserved_id[0] in self.indices[i3] and created_id_accompany[0] in self.indices[i3]):
                                        kp_num3 = len(self.indices[i3])
                                        replace_ids_1.append(i3)
                                        position = self.indices[i3].index(preserved_id[0])
                                        previous_id = self.indices[i3][((position - 1) + kp_num3) % kp_num3]
                                        next_id = self.indices[i3][((position + 1) + kp_num3) % kp_num3]
                                        if previous_id == created_id_accompany[0]:
                                            created_id_accompany[0] = next_id
                                            created_crease_type[0] = self.indices_crease_type[i3][((position) + kp_num3) % kp_num3]
                                        else:
                                            created_id_accompany[0] = previous_id
                                            created_crease_type[0] = self.indices_crease_type[i3][((position - 1) + kp_num3) % kp_num3]
                                        if created_crease_type[0] == BORDER:
                                            end_flag[2] = 1
                                        steps[2] = 1
                                        
                                    if not end_flag[3] and not steps[3] and (i3 not in replace_ids_2 and preserved_id[1] in self.indices[i3] and created_id_accompany[1] in self.indices[i3]):
                                        kp_num3 = len(self.indices[i3])
                                        replace_ids_2.append(i3)
                                        position = self.indices[i3].index(preserved_id[1])
                                        previous_id = self.indices[i3][((position - 1) + kp_num3) % kp_num3]
                                        next_id = self.indices[i3][((position + 1) + kp_num3) % kp_num3]
                                        if previous_id == created_id_accompany[1]:
                                            created_id_accompany[1]= next_id
                                            created_crease_type[1] = self.indices_crease_type[i3][((position) + kp_num3) % kp_num3]
                                        else:
                                            created_id_accompany[1] = previous_id
                                            created_crease_type[1] = self.indices_crease_type[i3][((position - 1) + kp_num3) % kp_num3]
                                        if created_crease_type[1] == BORDER:
                                            end_flag[3] = 1
                                        steps[3] = 1
                                            
                            backup_end_flag = deepcopy(end_flag)
                            
                            # 防止死循环：检查是否没有进展
                            iteration_count += 1
                            if iteration_count >= max_iterations:
                                print(f"[Warning] optimizeIndices: 达到最大迭代次数 {max_iterations}，强制退出循环")
                                print(f"  当前单元: {i}, 配对单元: {i2}")
                                print(f"  end_flag: {end_flag}, steps: {steps}")
                                break
                            
                            # # 如果steps全为0且end_flag没有变化，说明无法继续，强制退出
                            # if sum(steps) == 0 and end_flag == backup_end_flag:
                            #     print(f"[Warning] optimizeIndices: 检测到无法继续推进，强制退出循环")
                            #     print(f"  当前单元: {i}, 配对单元: {i2}")
                            #     print(f"  end_flag: {end_flag}")
                            #     break
                            
                        check_duplication = 0
                        while check_duplication < len(replace_ids_1):
                            position = self.indices[replace_ids_1[check_duplication]].index(preserved_id[0])
                            for ele in replacement_buffer:
                                if replace_ids_1[check_duplication] == ele[0] and position == ele[1]:
                                    check_duplication -= 1
                                    del(replace_ids_1[check_duplication])
                                    break
                            check_duplication += 1
                        
                        check_duplication = 0
                        while check_duplication < len(replace_ids_2):
                            position = self.indices[replace_ids_2[check_duplication]].index(preserved_id[1])
                            for ele in replacement_buffer:
                                if replace_ids_2[check_duplication] == ele[0] and position == ele[1]:
                                    check_duplication -= 1
                                    del(replace_ids_2[check_duplication])
                                    break
                            check_duplication += 1
 
                        if len(replace_ids_1) and (np.array(replace_ids_1) > i).all():
                            created_kps.append(deepcopy(self.kps[preserved_id[0]]))
                            created_mass_ref.append(preserved_id[0])
                            for id in replace_ids_1:
                                position = self.indices[id].index(preserved_id[0])
                                replacement_buffer.append([id, position, len(self.kps) + len(created_kps) - 1])
                                # self.indices[id][position] = len(self.kps) + len(created_kps) - 1

                        if len(replace_ids_2) and (np.array(replace_ids_2) > i).all():
                            created_kps.append(deepcopy(self.kps[preserved_id[1]]))
                            created_mass_ref.append(preserved_id[1])
                            for id in replace_ids_2:
                                position = self.indices[id].index(preserved_id[1])
                                replacement_buffer.append([id, position, len(self.kps) + len(created_kps) - 1])
                                # self.indices[id][position] = len(self.kps) + len(created_kps) - 1
        
        for ele in replacement_buffer:
            self.indices[ele[0]][ele[1]] = ele[2]
        
        self.new_kp_origami_id = []
        for ele in replacement_buffer:
            unit_id = ele[0]
            kp_id = ele[2]
            for k in range(len(self.split_unit_list)):
                if self.split_unit_list[k] <= unit_id and (k == len(self.split_unit_list) - 1 or self.split_unit_list[k + 1] > unit_id):
                    break
            if [kp_id, k] not in self.new_kp_origami_id:
                self.new_kp_origami_id.append([kp_id, k])
            
        self.line_indices.clear()
        for i1 in range(len(self.unit_list)):
            unit = self.unit_list[i1]
            for i in range(len(self.indices[i1])):
                next_i = (i + 1) % len(self.indices[i1])
                linetype = unit.crease[i].getType()
                if unit.crease[i].hard:
                    linetype = 3
                indice1 = self.indices[i1][i]
                indice2 = self.indices[i1][next_i]
                self.addToLineIndices([indice1, indice2], linetype, unit.crease[i].folding_angle_upper_bound, unit.crease[i].folding_angle_lower_bound)

        created_mass = [0.0 for _ in range(len(created_mass_ref))]
        original_kp_len = len(self.kps)
        self.kps += created_kps  
        for i in range(original_kp_len):
            basic_mass = self.mass_list[i]
            portion = 1
            idx = []
            for j in range(len(created_mass_ref)):
                ele = created_mass_ref[j]
                if ele == i:
                    portion += 1
                    idx.append(j)
            if portion > 1:
                final_mass = basic_mass / portion
                self.mass_list[i] = final_mass
                for ele in idx:
                    created_mass[ele] = final_mass
        self.mass_list += created_mass

    def addUnit(self, unit: Unit, special=False, scale=1.0, tol=1.0):    
        temp_indice = []
        temp_indice_crease_type = []
        unit.repair(tol)
        kps = unit.getSeqPoint()
        kp_num = len(kps)
        if kp_num < 3:
            return
        # invalid unit
        if kp_num > self.unit_edge_max:
            self.unit_edge_max = kp_num
        # area = unit.calculateArea()
        # mass = area * self.material_density / kp_num
        if special:
            unit.setupMassDirectly(unit.special_mass)
            self.special_num += 1
        else:
            unit.setupMass(self.material_density * scale)

        for i in range(kp_num):
            kp = kps[i]
            if len(kp) == 2:
                kp += [0.0]
            # connected_crease_type = [unit.crease[(i - 1 + kp_num) % kp_num].getType(), unit.crease[i].getType()]
            # if connected_crease_type[START] != 2 or connected_crease_type[END] != 2:   
            #     previous_kp = kps[(i - 1 + kp_num) % kp_num]
            #     next_kp = kps[(i + 1) % kp_num]
            exist_indice = self.pointInList(kp, tolerance=tol)
                # if connected_crease_type[START] == 2 and self.pointInList(previous_kp) >= 0 and self.pointInList(next_kp) < 0 and exist_indice >= 0:
                #     exist_indice = -1
                # if connected_crease_type[END] == 2 and self.pointInList(next_kp) >= 0 and self.pointInList(previous_kp) < 0 and exist_indice >= 0:
                #     exist_indice = -1
            # else:
            #     exist_indice = -1
            if exist_indice < 0:
                temp_indice.append(len(self.kps))
                self.kps.append(kp)
                self.mass_list.append(unit.mass[i])
                self.dup_time_list.append(1.)
            else:
                self.kps[exist_indice] = [
                    (self.kps[exist_indice][X] * self.dup_time_list[exist_indice] + kp[X]) / (self.dup_time_list[exist_indice] + 1.),
                    (self.kps[exist_indice][Y] * self.dup_time_list[exist_indice] + kp[Y]) / (self.dup_time_list[exist_indice] + 1.),
                    (self.kps[exist_indice][Z] * self.dup_time_list[exist_indice] + kp[Z]) / (self.dup_time_list[exist_indice] + 1.)
                ]
                self.dup_time_list[exist_indice] += 1.
                temp_indice.append(exist_indice)
                self.mass_list[exist_indice] += unit.mass[i]
            temp_indice_crease_type.append(unit.crease[i].getType())
    
        for i in range(len(kps)):
            next_i = (i + 1) % len(kps)
            linetype = unit.crease[i].getType()
            if unit.crease[i].hard:
                linetype = 3
            indice1 = temp_indice[i]
            indice2 = temp_indice[next_i]
            self.addToLineIndices([indice1, indice2], linetype, unit.crease[i].folding_angle_upper_bound,unit.crease[i].folding_angle_lower_bound)

        self.unit_list.append(unit)
        self.indices.append(temp_indice)
        self.consistent_mass_list.append(unit.consistent_mass)
        self.indices_crease_type.append(temp_indice_crease_type)

    def fillBlankIndices(self):
        for i, indice in enumerate(self.indices):
            # fill the blank
            indice_num = len(indice)
            for _ in range(indice_num, self.unit_edge_max):
                indice.append(-1)
            self.consistent_mass_list[i] = np.pad(self.consistent_mass_list[i], pad_width=((0, self.unit_edge_max - indice_num), (0, self.unit_edge_max - indice_num)), mode="constant", constant_values=0.0)

    def calculateElementK(self, tri_indices):
        k1 = self.spring_k
        # for i in range(0, n):
        #     k_element[i][(i + 1) % n] = k1
        #     k_element[(i + 1) % n][i] = k1
        for index in tri_indices:
            self.connection_matrix[index[0]][index[1]] = k1
            self.connection_matrix[index[1]][index[0]] = k1
            self.connection_matrix[index[0]][index[2]] = k1
            self.connection_matrix[index[2]][index[0]] = k1
            self.connection_matrix[index[1]][index[2]] = k1
            self.connection_matrix[index[2]][index[1]] = k1
        # return k_element

    def calculateMaximumDeltaAngle(self, x0, x1, x2):
        x0x1 = self.distance(x0, x1)
        x1x2 = self.distance(x1, x2)
        x2x0 = self.distance(x2, x0)
        alpha0 = math.acos(max(min((x0x1**2+x2x0**2-x1x2**2)/(2*x0x1*x2x0), 1.), -1.))
        alpha1 = math.acos(max(min((x1x2**2+x0x1**2-x2x0**2)/(2*x1x2*x0x1), 1.), -1.))
        alpha2 = math.acos(max(min((x2x0**2+x1x2**2-x0x1**2)/(2*x2x0*x1x2), 1.), -1.))
        return max([abs(alpha0 - alpha1), abs(alpha0 - alpha2), abs(alpha1 - alpha2)])
    
    def mesh(self):
        self.optimizeIndices()
        self.tri_indices.clear()
        self.tri_indices_ref.clear()
        kp_len = len(self.kps)
        self.connection_matrix = [[0] * kp_len for _ in range(kp_len)]
        self.origin_distance_matrix = [[0] * kp_len for _ in range(kp_len)]

        #origin distance
        for i in range(kp_len):
            for j in range(kp_len):
                self.origin_distance_matrix[i][j] = self.distance(self.kps[i], self.kps[j])
        
        # for i in range(kp_len):
        #     self.mass_list[i] = 5e-7

        self.facet_cons_id = []
        facet_pair = []
        current_tri_indice_ref = 0
        #spring force
        for i in range(len(self.unit_list)):
            indices = self.indices[i]
            unit = self.unit_list[i].getSeqPoint()
            unit_kp_len = len(indices)
            
            complete_id = []
            tri_indices = []
            forbidden_indices = []

            while unit_kp_len - len(complete_id) >= 3:
                delta_angle_max = 3.14
                temp_tri_indices = None
                pointer = 0
                
                while pointer < unit_kp_len - len(complete_id):
                    while indices[pointer] in complete_id:
                        pointer += 1
                    next_pointer = (pointer + 1) % unit_kp_len
                    while indices[next_pointer] in complete_id:
                        next_pointer = (next_pointer + 1) % unit_kp_len
                    next_next_pointer = (next_pointer + 1) % unit_kp_len
                    while indices[next_next_pointer] in complete_id:
                        next_next_pointer = (next_next_pointer + 1) % unit_kp_len
                    
                    current_delta_angle = self.calculateMaximumDeltaAngle(unit[pointer], unit[next_pointer], unit[next_next_pointer])
                    if current_delta_angle < delta_angle_max and [indices[pointer], indices[next_pointer], indices[next_next_pointer]] not in forbidden_indices:
                        temp_tri_indices = [indices[pointer], indices[next_pointer], indices[next_next_pointer]]
                        delta_angle_max = current_delta_angle
                    pointer += 1

                if temp_tri_indices == None:
                    complete_id.pop()
                    forbidden_indices.append(tri_indices[-1])
                    tri_indices.pop()
                    facet_pair.pop()
                    self.facet_cons_id.pop()
                    continue

                if unit_kp_len - len(complete_id) > 3:
                    facet_pair.append([temp_tri_indices[0], temp_tri_indices[2]])
                    if self.split_unit_list != None:
                        for k in range(len(self.split_unit_list)):
                            if self.split_unit_list[k] <= i and (k == len(self.split_unit_list) - 1 or self.split_unit_list[k + 1] > i):
                                break
                        self.facet_cons_id.append(k)
                tri_indices.append(temp_tri_indices)
                complete_id.append(temp_tri_indices[1])
                
                # self.addToLineIndices([temp_tri_indices[0], temp_tri_indices[2]], 3, folding_angle_upper_bound=math.pi, folding_angle_lower_bound=-math.pi)
            # temp_tri_indices = []
            # for pointer in range(unit_kp_len):
            #     if indices[pointer] not in complete_id:
            #         temp_tri_indices.append(indices[pointer])
            # tri_indices.append(temp_tri_indices)
            
            self.calculateElementK(tri_indices)

            for j in range(len(tri_indices)):
                tri_index = tri_indices[j]
                self.tri_indices += [tri_index[0], tri_index[1], tri_index[2]]
            
            self.tri_indices_ref.append(current_tri_indice_ref)
            current_tri_indice_ref += len(tri_indices)
            # for k in range(unit_kp_len):
            #     indice_k = indices[k]
            #     for l in range(unit_kp_len):
            #         indice_l = indices[l]
            #         self.connection_matrix[indice_k][indice_l] = k_element[k][l]
        
        #bending force
        for i in range(len(self.line_indices)):
            # indices = self.indices[i]
            # unit_kp_len = len(indices)
            # for j in range(unit_kp_len):
            # line_start_indice = indices[j]
            # line_end_indice = indices[(j + 1) % unit_kp_len]
            line_start_indice = self.line_indices[i][0][0]
            line_end_indice = self.line_indices[i][0][1]
            line_type = self.line_indices[i][1]
            start_row = self.connection_matrix[line_start_indice]
            end_row = self.connection_matrix[line_end_indice]

            relevant_kp = []
            for k in range(kp_len):
                if abs(start_row[k] - end_row[k]) < 1e-5 and start_row[k] > 0:
                    relevant_kp.append(k)
            
            if len(relevant_kp) == 2 and [line_end_indice, line_start_indice] not in self.crease_pairs:
                crease_pair = [line_start_indice, line_end_indice]
                result = 0
                for j in range(len(self.tri_indices) // 3):
                    tri_index = [self.tri_indices[3 * j], self.tri_indices[3 * j + 1], self.tri_indices[3 * j + 2]]
                    if line_start_indice in tri_index and line_end_indice in tri_index and relevant_kp[0] in tri_index:
                        index_start = tri_index.index(line_start_indice)
                        index_end = tri_index.index(line_end_indice)
                        if index_end == index_start + 1 or (index_end == 0 and index_start == 2):
                            result = 1
                        else:
                            result = -1
                        break
                
                if result >= 0:
                    if line_type == VALLEY or line_type == MOUNTAIN:
                        self.bending_pairs.append([relevant_kp[0], relevant_kp[1]])
                    elif line_type == 3:
                        self.facet_bending_pairs.append([relevant_kp[0], relevant_kp[1]])
                else:
                    if line_type == VALLEY or line_type == MOUNTAIN:
                        self.bending_pairs.append([relevant_kp[1], relevant_kp[0]])
                    elif line_type == 3:
                        self.facet_bending_pairs.append([relevant_kp[1], relevant_kp[0]])
                if line_type == VALLEY or line_type == MOUNTAIN:
                    self.crease_pairs.append(crease_pair)
                elif line_type == 3:
                    self.facet_crease_pairs.append(crease_pair)
                    if self.split_unit_list != None:
                        for k in range(len(self.split_unit_list)):
                            if self.split_unit_list[k] <= i and (k == len(self.split_unit_list) - 1 or self.split_unit_list[k + 1] > i):
                                break
                        self.facet_cons_id.append(k)

        # facet
        for i in range(len(facet_pair)):
            line_start_indice = facet_pair[i][0]
            line_end_indice = facet_pair[i][1]
            start_row = self.connection_matrix[line_start_indice]
            end_row = self.connection_matrix[line_end_indice]

            relevant_kp = []
            for k in range(kp_len):
                if abs(start_row[k] - end_row[k]) < 1e-5 and start_row[k] > 0:
                    relevant_kp.append(k)
            
            if len(relevant_kp) == 2 and [line_end_indice, line_start_indice] not in self.facet_crease_pairs:
                facet_crease_pair = [line_start_indice, line_end_indice]
                vec1xy = [self.kps[line_start_indice][0] - self.kps[relevant_kp[0]][0], self.kps[line_start_indice][1] - self.kps[relevant_kp[0]][1]]
                vec2xy = [self.kps[relevant_kp[1]][0] - self.kps[line_start_indice][0], self.kps[relevant_kp[1]][1] - self.kps[line_start_indice][1]]
                result = vec1xy[0] * vec2xy[1] - vec1xy[1] * vec2xy[0]
                if result >= 0:
                    self.facet_bending_pairs.append([relevant_kp[0], relevant_kp[1]])
                else:
                    self.facet_bending_pairs.append([relevant_kp[1], relevant_kp[0]])
                self.facet_crease_pairs.append(facet_crease_pair)
                
        
        while len(self.mass_list) < len(self.kps):
            self.mass_list.insert(0, 0.0)

        total_mass = sum(self.mass_list[0: len(self.mass_list) - 4 * self.special_num])
        avg_mass = total_mass / (len(self.mass_list) - 4 * self.special_num)
        # for i in range(len(self.mass_list) - 4 * self.special_num):
        #     self.mass_list[i] = avg_mass