from utils import *
from spatialhash import SpatialHash
import time
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

        self.spatialhash = SpatialHash(cell_size=1.0)

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
        
    def pointInList(self, kp, tolerance=1): # strict or loose
        # candidate_points_id = []
        # for i in range(len(self.kps)):
        #     if self.distance(kp, self.kps[i]) < tolerance:
        #         candidate_points_id.append(i)
        # if len(candidate_points_id) == 1:
        #     return candidate_points_id[0]
        # elif len(candidate_points_id) > 1:
        #     return candidate_points_id[0]   
        # return -1
        return self.spatialhash.find(kp, tolerance)
    
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

    def optimizeIndices_v2(self):
        """Optimized version of optimizeIndices using inverted index.

        Complexity: O(F*E + V*D^2) where F=panels, E=edges/panel, V=vertices, D=avg degree
        vs original O(F^3 * E).

        Key optimizations:
        1. vertex->panels inverted index replaces O(F^2) scan for shared border discovery
        2. Chain tracing uses inverted index instead of scanning all panels each iteration
        3. replacement_buffer uses set for O(1) deduplication
        4. line_indices uses set for O(1) edge deduplication
        5. Mass redistribution uses Counter for O(V+D) instead of O(V*D)
        """
        from collections import Counter, defaultdict

        self.unit_edge_max = max(len(self.indices[i]) for i in range(len(self.indices)))
        F = len(self.indices)

        # ---- Phase 1: Build vertex -> panels inverted index ----
        # vertex_to_panels[v] = [(panel_id, position), ...] sorted by panel_id
        vertex_to_panels = defaultdict(list)
        for pid in range(F):
            n = len(self.indices[pid])
            for pos in range(n):
                v = self.indices[pid][pos]
                vertex_to_panels[v].append((pid, pos))

        def find_first_panel_with_both(v, w, min_pid=-1):
            """Find first panel with id > min_pid containing both v and w.
            Uses two-pointer merge on sorted lists. O(deg(v) + deg(w))."""
            vp = vertex_to_panels.get(v, [])
            wp = vertex_to_panels.get(w, [])
            i, j = 0, 0
            while i < len(vp) and j < len(wp):
                pa, pb = vp[i][0], wp[j][0]
                if pa < pb:
                    i += 1
                elif pa > pb:
                    j += 1
                else:
                    if pa > min_pid:
                        return pa
                    i += 1
                    j += 1
            return -1

        def find_first_panel_with_both_excluded(v, w, excluded):
            """Find first panel containing both v and w, not in excluded set.
            O(deg(v) + deg(w))."""
            vp = vertex_to_panels.get(v, [])
            wp = vertex_to_panels.get(w, [])
            i, j = 0, 0
            while i < len(vp) and j < len(wp):
                pa, pb = vp[i][0], wp[j][0]
                if pa < pb:
                    i += 1
                elif pa > pb:
                    j += 1
                else:
                    if pa not in excluded:
                        return pa
                    i += 1
                    j += 1
            return -1

        def check_full_in(vertex, accompany_pres, accompany_crt):
            """Check if vertex + accompany_pres + accompany_crt are in the same
            panel with non-BORDER crease types. Matches original: finds FIRST
            panel (by id) containing all 3 points, checks its crease types.
            O(deg(vertex) + deg(accompany_pres) + deg(accompany_crt))."""
            vp_set = {p for p, _ in vertex_to_panels.get(vertex, [])}
            if not vp_set:
                return False
            pp_set = {p for p, _ in vertex_to_panels.get(accompany_pres, [])}
            if not pp_set:
                return False
            cp_set = {p for p, _ in vertex_to_panels.get(accompany_crt, [])}
            if not cp_set:
                return False
            common = vp_set & pp_set & cp_set
            if not common:
                return False
            first_panel = min(common)
            idx = self.indices[first_panel]
            ct = self.indices_crease_type[first_panel]
            pos = idx.index(vertex)
            n = len(idx)
            return ct[pos] != BORDER and ct[(pos - 1 + n) % n] != BORDER

        # ---- Phase 2: Process shared BORDER edges ----
        replacement_buffer = []  # [panel_id, position, new_kp_id]
        replacement_set = set()  # (panel_id, position) for O(1) lookup
        created_kps = []
        created_mass_ref = []

        for i in range(F):
            current_indice = self.indices[i]
            current_crease_type_list = self.indices_crease_type[i]
            kp_num = len(current_indice)

            for j in range(kp_num):
                current_crease_type = current_crease_type_list[j]
                if current_crease_type != BORDER:
                    continue

                current_kp_id = current_indice[j]
                next_kp_id = current_indice[(j + 1) % kp_num]

                # Find first panel i2 > i containing both endpoints (inverted index)
                i2 = find_first_panel_with_both(current_kp_id, next_kp_id, i)
                if i2 < 0:
                    continue

                other_indice = self.indices[i2]
                kp_num2 = len(other_indice)
                other_crease_type_list = self.indices_crease_type[i2]
                indice_position = other_indice.index(next_kp_id)

                # Set up chain tracing variables (same as original)
                preserved_id = [current_kp_id, next_kp_id]
                preserved_id_accompany = [
                    current_indice[(j - 1 + kp_num) % kp_num],
                    current_indice[(j + 2 + kp_num) % kp_num]
                ]
                preserved_crease_type = [
                    current_crease_type_list[(j - 1 + kp_num) % kp_num],
                    current_crease_type_list[(j + 1 + kp_num) % kp_num]
                ]

                created_id_accompany = [
                    other_indice[(indice_position + 2 + kp_num2) % kp_num2],
                    other_indice[(indice_position - 1 + kp_num2) % kp_num2]
                ]
                created_crease_type = [
                    other_crease_type_list[(indice_position + 1 + kp_num2) % kp_num2],
                    other_crease_type_list[(indice_position - 1 + kp_num2) % kp_num2]
                ]

                backup_end_flag = [0, 0, 0, 0]
                end_flag = [0, 0, 0, 0]

                # Use sets for O(1) membership check during tracing
                preserved_ids_1 = {i}
                preserved_ids_2 = {i}
                replace_ids_1_set = {i2}
                replace_ids_2_set = {i2}
                # Keep ordered lists for replacement recording
                replace_ids_1_list = [i2]
                replace_ids_2_list = [i2]

                max_iterations = 1000
                iteration_count = 0
                steps = [0, 0, 0, 0]

                while 0 in end_flag:
                    # con-stop condition
                    if preserved_id_accompany[0] == created_id_accompany[0] and \
                       created_crease_type[0] != BORDER and preserved_crease_type[0] != BORDER:
                        replace_ids_1_set.clear()
                        replace_ids_1_list.clear()
                        end_flag[0] = end_flag[2] = 1
                    if preserved_id_accompany[1] == created_id_accompany[1] and \
                       created_crease_type[1] != BORDER and preserved_crease_type[1] != BORDER:
                        replace_ids_2_set.clear()
                        replace_ids_2_list.clear()
                        end_flag[1] = end_flag[3] = 1

                    # full-in condition (using inverted index)
                    if check_full_in(preserved_id[0], preserved_id_accompany[0], created_id_accompany[0]):
                        replace_ids_1_set.clear()
                        replace_ids_1_list.clear()
                        end_flag[0] = end_flag[2] = 1
                    if check_full_in(preserved_id[1], preserved_id_accompany[1], created_id_accompany[1]):
                        replace_ids_2_set.clear()
                        replace_ids_2_list.clear()
                        end_flag[1] = end_flag[3] = 1

                    # single stop
                    if preserved_crease_type[0] == BORDER:
                        end_flag[0] = 1
                    if preserved_crease_type[1] == BORDER:
                        end_flag[1] = 1
                    if created_crease_type[0] == BORDER:
                        end_flag[2] = 1
                    if created_crease_type[1] == BORDER:
                        end_flag[3] = 1

                    # roll (only if end_flag hasn't changed since last iteration)
                    if end_flag == backup_end_flag:
                        steps = [0, 0, 0, 0]

                        # Chain 0: preserved_ids_1 (vertex preserved_id[0], prev direction)
                        if not end_flag[0] and not steps[0]:
                            i3 = find_first_panel_with_both_excluded(
                                preserved_id[0], preserved_id_accompany[0], preserved_ids_1)
                            if i3 >= 0:
                                kp_num3 = len(self.indices[i3])
                                preserved_ids_1.add(i3)
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

                        # Chain 1: preserved_ids_2 (vertex preserved_id[1], next direction)
                        if not end_flag[1] and not steps[1]:
                            i3 = find_first_panel_with_both_excluded(
                                preserved_id[1], preserved_id_accompany[1], preserved_ids_2)
                            if i3 >= 0:
                                kp_num3 = len(self.indices[i3])
                                preserved_ids_2.add(i3)
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

                        # Chain 2: replace_ids_1 (vertex preserved_id[0], created side)
                        if not end_flag[2] and not steps[2]:
                            i3 = find_first_panel_with_both_excluded(
                                preserved_id[0], created_id_accompany[0], replace_ids_1_set)
                            if i3 >= 0:
                                kp_num3 = len(self.indices[i3])
                                replace_ids_1_set.add(i3)
                                replace_ids_1_list.append(i3)
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

                        # Chain 3: replace_ids_2 (vertex preserved_id[1], created side)
                        if not end_flag[3] and not steps[3]:
                            i3 = find_first_panel_with_both_excluded(
                                preserved_id[1], created_id_accompany[1], replace_ids_2_set)
                            if i3 >= 0:
                                kp_num3 = len(self.indices[i3])
                                replace_ids_2_set.add(i3)
                                replace_ids_2_list.append(i3)
                                position = self.indices[i3].index(preserved_id[1])
                                previous_id = self.indices[i3][((position - 1) + kp_num3) % kp_num3]
                                next_id = self.indices[i3][((position + 1) + kp_num3) % kp_num3]
                                if previous_id == created_id_accompany[1]:
                                    created_id_accompany[1] = next_id
                                    created_crease_type[1] = self.indices_crease_type[i3][((position) + kp_num3) % kp_num3]
                                else:
                                    created_id_accompany[1] = previous_id
                                    created_crease_type[1] = self.indices_crease_type[i3][((position - 1) + kp_num3) % kp_num3]
                                if created_crease_type[1] == BORDER:
                                    end_flag[3] = 1
                                steps[3] = 1

                        # no progress -> stop (only when roll actually executed)
                        if steps[0] == 0 and end_flag[0] == 0:
                            end_flag[0] = 1
                        if steps[1] == 0 and end_flag[1] == 0:
                            end_flag[1] = 1
                        if steps[2] == 0 and end_flag[2] == 0:
                            end_flag[2] = 1
                        if steps[3] == 0 and end_flag[3] == 0:
                            end_flag[3] = 1

                    backup_end_flag = end_flag[:]

                    iteration_count += 1
                    if iteration_count >= max_iterations:
                        print(f"[Warning] optimizeIndices_v2: 达到最大迭代次数 {max_iterations}，强制退出循环")
                        print(f"  当前单元: {i}, 配对单元: {i2}")
                        print(f"  end_flag: {end_flag}, steps: {steps}")
                        break

                # Deduplicate replace_ids_1_list against replacement_set
                filtered_1 = []
                for pid in replace_ids_1_list:
                    pos = self.indices[pid].index(preserved_id[0])
                    if (pid, pos) not in replacement_set:
                        filtered_1.append(pid)

                # Deduplicate replace_ids_2_list against replacement_set
                filtered_2 = []
                for pid in replace_ids_2_list:
                    pos = self.indices[pid].index(preserved_id[1])
                    if (pid, pos) not in replacement_set:
                        filtered_2.append(pid)

                if len(filtered_1) and all(pid > i for pid in filtered_1):
                    created_kps.append(deepcopy(self.kps[preserved_id[0]]))
                    created_mass_ref.append(preserved_id[0])
                    new_kp_id = len(self.kps) + len(created_kps) - 1
                    for pid in filtered_1:
                        pos = self.indices[pid].index(preserved_id[0])
                        replacement_buffer.append([pid, pos, new_kp_id])
                        replacement_set.add((pid, pos))

                if len(filtered_2) and all(pid > i for pid in filtered_2):
                    created_kps.append(deepcopy(self.kps[preserved_id[1]]))
                    created_mass_ref.append(preserved_id[1])
                    new_kp_id = len(self.kps) + len(created_kps) - 1
                    for pid in filtered_2:
                        pos = self.indices[pid].index(preserved_id[1])
                        replacement_buffer.append([pid, pos, new_kp_id])
                        replacement_set.add((pid, pos))

        # ---- Phase 3: Apply replacements ----
        for ele in replacement_buffer:
            self.indices[ele[0]][ele[1]] = ele[2]

        # ---- Phase 4: Compute new_kp_origami_id ----
        self.new_kp_origami_id = []
        seen_kp_section = set()
        for ele in replacement_buffer:
            unit_id = ele[0]
            kp_id = ele[2]
            for k in range(len(self.split_unit_list)):
                if self.split_unit_list[k] <= unit_id and (k == len(self.split_unit_list) - 1 or self.split_unit_list[k + 1] > unit_id):
                    break
            key = (kp_id, k)
            if key not in seen_kp_section:
                seen_kp_section.add(key)
                self.new_kp_origami_id.append([kp_id, k])

        # ---- Phase 5: Redistribute mass ----
        mass_ref_count = Counter(created_mass_ref)
        mass_ref_indices = defaultdict(list)
        for j, v in enumerate(created_mass_ref):
            mass_ref_indices[v].append(j)

        created_mass = [0.0] * len(created_mass_ref)
        original_kp_len = len(self.kps)
        self.kps += created_kps
        for orig_v, count in mass_ref_count.items():
            portion = count + 1
            final_mass = self.mass_list[orig_v] / portion
            self.mass_list[orig_v] = final_mass
            for j in mass_ref_indices[orig_v]:
                created_mass[j] = final_mass
        self.mass_list += created_mass

        # ---- Phase 6: Rebuild line_indices (with set-based deduplication) ----
        self.line_indices.clear()
        seen_edges = set()
        for i1 in range(len(self.unit_list)):
            unit = self.unit_list[i1]
            n = len(self.indices[i1])
            for i in range(n):
                next_i = (i + 1) % n
                linetype = unit.crease[i].getType()
                if unit.crease[i].hard:
                    linetype = 3
                indice1 = self.indices[i1][i]
                indice2 = self.indices[i1][next_i]
                edge_key = (min(indice1, indice2), max(indice1, indice2))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    self.line_indices.append([[indice1, indice2], linetype,
                        unit.crease[i].folding_angle_upper_bound,
                        unit.crease[i].folding_angle_lower_bound])
                    
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
                self.spatialhash.insert(kp, len(self.kps) - 1)
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
        # self.optimizeIndices()
        self.optimizeIndices_v2()
        self.tri_indices.clear()
        self.tri_indices_ref.clear()
        kp_len = len(self.kps)
        self.connection_matrix = [[0] * kp_len for _ in range(kp_len)]
        # self.origin_distance_matrix = [[0] * kp_len for _ in range(kp_len)]

        # #origin distance
        # for i in range(kp_len):
        #     for j in range(kp_len):
        #         self.origin_distance_matrix[i][j] = self.distance(self.kps[i], self.kps[j])
        
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

        # Build adjacency sets and edge-to-triangle index for O(1) lookup
        # Replaces O(N) connection_matrix row scan with O(min(deg)) set intersection
        adjacency = [set() for _ in range(kp_len)]
        edge_to_tris = {}
        for j in range(len(self.tri_indices) // 3):
            a = self.tri_indices[3 * j]
            b = self.tri_indices[3 * j + 1]
            c = self.tri_indices[3 * j + 2]
            adjacency[a].add(b); adjacency[b].add(a)
            adjacency[b].add(c); adjacency[c].add(b)
            adjacency[a].add(c); adjacency[c].add(a)
            for e in (frozenset((a, b)), frozenset((b, c)), frozenset((a, c))):
                if e not in edge_to_tris:
                    edge_to_tris[e] = []
                edge_to_tris[e].append(j)

        #bending force
        for i in range(len(self.line_indices)):
            line_start_indice = self.line_indices[i][0][0]
            line_end_indice = self.line_indices[i][0][1]
            line_type = self.line_indices[i][1]

            relevant_kp = sorted(adjacency[line_start_indice] & adjacency[line_end_indice])

            if len(relevant_kp) == 2 and [line_end_indice, line_start_indice] not in self.crease_pairs:
                crease_pair = [line_start_indice, line_end_indice]
                result = 0
                for j in edge_to_tris.get(frozenset((line_start_indice, line_end_indice)), []):
                    tri_index = [self.tri_indices[3 * j], self.tri_indices[3 * j + 1], self.tri_indices[3 * j + 2]]
                    if relevant_kp[0] in tri_index:
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

            relevant_kp = sorted(adjacency[line_start_indice] & adjacency[line_end_indice])
            
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