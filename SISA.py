import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import itertools
import psi4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psi4


class IsomerGenerator:
    def __init__(self, formula, verbose=True):
        self.verbose = verbose
        self.formula = formula
        self.parse_formula()
        self.dbe = self.calculate_dbe()
        self.isomers = []
        
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def parse_formula(self):
        import re
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, self.formula)

        self.atom_counts = {}
        for element, count in matches:
            self.atom_counts[element] = int(count) if count else 1

        self._vprint(f"분자식: {self.formula}")
        self._vprint(f"원자 구성: {self.atom_counts}")

    def calculate_dbe(self):
        """포화도 계산"""
        C = self.atom_counts.get('C', 0)
        H = self.atom_counts.get('H', 0)
        N = self.atom_counts.get('N', 0)
        X = sum(self.atom_counts.get(x, 0) for x in ['F', 'Cl', 'Br', 'I'])

        dbe = (2*C + 2 + N - H - X) / 2

        self._vprint(f"포화도 (DBE) = {dbe}")

        if dbe < 0:
            raise ValueError("유효하지 않은 분자식")

        return int(dbe)

    def enumerate_unsaturation_types(self):
        """
        포화도를 이중결합, 삼중결합, 고리로 분배하는 모든 방법
        """
        combinations = []

        for n_triple in range(self.dbe // 2 + 1):
            remaining = self.dbe - 2 * n_triple

            for n_rings in range(remaining + 1):
                n_double = remaining - n_rings

                if n_double >= 0:
                    combinations.append({
                        'double': n_double,
                        'triple': n_triple,
                        'rings': n_rings
                    })

        self._vprint(f"\n포화도 분배 방법: {len(combinations)}가지")
        for i, combo in enumerate(combinations, 1):
            self._vprint(f"  {i}. 이중결합={combo['double']}, 삼중결합={combo['triple']}, 고리={combo['rings']}")

        return combinations

    def generate_skeletons_with_rings(self, n_carbons, n_rings):
        """
        고리를 포함한 탄소 골격 생성
        """
        self._vprint(f"\n  골격 생성: C={n_carbons}, 고리={n_rings}")
        
        if n_rings == 0:
            # 트리 구조
            from networkx.generators.nonisomorphic_trees import nonisomorphic_trees
            skeletons = list(nonisomorphic_trees(n_carbons))
            
            valid_skeletons = []
            for tree in skeletons:
                max_degree = max(dict(tree.degree()).values()) if tree.number_of_nodes() > 0 else 0
                if max_degree <= 4:
                    for node in tree.nodes():
                        tree.nodes[node]['element'] = 'C'
                    valid_skeletons.append(tree)
            
            self._vprint(f"    트리: {len(valid_skeletons)}개")
            return valid_skeletons
        
        elif n_rings == 1:
            # 단일 고리
            return self._generate_single_ring_skeletons(n_carbons)
        
        else:
            # 다중 고리
            return self._generate_multi_ring_skeletons(n_carbons, n_rings)

    def _generate_single_ring_skeletons(self, n_carbons):
        """
        하나의 고리를 포함한 골격
        """
        skeletons = []

        # 최소 고리 크기는 3
        for ring_size in range(3, min(n_carbons + 1, 8)):  # 큰 고리는 제한
            # 고리 생성
            ring = nx.cycle_graph(ring_size)

            # 원소 레이블
            for node in ring.nodes():
                ring.nodes[node]['element'] = 'C'

            remaining = n_carbons - ring_size

            if remaining == 0:
                # 고리만
                skeletons.append(ring)
            elif remaining > 0:
                # 나머지 탄소를 붙이기
                for attach_node in ring.nodes():
                    if ring.degree(attach_node) < 4:
                        g = ring.copy()

                        # 체인 붙이기
                        chain_start = max(g.nodes()) + 1
                        g.add_edge(attach_node, chain_start)
                        g.nodes[chain_start]['element'] = 'C'

                        for i in range(1, remaining):
                            prev = chain_start + i - 1
                            curr = chain_start + i
                            g.add_edge(prev, curr)
                            g.nodes[curr]['element'] = 'C'

                        skeletons.append(g)

        self._vprint(f"    고리 골격: {len(skeletons)}개")
        return skeletons

    def place_multiple_bonds(self, skeleton, n_double, n_triple):
        """
        골격에 이중/삼중결합 배치
        """
        if n_double == 0 and n_triple == 0:
            # 다중결합 없음
            g = skeleton.copy()
            nx.set_edge_attributes(g, 1, 'bond_order')
            return [g]

        structures = []
        edges = list(skeleton.edges())

        # 이중결합 위치 선택
        if n_double > 0:
            for double_edges in itertools.combinations(edges, n_double):
                # 삼중결합 위치
                remaining = [e for e in edges if e not in double_edges]

                if n_triple > 0:
                    for triple_edges in itertools.combinations(remaining, n_triple):
                        g = self._create_multigraph(skeleton, double_edges, triple_edges)
                        if g and self._check_valence(g):
                            structures.append(g)
                else:
                    g = self._create_multigraph(skeleton, double_edges, [])
                    if g and self._check_valence(g):
                        structures.append(g)
        else:
            # 삼중결합만
            for triple_edges in itertools.combinations(edges, n_triple):
                g = self._create_multigraph(skeleton, [], triple_edges)
                if g and self._check_valence(g):
                    structures.append(g)

        return structures

    def _create_multigraph(self, skeleton, double_edges, triple_edges):
        """다중결합을 포함한 그래프 생성"""
        g = skeleton.copy()

        # 기본 단일결합
        nx.set_edge_attributes(g, 1, 'bond_order')

        # 이중결합
        for edge in double_edges:
            g.edges[edge]['bond_order'] = 2

        # 삼중결합
        for edge in triple_edges:
            g.edges[edge]['bond_order'] = 3

        return g

    def _check_valence(self, graph):
        """원자가 확인"""
        valence_max = {'C': 4, 'O': 2, 'N': 3, 'H': 1}

        for node in graph.nodes():
            element = graph.nodes[node].get('element', 'C')
            max_val = valence_max.get(element, 4)

            # 총 결합 차수
            total_bonds = 0
            for neighbor in graph.neighbors(node):
                bond_order = graph.edges[node, neighbor].get('bond_order', 1)
                total_bonds += bond_order

            if total_bonds > max_val:
                return False

        return True

    def add_hydrogens(self, structures):
        """수소 추가"""
        completed = []
        valence_max = {'C': 4, 'O': 2, 'N': 3}

        for struct in structures:
            g = struct.copy()
            h_idx = max(g.nodes()) + 1 if g.number_of_nodes() > 0 else 0

            for node in list(g.nodes()):
                element = g.nodes[node].get('element', 'C')

                if element == 'H':
                    continue

                # 현재 결합 차수 합
                total_bonds = 0
                for neighbor in g.neighbors(node):
                    bond_order = g.edges[node, neighbor].get('bond_order', 1)
                    total_bonds += bond_order

                # 필요한 수소
                max_val = valence_max.get(element, 4)
                needed_h = max_val - total_bonds

                # 수소 추가
                for _ in range(needed_h):
                    g.add_node(h_idx, element='H')
                    g.add_edge(node, h_idx)
                    g.edges[node, h_idx]['bond_order'] = 1
                    h_idx += 1

            # 수소 개수 확인
            h_count = sum(1 for n in g.nodes() if g.nodes[n]['element'] == 'H')
            expected_h = self.atom_counts.get('H', 0)

            if h_count == expected_h:
                completed.append(g)

        return completed

    def graph_to_smiles(self, graph):
        """그래프를 SMILES로 변환"""
        try:
            mol = Chem.RWMol()
            node_to_idx = {}

            # 원자 추가
            for node in graph.nodes():
                element = graph.nodes[node]['element']
                atom = Chem.Atom(element)
                idx = mol.AddAtom(atom)
                node_to_idx[node] = idx

            # 결합 추가
            for u, v in graph.edges():
                bond_order = graph.edges[u, v].get('bond_order', 1)

                if bond_order == 1:
                    bond_type = Chem.BondType.SINGLE
                elif bond_order == 2:
                    bond_type = Chem.BondType.DOUBLE
                elif bond_order == 3:
                    bond_type = Chem.BondType.TRIPLE
                else:
                    bond_type = Chem.BondType.SINGLE

                mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol)
        except:
            return None

    def generate_all(self):
        self._vprint(f"\n{'='*60}")
        self._vprint(f"  고급 이성질체 생성: {self.formula}")
        self._vprint(f"{'='*60}\n")

        # 포화도 분배
        combos = self.enumerate_unsaturation_types()

        all_structures = []
        n_C = self.atom_counts.get('C', 0)

        # 각 분배에 대해 구조 생성
        for combo in combos:
            self._vprint(f"\n--- 조합: 이중={combo['double']}, 삼중={combo['triple']}, 고리={combo['rings']} ---")

            # 골격 생성
            skeletons = self.generate_skeletons_with_rings(n_C, combo['rings'])

            # 다중결합 배치
            for skeleton in skeletons:
                structures = self.place_multiple_bonds(
                    skeleton,
                    combo['double'],
                    combo['triple']
                )
                all_structures.extend(structures)

        self._vprint(f"\n다중결합 배치 후: {len(all_structures)}개 구조")

        # 수소 추가
        with_hydrogen = self.add_hydrogens(all_structures)
        self._vprint(f"수소 추가 후: {len(with_hydrogen)}개 구조")

        # SMILES 변환 및 중복 제거
        unique_smiles = set()
        for struct in with_hydrogen:
            smiles = self.graph_to_smiles(struct)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical = Chem.MolToSmiles(mol)
                    unique_smiles.add(canonical)

        self.isomers = sorted(list(unique_smiles))

        self._vprint(f"\n{'='*60}")
        self._vprint(f"최종 이성질체: {len(self.isomers)}개")
        self._vprint(f"{'='*60}\n")

        for i, smi in enumerate(self.isomers, 1):
            self._vprint(f"  {i}. {smi}")

        return self.isomers

    def visualize(self):
        if not self.isomers:
            self._vprint("생성된 이성질체가 없습니다.")
            return

        mols = [Chem.MolFromSmiles(s) for s in self.isomers]

        for mol in mols:
            AllChem.Compute2DCoords(mol)

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(250, 250),
            legends=self.isomers,
            returnPNG=False
        )

        img.save("isomers.png")
        img.show()
        
    def _generate_multi_ring_skeletons(self, n_carbons, n_rings):
        """
        여러 개의 고리를 포함한 골격 생성
        """
        skeletons = []
        
        self._vprint(f"    다중 고리 생성: {n_rings}개 고리")
        
        # 고리 크기 조합 생성
        ring_size_combos = self._generate_ring_size_combinations(n_carbons, n_rings)
        
        self._vprint(f"      가능한 고리 조합: {len(ring_size_combos)}가지")
        
        for combo in ring_size_combos[:10]:  # 처음 10개만 (시간 제한)
            self._vprint(f"      처리 중: {combo}")
            
            # 각 조합에 대해 연결 방식 시도
            
            # 융합 고리 (Fused rings)
            fused = self._create_fused_rings(combo)
            if fused:
                skeletons.extend(fused)
            
            # 독립 고리 (Spiro rings)
            spiro = self._create_spiro_rings(combo) 
            if spiro:
                skeletons.extend(spiro)
            
            # 가교 고리 (Bridged rings)
            if n_rings == 2:  # 가교는 2개 고리일 때만
                bridged = self._create_bridged_rings(combo)
                if bridged:
                    skeletons.extend(bridged)
        
        self._vprint(f"    다중 고리 골격: {len(skeletons)}개")
        return skeletons

    def _generate_ring_size_combinations(self, n_carbons, n_rings):
        """
        n개 고리를 만들 수 있는 고리 크기 조합
        """
        combos = []
        
        # 최소 고리 크기 3, 최대 6 (큰 고리는 불안정)
        min_ring_size = 3
        max_ring_size = 6
        
        def backtrack(remaining_carbons, remaining_rings, current_combo):
            if remaining_rings == 0:
                if remaining_carbons >= 0:
                    combos.append(current_combo[:])
                return
            
            # 융합 고리는 1개 탄소 공유, 스피로는 1개 공유
            min_needed = remaining_rings * min_ring_size - (remaining_rings - 1)
            
            if remaining_carbons < min_needed:
                return
            
            for size in range(min_ring_size, min(max_ring_size + 1, remaining_carbons + 1)):
                current_combo.append(size)
                
                # 융합이면 1개 공유
                shared = 1 if len(current_combo) > 1 else 0
                
                backtrack(remaining_carbons - size + shared, remaining_rings - 1, current_combo)
                current_combo.pop()
        
        backtrack(n_carbons, n_rings, [])
        
        # 중복 제거 및 정렬
        unique_combos = []
        seen = set()
        
        for combo in combos:
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                unique_combos.append(list(key))
        
        return unique_combos

    def _create_fused_rings(self, ring_sizes):
        """
        융합 고리 생성
        """
        if len(ring_sizes) != 2:
            return []  # 일단 2개 고리만
        
        size1, size2 = ring_sizes
        skeletons = []
        
        # 첫 번째 고리
        ring1 = nx.cycle_graph(size1)
        
        # 두 번째 고리를 첫 번째 고리의 간선에 융합
        for edge in ring1.edges():
            g = ring1.copy()
            
            u, v = edge
            
            if size2 == 3:
                # 삼각형 융합
                # u-v를 공유, 새 노드 1개 추가
                new_node = max(g.nodes()) + 1
                g.add_edge(u, new_node)
                g.add_edge(new_node, v)
                
            elif size2 == 4:
                # 사각형 융합
                # u-v를 공유, 새 노드 2개 추가
                n1 = max(g.nodes()) + 1
                n2 = n1 + 1
                g.add_edge(u, n1)
                g.add_edge(n1, n2)
                g.add_edge(n2, v)
                
            elif size2 == 5:
                # 오각형 융합
                n1 = max(g.nodes()) + 1
                n2 = n1 + 1
                n3 = n1 + 2
                g.add_edge(u, n1)
                g.add_edge(n1, n2)
                g.add_edge(n2, n3)
                g.add_edge(n3, v)
                
            elif size2 == 6:
                # 육각형 융합
                n1 = max(g.nodes()) + 1
                n2 = n1 + 1
                n3 = n1 + 2
                n4 = n1 + 3
                g.add_edge(u, n1)
                g.add_edge(n1, n2)
                g.add_edge(n2, n3)
                g.add_edge(n3, n4)
                g.add_edge(n4, v)
            
            else:
                continue
            
            # 원소 레이블
            for node in g.nodes():
                if 'element' not in g.nodes[node]:
                    g.nodes[node]['element'] = 'C'
            
            skeletons.append(g)
        
        return skeletons

    def _create_spiro_rings(self, ring_sizes):
        """
        스피로 고리 생성
        """
        if len(ring_sizes) != 2:
            return []
        
        size1, size2 = ring_sizes
        skeletons = []
        
        # 첫 번째 고리
        ring1 = nx.cycle_graph(size1)
        
        # 각 노드를 중심으로 시도
        for spiro_center in ring1.nodes():
            g = ring1.copy()
            
            start_node = max(g.nodes()) + 1
            
            prev = spiro_center
            for i in range(size2 - 1):
                curr = start_node + i
                g.add_edge(prev, curr)
                g.nodes[curr]['element'] = 'C'
                prev = curr
            
            # 마지막 노드를 spiro_center에 연결
            g.add_edge(prev, spiro_center)
            
            skeletons.append(g)
        
        return skeletons

    def _create_bridged_rings(self, ring_sizes):
        """
        가교 고리 생성
        """
        if len(ring_sizes) != 2:
            return []
        
        size1, size2 = ring_sizes
        
        if size1 < 3 or size2 < 2:
            return []
        
        skeletons = []
        
        # 기본 고리
        ring = nx.cycle_graph(size1)
        for node in ring.nodes():
            ring.nodes[node]['element'] = 'C'
        
        nodes = list(ring.nodes())
        
        # 고리의 두 노드를 가교로 연결
        for i in range(len(nodes)):
            for j in range(i + 2, len(nodes)):
                # 인접 노드 제외
                if j - i < 2 or len(nodes) - j + i < 2:
                    continue
                
                g = ring.copy()
                
                start = nodes[i]
                end = nodes[j]
                
                # 가교 경로의 중간 노드 개수
                n_bridge_nodes = size2 - 2  # 시작/끝 제외
                
                if n_bridge_nodes < 0:
                    continue
                
                if n_bridge_nodes == 0:
                    continue
                
                prev_node = start
                
                for k in range(n_bridge_nodes):
                    # 새 노드 생성
                    new_node = max(g.nodes()) + 1
                    g.add_node(new_node, element='C')
                    
                    # 이전 노드와 연결
                    g.add_edge(prev_node, new_node)
                    
                    # 다음 연결을 위해 업데이트
                    prev_node = new_node
                
                # 마지막 노드를 끝점과 연결
                g.add_edge(prev_node, end)
                
                if g.number_of_nodes() <= 15:
                    skeletons.append(g)
        
        return skeletons


class IsomerDFTAnalyzer:
    def __init__(self, smiles_list, method='B3LYP', basis='6-31G(d)', verbose=True):
        self.smiles_list = smiles_list
        self.method = method
        self.basis = basis
        
        self.results = []
        
        self.verbose = verbose
        
        # Psi4 설정
        psi4.set_memory('2 GB')
        psi4.set_num_threads(4)
        psi4.core.set_output_file('psi4_output.dat', False)
        
        print(f"DFT 설정: {method}/{basis}")
        print(f"이성질체 개수: {len(smiles_list)}")
        
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def smiles_to_psi4_geometry(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # 3D 좌표 생성
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        
        geom_str = "0 1\n"
        
        for i, atom in enumerate(atoms):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            geom_str += f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n"
        
        return geom_str
    
    def calculate_single_isomer(self, smiles, optimize=True):
        self._vprint(f"\n계산 중: {smiles}")
        
        try:
            # Psi4 분자 객체
            geom_str = self.smiles_to_psi4_geometry(smiles)
            mol = psi4.geometry(geom_str)
            
            psi4.set_options({
                'basis': self.basis,
                'scf_type': 'df',
                'reference': 'rhf'
            })
            
            # 구조 최적화 + 에너지
            if optimize:
                energy, wfn = psi4.optimize(
                    f'{self.method}/{self.basis}', 
                    molecule=mol,
                    return_wfn=True  # 파동 함수
                )
            else:
                energy, wfn = psi4.energy(
                    f'{self.method}/{self.basis}', 
                    molecule=mol,
                    return_wfn=True
                )
            
            # HOMO-LUMO
            eps_a = wfn.epsilon_a()  # Alpha 오비탈 에너지
            homo_idx = wfn.nalpha() - 1
            lumo_idx = wfn.nalpha()
            
            eps_array = np.array([eps_a.get(i) for i in range(eps_a.dim())])
            
            homo_energy = eps_array[homo_idx] * 27.211  # Hartree → eV
            lumo_energy = eps_array[lumo_idx] * 27.211
            gap = lumo_energy - homo_energy
            
            # 쌍극자 모멘트
            try:
                dipole_x = wfn.variable('CURRENT DIPOLE X')
                dipole_y = wfn.variable('CURRENT DIPOLE Y')
                dipole_z = wfn.variable('CURRENT DIPOLE Z')
                dipole_magnitude = np.sqrt(dipole_x**2 + dipole_y**2 + dipole_z**2)
            except:
                dipole_magnitude = 0.0
            
            result = {
                'smiles': smiles,
                'energy_hartree': energy,
                'energy_kcal': energy * 627.509,
                'homo_eV': homo_energy,
                'lumo_eV': lumo_energy,
                'gap_eV': gap,
                'dipole_debye': dipole_magnitude
            }
            
            self._vprint(f"  ✓ Complete: E = {energy:.6f} Hartree")
            self._vprint(f"    HOMO = {homo_energy:.3f} eV, LUMO = {lumo_energy:.3f} eV, Gap = {gap:.3f} eV")
            
            return result
            
        except Exception as e:
            self._vprint(f"  ✗ 오류: {str(e)}")
            return None
    
    def calculate_all(self, optimize=True):
        self._vprint(f"\n{'='*60}")
        self._vprint(f"  Start DFT Calculation")
        self._vprint(f"{'='*60}\n")
        
        self.results = []
        
        for i, smiles in enumerate(self.smiles_list, 1):
            self._vprint(f"[{i}/{len(self.smiles_list)}]", end=" ")
            
            result = self.calculate_single_isomer(smiles, optimize)
            
            if result:
                self.results.append(result)
        
        self._vprint(f"\n{'='*60}")
        self._vprint(f"  Calculation Complete: {len(self.results)}/{len(self.smiles_list)} success")
        self._vprint(f"{'='*60}\n")
        
        # 상대 에너지
        if self.results:
            min_energy = min(r['energy_kcal'] for r in self.results)
            
            for result in self.results:
                result['relative_energy_kcal'] = result['energy_kcal'] - min_energy
        
        return self.results
    
    def create_dataframe(self):
        if not self.results:
            print("There are no calculation results.")
            return None
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('relative_energy_kcal')
        df = df.reset_index(drop=True)
        df.index += 1
        
        return df
    
    def print_summary(self):
        df = self.create_dataframe()
        
        if df is None:
            return
        
        self._vprint("\n" + "="*80)
        self._vprint("  이성질체 안정도 분석 결과")
        self._vprint("="*80 + "\n")
        
        display_df = df[[
            'smiles', 
            'relative_energy_kcal', 
            'gap_eV', 
            'dipole_debye'
        ]].copy()
        
        display_df.columns = [
            'SMILES', 
            'ΔE (kcal/mol)', 
            'HOMO-LUMO Gap (eV)', 
            'Dipole (Debye)'
        ]
        
        display_df['ΔE (kcal/mol)'] = display_df['ΔE (kcal/mol)'].map('{:.2f}'.format)
        display_df['HOMO-LUMO Gap (eV)'] = display_df['HOMO-LUMO Gap (eV)'].map('{:.3f}'.format)
        display_df['Dipole (Debye)'] = display_df['Dipole (Debye)'].map('{:.3f}'.format)
        
        self._vprint(display_df.to_string())
        
        self._vprint("\n" + "="*80)
        self._vprint("\n[ ===== 해석 ===== ]")
        self._vprint(f"  - Most Stable: {df.iloc[0]['smiles']}")
        self._vprint(f"  - 에너지 범위: 0 ~ {df['relative_energy_kcal'].max():.2f} kcal/mol")
        self._vprint(f"  - 평균 Gap: {df['gap_eV'].mean():.3f} eV")
        
        # large difference in stability
        if df['relative_energy_kcal'].max() > 5:
            self._vprint(f"\n  ⚠️  가장 불안정한 이성질체는 {df['relative_energy_kcal'].max():.2f} kcal/mol 더 높음")
            self._vprint(f"     (실온에서 거의 존재하지 않음)")
        
        self._vprint()
    
    def visualize_energy_diagram(self):
        df = self.create_dataframe()
        
        if df is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 상대 에너지
        x = range(len(df))
        bars = ax1.bar(x, df['relative_energy_kcal'], color='steelblue', alpha=0.7, edgecolor='black')
        
        colors = plt.cm.RdYlGn_r(df['relative_energy_kcal'] / df['relative_energy_kcal'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xlabel('Isomer', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Relative Energy (kcal/mol)', fontsize=12, fontweight='bold')
        ax1.set_title('Relative Stability', fontsize=14, fontweight='bold')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax1.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(df['relative_energy_kcal']):
            ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(range(1, len(df)+1), fontsize=10)
        
        # HOMO-LUMO vs stability
        scatter = ax2.scatter(df['gap_eV'], df['relative_energy_kcal'], 
                   s=150, alpha=0.7, c=df['dipole_debye'], 
                   cmap='viridis', edgecolors='black', linewidth=1.5)
        
        for i, (gap, energy) in enumerate(zip(df['gap_eV'], df['relative_energy_kcal'])):
            ax2.annotate(str(i+1), (gap, energy), 
                        ha='center', va='center', 
                        fontsize=9, fontweight='bold', color='white')
        
        ax2.set_xlabel('HOMO-LUMO Gap (eV)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Energy (kcal/mol)', fontsize=12, fontweight='bold')
        ax2.set_title('Gap vs Stability', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Dipole Moment (Debye)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_homo_lumo_diagram(self):
        df = self.create_dataframe()
        
        if df is None:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x_positions = range(len(df))
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.plot([i-0.35, i+0.35], [row['homo_eV'], row['homo_eV']], 
                   'r-', linewidth=3, label='HOMO' if i == 0 else '', alpha=0.8)
            
            ax.plot([i-0.35, i+0.35], [row['lumo_eV'], row['lumo_eV']], 
                   'b-', linewidth=3, label='LUMO' if i == 0 else '', alpha=0.8)
            
            ax.plot([i, i], [row['homo_eV'], row['lumo_eV']], 
                   'k--', alpha=0.4, linewidth=1.5)
            
            mid = (row['homo_eV'] + row['lumo_eV']) / 2
            ax.text(i, mid, f"{row['gap_eV']:.2f} eV", 
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='yellow', alpha=0.5, edgecolor='black'))
        
        ax.set_xlabel('Isomer', fontsize=13, fontweight='bold')
        ax.set_ylabel('Energy (eV)', fontsize=13, fontweight='bold')
        ax.set_title('HOMO-LUMO Energy Levels', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(range(1, len(df)+1), fontsize=11)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename='isomer_dft_results.csv'):
        df = self.create_dataframe()
        
        if df is not None:
            df.to_csv(filename, index=False)
            self._vprint(f"✓ 결과 저장: {filename}")
