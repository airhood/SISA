from SISA import IsomerGenerator, IsomerDFTAnalyzer

def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def run_SISA(formula, optim, verbose=True):
    vprint(verbose, "### 이성질체 생성 ###\n")
    
    gen = IsomerGenerator(formula)
    isomers = gen.generate_all()
    
    vprint(verbose, "\n" + "="*80 + "\n")
    
    vprint(verbose, "### DFT 계산 ###\n")
    analyzer = IsomerDFTAnalyzer(
        smiles_list=isomers,
        method='B3LYP',
        basis='6-31G(d)'
    )
    
    analyzer.calculate_all(optimize=optim)
    
    vprint(verbose, "\n" + "="*80 + "\n")
    
    vprint(verbose, "### 결과 분석 ###\n")
    
    analyzer.print_summary()
    analyzer.visualize_energy_diagram()
    analyzer.visualize_homo_lumo_diagram()
    
    analyzer.export_results('c4h10_dft_results.csv')
    
def main():
    formula = input("분자식: ")
    optim = input("구조 최적화 (Y/N): ")
    optim.capitalize()
    if optim == "Y":
        optim = True
    elif optim == "N":
        optim = False
    else:
        optim = True
    run_SISA(formula.strip(), optim)
    
if __name__ == "__main__":
    main()
